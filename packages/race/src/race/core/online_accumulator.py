"""Online RACE accumulator for real-time evidence aggregation.

This module accumulates evidence during model inference without storing
activations on disk. It keeps only NIG posteriors in memory and releases
activations after computing evidence. Streaming mode processes one sample at a
time.

This is a generic accumulator that works with any model family (CV, VLM, LLM).
"""

from __future__ import annotations

import gc
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from race.core.analyzer import (
    LayerWeights,
    NIGPosterior,
    compute_attn_evidence_batch,
    compute_ffn_evidence_batch,
    init_nig,
)
from race.io.h5_schema import (
    append_axis_members,
    ensure_axis_value,
    ensure_metrics_group,
    ensure_race_meta,
    ensure_reports_group,
    update_total_instances,
)
from race.utils.device import resolve_device as _resolve_device

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_axis_id(key: str) -> int:
    """Convert a string key to a deterministic 6-digit numeric axis ID."""
    digest = hashlib.md5(str(key).encode()).hexdigest()
    return int(digest[:6], 16) % 900000 + 100000


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class OnlineRACEConfig:
    """Configuration for online RACE accumulator."""

    # NIG prior hyperparameters
    mu_0: float = 0.0
    lambda_0: float = 1.0
    alpha_0: float = 1.0
    beta_0: float = 1.0
    gamma: float = 0.05  # CAM confidence level (1-gamma upper t-quantile)
    eps: float = 1e-8
    device: str = "auto"
    model_family: str = "generic"  # 'cv', 'vlm', 'llm', 'generic'


@dataclass
class DomainState:
    """NIG state for a single domain / label group."""

    nig_attn: Dict[int, NIGPosterior] = field(default_factory=dict)
    nig_mlp: Dict[int, NIGPosterior] = field(default_factory=dict)
    # Ablation baseline: accumulate raw activation absolute-value sums per layer
    # and token counts for computing activation_mean = sum_act / n_tokens.
    act_sum_attn: Dict[int, torch.Tensor] = field(default_factory=dict)
    act_n_attn: Dict[int, int] = field(default_factory=dict)
    act_sum_mlp: Dict[int, torch.Tensor] = field(default_factory=dict)
    act_n_mlp: Dict[int, int] = field(default_factory=dict)
    sample_count: int = 0
    correct_count: int = 0
    error_count: int = 0
    instance_indices: list = field(default_factory=list)


@dataclass
class OnlineRACEState:
    """Collection of domain-wise NIG states."""

    domains: Dict[str, DomainState] = field(default_factory=dict)
    domain_metadata: Dict[str, Dict] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Streaming context
# ---------------------------------------------------------------------------


@dataclass
class SampleStreamContext:
    """Per-sample streaming context for incremental evidence accumulation."""

    accumulator: "OnlineRACEAccumulator"
    domain: str
    domain_state: DomainState
    modules: Tuple[str, ...]
    instance_idx: int
    attn_evidence: Dict[int, torch.Tensor] = field(default_factory=dict)
    mlp_evidence: Dict[int, torch.Tensor] = field(default_factory=dict)
    # Raw activation buffers used to compute activation_mean ablation baseline.
    attn_act_buf: Dict[int, torch.Tensor] = field(default_factory=dict)
    mlp_act_buf: Dict[int, torch.Tensor] = field(default_factory=dict)
    active: bool = True

    def consume(
        self,
        activation_type: str,
        layer_idx: int,
        activation: torch.Tensor,
    ) -> None:
        self.accumulator._consume_stream_activation(
            self,
            activation_type=activation_type,
            layer_idx=layer_idx,
            activation=activation,
        )

    def finalize(self, is_correct: bool) -> None:
        self.accumulator._finalize_stream(self, is_correct=is_correct)

    def abort(self) -> None:
        self.accumulator._abort_stream(self)


# ---------------------------------------------------------------------------
# Main accumulator
# ---------------------------------------------------------------------------


class OnlineRACEAccumulator:
    """Online RACE accumulator that processes activations in real-time.

    Works directly with pre-extracted layer weights, accumulating evidence
    into NIG posteriors without storing intermediate activations.

    Usage::

        layer_weights = extract_vit_layer_weights(model, layers=None)
        accumulator = OnlineRACEAccumulator(
            model_name=model_name,
            layer_weights=layer_weights,
            config=OnlineRACEConfig(device="cuda", model_family="cv"),
        )

        for idx, (image, label) in enumerate(dataset):
            activations = record_sample(image)
            accumulator.accumulate_from_activations(
                domain=str(label),
                activations=activations,
                is_correct=(pred == label),
                instance_idx=idx,
            )

        accumulator.save_results("output.h5")
    """

    def __init__(
        self,
        model_name: str,
        layer_weights: Dict[int, LayerWeights],
        config: Optional[OnlineRACEConfig] = None,
    ):
        self.model_name = model_name
        self.config = config or OnlineRACEConfig()
        self.layer_weights = layer_weights

        self.state = OnlineRACEState()
        self.compute_device: torch.device = _resolve_device(self.config.device)
        self.use_torch_backend: bool = self.compute_device.type != "cpu"
        self.runtime_device: str = str(self.compute_device)

        if self.use_torch_backend:
            print(
                f"[Online-RACE] Using device '{self.compute_device}' for evidence accumulation."
            )
        else:
            print("[Online-RACE] Using CPU for evidence accumulation.")

        self._initialize_weight_storage()

    # ------------------------------------------------------------------
    # Weight management
    # ------------------------------------------------------------------

    def _prepare_weight_storage(self, value) -> torch.Tensor:
        tensor = (
            value.detach()
            if isinstance(value, torch.Tensor)
            else torch.as_tensor(value)
        )
        if self.use_torch_backend:
            if tensor.device != self.compute_device:
                tensor = tensor.to(device=self.compute_device)
        else:
            if tensor.device.type != "cpu":
                tensor = tensor.cpu()
            if tensor.dtype != torch.float32:
                tensor = tensor.to(dtype=torch.float32)
        return tensor.contiguous()

    def _initialize_weight_storage(self) -> None:
        for weights in self.layer_weights.values():
            if getattr(weights, "W_O", None) is not None:
                weights.W_O = self._prepare_weight_storage(weights.W_O)
            if getattr(weights, "W_down", None) is not None:
                weights.W_down = self._prepare_weight_storage(weights.W_down)
            setattr(weights, "W_O_tensor", None)
            setattr(weights, "W_down_tensor", None)

    def _get_weight_tensor(
        self, weights: LayerWeights, attr: str
    ) -> Optional[torch.Tensor]:
        cache_attr = f"{attr}_tensor"
        cached = getattr(weights, cache_attr, None)
        target_device = (
            self.compute_device if self.use_torch_backend else torch.device("cpu")
        )

        if cached is not None:
            desired_dtype = (
                torch.float32 if target_device.type == "cpu" else cached.dtype
            )
            if cached.device != target_device or cached.dtype != desired_dtype:
                cached = cached.to(device=target_device, dtype=desired_dtype)
                setattr(weights, cache_attr, cached)
            return cached

        base = getattr(weights, attr, None)
        if base is None:
            return None

        tensor = base if isinstance(base, torch.Tensor) else torch.as_tensor(base)
        desired_dtype = torch.float32 if target_device.type == "cpu" else tensor.dtype
        tensor = tensor.to(device=target_device)
        if tensor.dtype != desired_dtype:
            tensor = tensor.to(dtype=desired_dtype)
        setattr(weights, cache_attr, tensor)
        return tensor

    # ------------------------------------------------------------------
    # Domain / label state management
    # ------------------------------------------------------------------

    def _get_or_create_domain_state(
        self,
        domain: str,
        modules: Sequence[str],
    ) -> DomainState:
        if domain not in self.state.domains:
            self._init_domain_state(domain, modules)
        return self.state.domains[domain]

    def _init_domain_state(
        self,
        domain: str,
        modules: Sequence[str],
    ) -> None:
        """Initialize NIG priors for a new domain / label group."""
        domain_state = DomainState()
        weight_device = self.compute_device
        cfg = self.config

        if "attn" in modules:
            for layer_idx, weights in self.layer_weights.items():
                matrix = getattr(weights, "W_O", None)
                if matrix is None:
                    continue
                dim = int(matrix.shape[0])
                domain_state.nig_attn[layer_idx] = init_nig(
                    dim=dim,
                    mu_0=cfg.mu_0,
                    lambda_0=cfg.lambda_0,
                    alpha_0=cfg.alpha_0,
                    beta_0=cfg.beta_0,
                    device=weight_device,
                )

        if "mlp" in modules:
            for layer_idx, weights in self.layer_weights.items():
                matrix = getattr(weights, "W_down", None)
                if matrix is None:
                    continue
                dim = int(matrix.shape[1])
                domain_state.nig_mlp[layer_idx] = init_nig(
                    dim=dim,
                    mu_0=cfg.mu_0,
                    lambda_0=cfg.lambda_0,
                    alpha_0=cfg.alpha_0,
                    beta_0=cfg.beta_0,
                    device=weight_device,
                )

        self.state.domains[domain] = domain_state

    # ------------------------------------------------------------------
    # Activation tensor preparation
    # ------------------------------------------------------------------

    def _prepare_activation_tensor(
        self,
        activation: torch.Tensor,
        target_device: torch.device,
        target_dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if activation is None:
            return None

        tensor = activation
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach()
        else:
            tensor = torch.as_tensor(tensor)
        if tensor.device != target_device:
            tensor = tensor.to(device=target_device)
        if tensor.dtype != target_dtype:
            tensor = tensor.to(dtype=target_dtype)
        if tensor.ndim > 2:
            tensor = tensor.reshape(-1, tensor.shape[-1])
        elif tensor.ndim == 1:
            tensor = tensor.reshape(1, -1)
        if tensor.numel() == 0:
            return None
        return tensor.contiguous()

    # ------------------------------------------------------------------
    # Evidence computation
    # ------------------------------------------------------------------

    def _compute_attn_evidence(
        self,
        layer_idx: int,
        activation: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Compute attention evidence from captured activation tensor."""
        weights = self.layer_weights.get(layer_idx)
        if weights is None:
            return None

        weight_matrix = self._get_weight_tensor(weights, "W_O")
        if weight_matrix is None:
            return None

        h_tensor = self._prepare_activation_tensor(
            activation,
            target_device=weight_matrix.device,
            target_dtype=weight_matrix.dtype,
        )
        if h_tensor is None:
            return None

        with torch.inference_mode():
            e_batch = compute_attn_evidence_batch(
                h_batch=h_tensor,
                W_O=weight_matrix,
                eps=self.config.eps,
            )
            # Return all rows — NIGPosterior.add_evidence handles [N, dim]
            evidence = e_batch[0] if e_batch.shape[0] == 1 else e_batch
        return evidence.detach()

    def _compute_mlp_evidence(
        self,
        layer_idx: int,
        activation: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Compute MLP evidence from captured activation tensor."""
        weights = self.layer_weights.get(layer_idx)
        if weights is None:
            return None

        weight_matrix = self._get_weight_tensor(weights, "W_down")
        if weight_matrix is None:
            return None

        a_tensor = self._prepare_activation_tensor(
            activation,
            target_device=weight_matrix.device,
            target_dtype=weight_matrix.dtype,
        )
        if a_tensor is None:
            return None

        with torch.inference_mode():
            e_batch = compute_ffn_evidence_batch(
                a_batch=a_tensor,
                W_down=weight_matrix,
                eps=self.config.eps,
            )
            evidence = e_batch[0] if e_batch.shape[0] == 1 else e_batch
        return evidence.detach()

    # ------------------------------------------------------------------
    # Evidence buffer helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _add_evidence_to_buffer(
        buffer: Dict[int, torch.Tensor],
        layer_idx: int,
        evidence: torch.Tensor,
        nig: NIGPosterior,
    ) -> None:
        """Accumulate evidence into a per-layer buffer (list) for batch add_evidence calls."""
        if evidence is None:
            return
        ev = evidence.detach().to(device=nig.device, dtype=torch.float32)
        if layer_idx not in buffer:
            buffer[layer_idx] = [ev]
        else:
            buffer[layer_idx].append(ev)
        del evidence

    @staticmethod
    def _register_sample(
        domain_state: DomainState,
        instance_idx: int,
        is_correct: bool,
    ) -> None:
        domain_state.sample_count += 1
        domain_state.instance_indices.append(instance_idx)
        if is_correct:
            domain_state.correct_count += 1
        else:
            domain_state.error_count += 1

    # ------------------------------------------------------------------
    # Streaming API
    # ------------------------------------------------------------------

    def begin_stream(
        self,
        domain: str,
        instance_idx: int,
        modules: Sequence[str] = ("attn", "mlp"),
    ) -> SampleStreamContext:
        """Create a streaming context for an in-flight sample."""
        domain_state = self._get_or_create_domain_state(domain, modules)
        return SampleStreamContext(
            accumulator=self,
            domain=domain,
            domain_state=domain_state,
            modules=tuple(modules),
            instance_idx=instance_idx,
        )

    def _consume_stream_activation(
        self,
        stream: SampleStreamContext,
        activation_type: str,
        layer_idx: int,
        activation: torch.Tensor,
    ) -> None:
        if not stream.active:
            return

        if activation_type == "attn_pre_out":
            if "attn" not in stream.modules:
                return
            if layer_idx not in stream.domain_state.nig_attn:
                return
            evidence = self._compute_attn_evidence(layer_idx, activation)
            if evidence is None:
                return
            nig = stream.domain_state.nig_attn[layer_idx]
            self._add_evidence_to_buffer(
                stream.attn_evidence,
                layer_idx,
                evidence,
                nig,
            )
            # Buffer raw activation for activation_mean ablation baseline
            act = activation.detach().to(dtype=torch.float32, device=nig.device)
            if layer_idx not in stream.attn_act_buf:
                stream.attn_act_buf[layer_idx] = [act]
            else:
                stream.attn_act_buf[layer_idx].append(act)
        elif activation_type == "mlp_pre_down":
            if "mlp" not in stream.modules:
                return
            if layer_idx not in stream.domain_state.nig_mlp:
                return
            evidence = self._compute_mlp_evidence(layer_idx, activation)
            if evidence is None:
                return
            nig = stream.domain_state.nig_mlp[layer_idx]
            self._add_evidence_to_buffer(
                stream.mlp_evidence,
                layer_idx,
                evidence,
                nig,
            )
            # Buffer raw activation for activation_mean ablation baseline
            act = activation.detach().to(dtype=torch.float32, device=nig.device)
            if layer_idx not in stream.mlp_act_buf:
                stream.mlp_act_buf[layer_idx] = [act]
            else:
                stream.mlp_act_buf[layer_idx].append(act)

    def _finalize_stream(
        self,
        stream: SampleStreamContext,
        is_correct: bool,
    ) -> None:
        if not stream.active:
            return

        self._register_sample(stream.domain_state, stream.instance_idx, is_correct)

        if is_correct:
            for layer_idx, ev_list in stream.attn_evidence.items():
                if layer_idx in stream.domain_state.nig_attn and ev_list:
                    nig = stream.domain_state.nig_attn[layer_idx]
                    for ev in ev_list:
                        nig.add_evidence(ev)
            for layer_idx, ev_list in stream.mlp_evidence.items():
                if layer_idx in stream.domain_state.nig_mlp and ev_list:
                    nig = stream.domain_state.nig_mlp[layer_idx]
                    for ev in ev_list:
                        nig.add_evidence(ev)
            # Update activation_mean stats from buffered raw activations
            for layer_idx, act_list in stream.attn_act_buf.items():
                if act_list:
                    nig = stream.domain_state.nig_attn.get(layer_idx)
                    if nig is not None:
                        merged = (
                            torch.cat(act_list, dim=0)
                            if act_list[0].ndim > 1
                            else act_list[0]
                        )
                        self._update_act_stats(
                            stream.domain_state.act_sum_attn,
                            stream.domain_state.act_n_attn,
                            layer_idx,
                            merged,
                            nig.device,
                        )
            for layer_idx, act_list in stream.mlp_act_buf.items():
                if act_list:
                    nig = stream.domain_state.nig_mlp.get(layer_idx)
                    if nig is not None:
                        merged = (
                            torch.cat(act_list, dim=0)
                            if act_list[0].ndim > 1
                            else act_list[0]
                        )
                        self._update_act_stats(
                            stream.domain_state.act_sum_mlp,
                            stream.domain_state.act_n_mlp,
                            layer_idx,
                            merged,
                            nig.device,
                        )

        stream.attn_evidence.clear()
        stream.mlp_evidence.clear()
        stream.attn_act_buf.clear()
        stream.mlp_act_buf.clear()
        stream.active = False

    def _abort_stream(self, stream: SampleStreamContext) -> None:
        stream.attn_evidence.clear()
        stream.mlp_evidence.clear()
        stream.attn_act_buf.clear()
        stream.mlp_act_buf.clear()
        stream.active = False

    # ------------------------------------------------------------------
    # Batch accumulation API
    # ------------------------------------------------------------------

    def accumulate_from_activations(
        self,
        domain: str,
        activations: Dict[str, Dict[int, torch.Tensor]],
        is_correct: bool,
        instance_idx: int,
        modules: Sequence[str] = ("attn", "mlp"),
    ) -> None:
        """Accumulate evidence from a single sample's activations.

        Args:
            domain: Grouping key — label name, class index string, etc.
            activations: ``{"attn_pre_out": {layer: tensor[T, dim]},
                           "mlp_pre_down": {layer: tensor[T, dim]}}``
                         where T is the number of generated tokens.
            is_correct: Whether the prediction was correct
            instance_idx: Global instance index
            modules: Which modules to process (``"attn"``, ``"mlp"``, or both)
        """
        domain_state = self._get_or_create_domain_state(domain, modules)
        self._register_sample(domain_state, instance_idx, is_correct)

        if not is_correct:
            return

        if "attn" in modules and "attn_pre_out" in activations:
            self._accumulate_attn(domain_state, activations["attn_pre_out"])

        if "mlp" in modules and "mlp_pre_down" in activations:
            self._accumulate_mlp(domain_state, activations["mlp_pre_down"])

    def accumulate_batch(
        self,
        domains: Sequence[str],
        is_correct_flags: Sequence[bool],
        instance_indices: Sequence[int],
        attn_activations: Dict[int, torch.Tensor],
        mlp_activations: Dict[int, torch.Tensor],
        modules: Sequence[str] = ("attn", "mlp"),
    ) -> None:
        """Accumulate evidence from a full batch, grouped by domain.

        Much faster than calling :meth:`accumulate_from_activations` per
        sample because it batches the evidence matrix multiplications for
        all samples sharing the same predicted label.

        Args:
            domains: Predicted label / domain string for each sample.
            is_correct_flags: Whether each prediction was correct.
            instance_indices: Global instance index for each sample.
            attn_activations: ``{layer_idx: tensor[B, dim]}`` captured by hooks.
            mlp_activations: ``{layer_idx: tensor[B, dim]}`` captured by hooks.
            modules: Which sub-modules to process (``"attn"``, ``"mlp"``).
        """
        B = len(domains)

        # ---- group samples by domain (predicted label) ----
        groups: Dict[str, Dict] = {}
        for i in range(B):
            d = domains[i]
            if d not in groups:
                groups[d] = {
                    "sample_info": [],  # (instance_idx, is_correct)
                    "correct_batch_idx": [],  # indices into the batch dim
                }
            groups[d]["sample_info"].append((instance_indices[i], is_correct_flags[i]))
            if is_correct_flags[i]:
                groups[d]["correct_batch_idx"].append(i)

        # ---- per-domain: register samples & batch-compute evidence ----
        for domain, grp in groups.items():
            domain_state = self._get_or_create_domain_state(domain, modules)

            # Register every sample (both correct and incorrect)
            for inst_idx, is_corr in grp["sample_info"]:
                self._register_sample(domain_state, inst_idx, is_corr)

            correct_idx = grp["correct_batch_idx"]
            if not correct_idx:
                continue

            sel = torch.tensor(correct_idx, dtype=torch.long)

            # Attention evidence — batch all correct samples at once
            if "attn" in modules and attn_activations:
                for layer_idx, act_batch in attn_activations.items():
                    if layer_idx not in domain_state.nig_attn:
                        continue
                    h_sub = act_batch[sel]  # [N_correct, dim]
                    evidence = self._compute_attn_evidence(layer_idx, h_sub)
                    if evidence is not None:
                        nig = domain_state.nig_attn[layer_idx]
                        nig.add_evidence(evidence)
                        self._update_act_stats(
                            domain_state.act_sum_attn,
                            domain_state.act_n_attn,
                            layer_idx,
                            h_sub,
                            nig.device,
                        )

            # MLP evidence — batch all correct samples at once
            if "mlp" in modules and mlp_activations:
                for layer_idx, act_batch in mlp_activations.items():
                    if layer_idx not in domain_state.nig_mlp:
                        continue
                    a_sub = act_batch[sel]  # [N_correct, dim]
                    evidence = self._compute_mlp_evidence(layer_idx, a_sub)
                    if evidence is not None:
                        nig = domain_state.nig_mlp[layer_idx]
                        nig.add_evidence(evidence)
                        self._update_act_stats(
                            domain_state.act_sum_mlp,
                            domain_state.act_n_mlp,
                            layer_idx,
                            a_sub,
                            nig.device,
                        )

    @staticmethod
    def _update_act_stats(
        act_sum: Dict[int, torch.Tensor],
        act_n: Dict[int, int],
        layer_idx: int,
        activation: torch.Tensor,
        target_device: torch.device,
    ) -> None:
        """Accumulate per-neuron absolute-value activation mean for one layer.

        Used by the Activation Mean ablation baseline.  The running sum and
        token count are maintained so that ``act_sum[layer] / act_n[layer]``
        gives the per-neuron mean absolute activation over all seen tokens.

        Args:
            act_sum: Per-layer tensor accumulator (modified in-place).
            act_n:   Per-layer token counter (modified in-place).
            layer_idx: Layer index.
            activation: Raw activation tensor ``[T, dim]`` (T = generated tokens).
            target_device: Device to place tensors on.
        """
        flat_activation = activation.detach().reshape(-1, activation.shape[-1])
        token_count = flat_activation.shape[0]
        if token_count == 0:
            return
        act_abs = (
            flat_activation.abs().sum(0).to(dtype=torch.float32, device=target_device)
        )
        if layer_idx not in act_sum:
            act_sum[layer_idx] = act_abs.clone()
            act_n[layer_idx] = token_count
        else:
            act_sum[layer_idx].add_(act_abs)
            act_n[layer_idx] += token_count

    def _accumulate_attn(
        self,
        domain_state: DomainState,
        attn_activations: Dict[int, torch.Tensor],
    ) -> None:
        for layer_idx, activation in attn_activations.items():
            if layer_idx not in domain_state.nig_attn:
                continue
            nig = domain_state.nig_attn[layer_idx]
            evidence = self._compute_attn_evidence(layer_idx, activation)
            if evidence is not None:
                nig.add_evidence(evidence)
                self._update_act_stats(
                    domain_state.act_sum_attn,
                    domain_state.act_n_attn,
                    layer_idx,
                    activation,
                    nig.device,
                )

    def _accumulate_mlp(
        self,
        domain_state: DomainState,
        mlp_activations: Dict[int, torch.Tensor],
    ) -> None:
        for layer_idx, activation in mlp_activations.items():
            if layer_idx not in domain_state.nig_mlp:
                continue
            nig = domain_state.nig_mlp[layer_idx]
            evidence = self._compute_mlp_evidence(layer_idx, activation)
            if evidence is not None:
                nig.add_evidence(evidence)
                self._update_act_stats(
                    domain_state.act_sum_mlp,
                    domain_state.act_n_mlp,
                    layer_idx,
                    activation,
                    nig.device,
                )

    # ------------------------------------------------------------------
    # Summary / inspection
    # ------------------------------------------------------------------

    def get_state(self) -> OnlineRACEState:
        return self.state

    def get_domain_summary(self, domain: str) -> Dict:
        if domain not in self.state.domains:
            raise ValueError(f"Domain '{domain}' not found")
        ds = self.state.domains[domain]
        # Pick any NIG to report observation count
        sample_nig = next(iter(ds.nig_attn.values()), None) or next(
            iter(ds.nig_mlp.values()), None
        )
        n_obs = sample_nig.N if sample_nig is not None else 0
        return {
            "domain": domain,
            "sample_count": ds.sample_count,
            "correct_count": ds.correct_count,
            "error_count": ds.error_count,
            "accuracy": (
                ds.correct_count / ds.sample_count if ds.sample_count > 0 else 0.0
            ),
            "attn_layers": len(ds.nig_attn),
            "mlp_layers": len(ds.nig_mlp),
            "nig_observations": n_obs,
            "mu_0": self.config.mu_0,
            "lambda_0": self.config.lambda_0,
            "alpha_0": self.config.alpha_0,
            "beta_0": self.config.beta_0,
        }

    def print_summary(self) -> None:
        print("\n" + "=" * 80)
        print("Online RACE Accumulation Summary")
        print("=" * 80)

        for domain in sorted(self.state.domains.keys()):
            s = self.get_domain_summary(domain)
            print(f"\n{domain}:")
            print(f"  Total samples: {s['sample_count']}")
            print(f"  Correct: {s['correct_count']}")
            print(f"  Errors: {s['error_count']}")
            print(f"  Accuracy: {s['accuracy']:.2%}")
            print(f"  Attn layers: {s['attn_layers']}")
            print(f"  MLP layers: {s['mlp_layers']}")
            print(f"  NIG observations: {s['nig_observations']}")

        print("=" * 80)

    # ------------------------------------------------------------------
    # H5 persistence
    # ------------------------------------------------------------------

    def init_h5_file(self, output_path: str) -> str:
        import h5py

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not output_path.endswith(".h5"):
            output_path = f"{output_path}_{timestamp}.h5"

        race_config_dict = {
            "mu_0": self.config.mu_0,
            "lambda_0": self.config.lambda_0,
            "alpha_0": self.config.alpha_0,
            "beta_0": self.config.beta_0,
            "gamma": self.config.gamma,
            "eps": self.config.eps,
            "device_requested": self.config.device,
            "device": self.runtime_device,
        }

        with h5py.File(output_path, "w") as f:
            run_payload = {
                "mode": "online_streaming",
                "compute_device": self.runtime_device,
            }
            ensure_race_meta(
                f,
                target_model_family=self.config.model_family,
                model_name=self.model_name,
                device=self.runtime_device,
                race_config=race_config_dict,
                created_at=timestamp,
                extra_run_payload=run_payload,
            )
            ensure_reports_group(f)

        print(f"Initialized H5 file: {output_path}")
        return output_path

    def _save_posteriors_to_group(
        self,
        modules_group,
        nig_dict: Dict[int, NIGPosterior],
        module_name: str,
        act_sum_dict: Optional[Dict[int, torch.Tensor]] = None,
        act_n_dict: Optional[Dict[int, int]] = None,
    ) -> None:
        """Write posterior metrics and ablation baselines for one module type.

        Saved datasets per layer:
        - ``posterior_mean``  — signed NIG posterior mean mu_n
        - ``cam_score``       — positive Conservative Alignment Magnitude
        - ``negative_cam_score`` — negative-CAM ablation score
        - ``empirical_mean``  — pure arithmetic mean of alignment scores (no prior)
        - ``empirical_snr``   — positive empirical mean / sample std
        - ``activation_mean`` — mean absolute neuron activation (direction-agnostic)
        - ``nig_sum_e`` / ``nig_sum_e2`` — sufficient statistics for recomputation

        Args:
            modules_group: Parent H5 group to write into.
            nig_dict: Mapping from layer index to :class:`NIGPosterior`.
            module_name: Name for the module subgroup (e.g. ``"attn_pre_output"``).
            act_sum_dict: Optional per-layer activation absolute-value sums
                accumulated during evidence collection.  Required for saving
                ``activation_mean``.
            act_n_dict: Optional per-layer token counts matching *act_sum_dict*.
        """
        if not nig_dict:
            return
        module_group = modules_group.require_group(module_name)
        for layer_idx, nig in nig_dict.items():
            layer_group = module_group.create_group(f"layer_{layer_idx:02d}")
            layer_group.attrs["feature_dim"] = nig.dim
            layer_group.attrs["N"] = nig.N
            layer_group.attrs["mu_0"] = nig.mu_0
            layer_group.attrs["lambda_0"] = nig.lambda_0
            layer_group.attrs["alpha_0"] = nig.alpha_0
            layer_group.attrs["beta_0"] = nig.beta_0
            layer_group.attrs["lambda_n"] = nig.lambda_n
            layer_group.attrs["alpha_n"] = nig.alpha_n

            # Signed posterior mean: positive=excitatory, negative=inhibitory
            posterior_mean = nig.posterior_mean.cpu().numpy().astype(np.float32)
            layer_group.create_dataset(
                "posterior_mean",
                data=posterior_mean,
                compression="gzip",
            )

            # CAM score: non-negative, statistically verified consistency
            cam = (
                nig.cam_score(gamma=self.config.gamma).cpu().numpy().astype(np.float32)
            )
            layer_group.create_dataset(
                "cam_score",
                data=cam,
                compression="gzip",
            )
            layer_group.create_dataset(
                "negative_cam_score",
                data=nig.negative_cam_score(gamma=self.config.gamma)
                .cpu()
                .numpy()
                .astype(np.float32),
                compression="gzip",
            )

            # Ablation baseline: pure empirical mean (no prior regularisation)
            layer_group.create_dataset(
                "empirical_mean",
                data=nig.empirical_mean.cpu().numpy().astype(np.float32),
                compression="gzip",
            )

            # Ablation baseline: frequentist SNR = mean(e) / std(e)
            layer_group.create_dataset(
                "empirical_snr",
                data=nig.empirical_snr.cpu().numpy().astype(np.float32),
                compression="gzip",
            )

            # Ablation baseline: mean absolute neuron activation (ignores alignment)
            if (
                act_sum_dict is not None
                and act_n_dict is not None
                and layer_idx in act_sum_dict
                and act_n_dict.get(layer_idx, 0) > 0
            ):
                act_mean = (
                    (act_sum_dict[layer_idx] / act_n_dict[layer_idx])
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                layer_group.create_dataset(
                    "activation_mean",
                    data=act_mean,
                    compression="gzip",
                )

            # Preserve sufficient statistics so gamma can be recomputed.
            sd = nig.state_dict()
            layer_group.create_dataset(
                "nig_sum_e",
                data=sd["sum_e"].astype(np.float64),
                compression="gzip",
            )
            layer_group.create_dataset(
                "nig_sum_e2",
                data=sd["sum_e2"].astype(np.float64),
                compression="gzip",
            )

    def save_domain_to_h5(
        self,
        output_path: str,
        domain_name: str,
        axis_type: str = "domain",
    ) -> None:
        """Save one domain's posteriors to an existing H5 file."""
        import h5py

        if domain_name not in self.state.domains:
            raise ValueError(f"Domain '{domain_name}' not found in state")

        domain_state = self.state.domains[domain_name]

        with h5py.File(output_path, "a") as f:
            meta_group = f.require_group("meta")
            axes_group = meta_group.require_group("axes")
            reports_group = f.require_group("reports")
            axes_reports_group = reports_group.require_group("axes")

            # Use integer key: try to parse domain_name as int, otherwise hash
            try:
                value_key = int(domain_name)
            except (ValueError, TypeError):
                value_key = _make_axis_id(domain_name)

            payload = self.state.domain_metadata.get(domain_name, {})
            value_group, axis_path = ensure_axis_value(
                axes_group,
                axis_type=axis_type,
                value_key=value_key,
                display_name=domain_name,
                raw_value={axis_type: domain_name, f"{axis_type}_id": value_key},
                payload=payload or None,
            )
            append_axis_members(value_group, domain_state.instance_indices)
            value_group.attrs["instance_count_total"] = domain_state.sample_count
            value_group.attrs["instance_count_effective"] = domain_state.correct_count
            value_token = axis_path.split("/")[-1]

            domain_result_group = axes_reports_group.require_group(value_token)
            domain_result_group.attrs["axis_path"] = axis_path
            domain_result_group.attrs["instance_count_total"] = (
                domain_state.sample_count
            )
            domain_result_group.attrs["instance_count_effective"] = (
                domain_state.correct_count
            )
            domain_result_group.attrs["error_count"] = domain_state.error_count

            modules_group = domain_result_group.require_group("modules")
            self._save_posteriors_to_group(
                modules_group,
                domain_state.nig_attn,
                "attn_pre_output",
                act_sum_dict=domain_state.act_sum_attn,
                act_n_dict=domain_state.act_n_attn,
            )
            self._save_posteriors_to_group(
                modules_group,
                domain_state.nig_mlp,
                "mlp_pre_down",
                act_sum_dict=domain_state.act_sum_mlp,
                act_n_dict=domain_state.act_n_mlp,
            )

        print(f"✓ Saved domain '{domain_name}' to {output_path}")

    def finalize_h5_file(self, output_path: str) -> None:
        import h5py

        total_samples = 0
        total_correct = 0
        for ds in self.state.domains.values():
            total_samples += ds.sample_count
            total_correct += ds.correct_count

        with h5py.File(output_path, "a") as f:
            reports_group = f.require_group("reports")
            summary_group = reports_group.require_group("summary")
            metrics_group = ensure_metrics_group(summary_group)
            metrics_group.attrs["total_domains"] = len(self.state.domains)
            metrics_group.attrs["total_samples"] = total_samples
            metrics_group.attrs["total_correct"] = total_correct
            metrics_group.attrs["gamma"] = self.config.gamma
            update_total_instances(f, total_samples)

        print(f"✓ Finalized H5 file: {output_path}")

    def clear_domain(self, domain_name: str) -> None:
        if domain_name in self.state.domains:
            ds = self.state.domains[domain_name]
            ds.nig_attn.clear()
            ds.nig_mlp.clear()
            ds.instance_indices.clear()
            del self.state.domains[domain_name]
            gc.collect()
            print(f"✓ Cleared domain '{domain_name}' from memory")

    def save_results(self, output_path: str, axis_type: str = "domain") -> str:
        """Save all domain results to a single H5 file."""
        import h5py

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not output_path.endswith(".h5"):
            output_path = f"{output_path}_{timestamp}.h5"

        race_config_dict = {
            "mu_0": self.config.mu_0,
            "lambda_0": self.config.lambda_0,
            "alpha_0": self.config.alpha_0,
            "beta_0": self.config.beta_0,
            "gamma": self.config.gamma,
            "eps": self.config.eps,
            "device_requested": self.config.device,
            "device": self.runtime_device,
        }

        with h5py.File(output_path, "w") as f:
            run_payload = {
                "mode": "online",
                "compute_device": self.runtime_device,
            }
            _, axes_group = ensure_race_meta(
                f,
                target_model_family=self.config.model_family,
                model_name=self.model_name,
                device=self.runtime_device,
                race_config=race_config_dict,
                created_at=timestamp,
                extra_run_payload=run_payload,
            )

            reports_group = ensure_reports_group(f)
            axes_reports_group = reports_group.require_group("axes")
            summary_group = reports_group.require_group("summary")
            metrics_group = ensure_metrics_group(summary_group)

            total_samples = 0
            total_correct = 0

            for domain_name, domain_state in self.state.domains.items():
                total_samples += domain_state.sample_count
                total_correct += domain_state.correct_count

                try:
                    value_key = int(domain_name)
                except (ValueError, TypeError):
                    value_key = _make_axis_id(domain_name)

                payload = self.state.domain_metadata.get(domain_name, {})
                value_group, axis_path = ensure_axis_value(
                    axes_group,
                    axis_type=axis_type,
                    value_key=value_key,
                    display_name=domain_name,
                    raw_value={axis_type: domain_name, f"{axis_type}_id": value_key},
                    payload=payload or None,
                )
                append_axis_members(value_group, domain_state.instance_indices)
                value_group.attrs["instance_count_total"] = domain_state.sample_count
                value_group.attrs["instance_count_effective"] = (
                    domain_state.correct_count
                )
                value_token = axis_path.split("/")[-1]

                domain_result_group = axes_reports_group.require_group(value_token)
                domain_result_group.attrs["axis_path"] = axis_path
                domain_result_group.attrs["instance_count_total"] = (
                    domain_state.sample_count
                )
                domain_result_group.attrs["instance_count_effective"] = (
                    domain_state.correct_count
                )
                domain_result_group.attrs["error_count"] = domain_state.error_count

                modules_group = domain_result_group.require_group("modules")
                self._save_posteriors_to_group(
                    modules_group,
                    domain_state.nig_attn,
                    "attn_pre_output",
                    act_sum_dict=domain_state.act_sum_attn,
                    act_n_dict=domain_state.act_n_attn,
                )
                self._save_posteriors_to_group(
                    modules_group,
                    domain_state.nig_mlp,
                    "mlp_pre_down",
                    act_sum_dict=domain_state.act_sum_mlp,
                    act_n_dict=domain_state.act_n_mlp,
                )

            metrics_group.attrs["total_domains"] = len(self.state.domains)
            metrics_group.attrs["total_samples"] = total_samples
            metrics_group.attrs["total_correct"] = total_correct
            metrics_group.attrs["gamma"] = self.config.gamma
            update_total_instances(f, total_samples)

        print(f"Saved RACE results to {output_path}")
        return output_path
