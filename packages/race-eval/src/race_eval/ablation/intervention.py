"""Base neuron modification (intervention) abstraction.

This module provides the abstract base class :class:`BaseNeuronModifier` that
encapsulates the common hook-registration pattern used by neuron modifiers.

Subclasses must implement:
- ``_get_layers()`` — retrieve the list of model layers
- ``_find_attn_proj(layer)`` — locate the attention output projection module
- ``_find_ffn_proj(layer)`` — locate the FFN down projection module
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional, Sequence, Set

import torch
from torch import nn

from race_eval.ablation.plan import AblationPlan, OperationMode


class BaseNeuronModifier(abc.ABC):
    """Abstract base class for neuron modification via forward hooks.

    Shared workflow:
    1. ``register()`` / ``apply()`` — attach hooks to the relevant modules.
    2. Model runs inference — hooks intercept and modify activations.
    3. ``remove()`` — detach all hooks.
    """

    def __init__(self, model: nn.Module, plan: AblationPlan):
        self.model = model
        self.plan = plan
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

    # ------------------------------------------------------------------
    # Abstract interface — each subclass defines how to find layers/modules
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _get_layers(self) -> Sequence[nn.Module]:
        """Return an ordered sequence of transformer layers."""
        ...

    @abc.abstractmethod
    def _find_attn_proj(self, layer: nn.Module) -> Optional[nn.Module]:
        """Return the attention output projection submodule, or None."""
        ...

    @abc.abstractmethod
    def _find_ffn_proj(self, layer: nn.Module) -> Optional[nn.Module]:
        """Return the FFN down-projection submodule, or None."""
        ...

    # ------------------------------------------------------------------
    # Hook creation — shared logic
    # ------------------------------------------------------------------

    def _make_modification_hook(
        self,
        neuron_indices: Sequence[int],
        operation_mode: str,
        enhancement_factor: float,
    ):
        """Create a forward-pre-hook that modifies specified neuron dimensions.

        The hook builds a multiplicative mask on first invocation and caches it
        for subsequent calls.
        """
        neuron_list = sorted(int(i) for i in neuron_indices)
        _cached_mask: Dict[str, torch.Tensor] = {}

        op = operation_mode if isinstance(operation_mode, str) else operation_mode.value

        def hook(module: nn.Module, inputs):
            if len(inputs) == 0:
                return inputs
            hidden = inputs[0]
            if not isinstance(hidden, torch.Tensor):
                return inputs

            # Build / retrieve cached mask
            cache_key = f"{hidden.shape[-1]}_{hidden.dtype}_{hidden.device}"
            if cache_key not in _cached_mask:
                mask = torch.ones(
                    hidden.shape[-1], dtype=hidden.dtype, device=hidden.device
                )
                if op in (
                    "suppress",
                    "keep_top",
                    OperationMode.SUPPRESS.value,
                    OperationMode.KEEP_TOP.value,
                ):
                    mask[neuron_list] = 0.0
                elif op in ("enhance", OperationMode.ENHANCE.value):
                    mask[neuron_list] = enhancement_factor
                _cached_mask[cache_key] = mask

            mask = _cached_mask[cache_key]

            # Broadcast multiply — works for both 2-D and 3-D tensors
            # View mask as (..., hidden_dim)
            expanded = mask
            for _ in range(hidden.dim() - 1):
                expanded = expanded.unsqueeze(0)

            return (hidden * expanded,) + inputs[1:]

        return hook

    def _make_mean_ablation_hook(
        self,
        neuron_indices: Sequence[int],
        mean_values: torch.Tensor,
    ):
        """Create a forward-pre-hook that replaces specified neurons with their calibration mean.

        Unlike the zero-ablation mask (which sets neurons to 0 and distorts
        the residual stream norm), this hook substitutes with the empirical
        expectation from a neutral corpus, preserving the statistical properties
        that LayerNorm and downstream layers expect.

        Args:
            neuron_indices: Neuron dimensions to ablate.
            mean_values: 1-D tensor of per-dimension means (full hidden dim).
                         Only the entries at ``neuron_indices`` are used.
        """
        neuron_list = sorted(int(i) for i in neuron_indices)
        if not neuron_list:

            def noop_hook(module: nn.Module, inputs):
                del module
                return inputs

            return noop_hook

        index_cpu = torch.tensor(neuron_list, dtype=torch.long)
        mean_subset_cpu = mean_values.index_select(0, index_cpu).contiguous()
        cached_tensors: Dict[
            tuple[torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor]
        ] = {}

        def hook(module: nn.Module, inputs):
            del module
            if len(inputs) == 0:
                return inputs
            hidden = inputs[0]
            if not isinstance(hidden, torch.Tensor):
                return inputs

            cache_key = (hidden.device, hidden.dtype)
            if cache_key not in cached_tensors:
                idx_tensor = index_cpu.to(device=hidden.device)
                mean_slice = mean_subset_cpu.to(
                    device=hidden.device,
                    dtype=hidden.dtype,
                )
                cached_tensors[cache_key] = (idx_tensor, mean_slice)

            idx_tensor, mean_slice = cached_tensors[cache_key]

            # In-place: safe under inference_mode since this tensor is a
            # fresh intermediate (attention/MLP output), not the residual stream.
            hidden[..., idx_tensor] = mean_slice
            return inputs

        return hook

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def remove(self) -> None:
        """Remove all registered hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def __enter__(self):
        self.apply()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()

    def apply(self) -> None:
        """Register hooks (alias used by LLM pipeline). Override if needed."""
        self.register()

    def register(self) -> None:
        """Register hooks (alias used by CV pipeline). Override if needed."""
        raise NotImplementedError("Subclass must implement register() or apply()")
