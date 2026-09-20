"""Activation recorder for Language Models (LLM) for RACE analysis.

This module records activations from decoder-only language models,
capturing the pre-projection states for both self-attention and MLP modules.

Key design decisions:

- **Always batch format** — single prompts are wrapped in lists of length 1,
  eliminating Union-type branching throughout the codebase.
- **CPU activation storage** — captured tensors are moved to CPU by default
  (via ``non_blocking`` transfer) to minimise GPU memory pressure during long
  generation runs.
- **Streaming evidence** — after generation, :meth:`stream_evidence` /
  :meth:`iter_sample_activations` process evidence sample-by-sample and
  release tensors incrementally, avoiding a full copy of all activations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from race.core.analyzer import LayerWeights
from race.core.recorder import BaseActivationRecorder
from race.llm.model_utils import (
    extract_attention_out_proj_weight,
    extract_mlp_down_proj_weight,
    find_attention_out_proj,
    find_mlp_down_proj,
    get_decoder_layers,
    get_language_model,
)

logger = logging.getLogger(__name__)


# ======================================================================
# Result container
# ======================================================================


@dataclass
class LLMSampleResult:
    """Inference metadata for a batch of LLM samples.

    Always uses batch format (``batch_size >= 1``).  Single-sample
    inference simply produces lists of length 1.

    **Activations are NOT stored here** — they remain in the recorder
    for streaming evidence processing via
    :meth:`LLMActivationRecorder.stream_evidence`.
    """

    sample_indices: List[int]
    prompts: List[str]
    generated_texts: List[str] | None
    generated_token_ids: List[List[int]] | None
    is_correct: List[bool]
    batch_size: int
    num_generated_tokens: List[int]


# ======================================================================
# Recorder
# ======================================================================


class LLMActivationRecorder(BaseActivationRecorder):
    """Record language model activations for RACE analysis.

    Captures pre-projection activations for:

    - ``attn_pre_out`` — before self-attention output projection (``o_proj``)
    - ``mlp_pre_down`` — before MLP down projection (``down_proj``)

    Activations are stored on **CPU** by default to minimise GPU memory
    pressure during generation.  Use :meth:`stream_evidence` after
    :meth:`record_sample` to process evidence incrementally and release
    memory.
    """

    def __init__(
        self,
        model_name: str,
        output_dir: str,
        use_cache: bool = True,
        first_token_only: bool = False,
        activation_device: str = "cuda",
    ):
        """Initialise the LLM activation recorder.

        Args:
            model_name: Hugging Face model identifier.
            output_dir: Directory to save results.
            use_cache: Enable KV cache for faster generation.
            first_token_only: Keep only the first generated token's
                activations for RACE evidence.
            activation_device: Device for captured activations
                (``"cpu"`` saves GPU memory; ``"cuda"`` avoids transfers).
        """
        super().__init__(model_name, output_dir)
        self.use_cache = use_cache
        self.first_token_only = first_token_only

        self.tokenizer: Optional[Any] = None
        self.model_dtype: torch.dtype = torch.float32

        self.attn_pre_out_activations: Dict[int, List[torch.Tensor]] = {}
        self.mlp_pre_down_activations: Dict[int, List[torch.Tensor]] = {}
        self.activation_storage_device = torch.device(activation_device)
        self.model_device_map: Optional[Any] = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    @staticmethod
    def _device_from_map_value(value: Any) -> Optional[torch.device]:
        """Convert a Hugging Face device-map value to a torch device."""
        if value is None:
            return None
        if isinstance(value, torch.device):
            return value
        if isinstance(value, int):
            return torch.device(f"cuda:{value}")
        if isinstance(value, str):
            if value == "disk":
                return None
            try:
                return torch.device(value)
            except RuntimeError:
                return None
        return None

    def _infer_input_device(self) -> torch.device:
        """Infer where tokenized inputs should be placed for generation."""
        if self.model is None:
            raise RuntimeError("Model must be loaded first")

        try:
            embeddings = self.model.get_input_embeddings()
            if embeddings is not None and hasattr(embeddings, "weight"):
                weight_device = embeddings.weight.device
                if weight_device.type != "meta":
                    return weight_device
        except Exception:
            logger.debug("Could not infer input device from embeddings", exc_info=True)

        device_map = getattr(self.model, "hf_device_map", None)
        if isinstance(device_map, dict):
            for name, value in device_map.items():
                lower_name = str(name).lower()
                if any(key in lower_name for key in ("embed", "wte", "tok")):
                    device = self._device_from_map_value(value)
                    if device is not None:
                        return device
            for value in device_map.values():
                device = self._device_from_map_value(value)
                if device is not None and device.type != "cpu":
                    return device

        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    def load_model(
        self,
        device: str = "cuda",
        device_map: Optional[Any] = None,
        input_device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        compile_model: bool = True,
    ) -> None:
        """Load the language model and tokenizer.

        Args:
            device: Default model device or device-map value.
            device_map: Hugging Face device map. Use ``"auto"`` for
                multi-GPU sharded loading.
            input_device: Device for tokenizer outputs. Defaults to the input
                embedding device, which is required for sharded models.
            dtype: Model precision (default: ``bfloat16``).
            compile_model: Compile the model after loading. Disabled
                automatically for non-single-device maps.
        """
        logger.info("Loading model '%s' ...", self.model_name)

        if device_map is None:
            device_map = device
        self.model_device_map = device_map

        # Fast (Rust-based) tokenizer avoids CPU bottleneck
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_fast=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Left padding is CRITICAL for decoder-only batch generation.
        # Right-padding causes incorrect results because the model attends
        # to padding tokens on the right during generation.
        self.tokenizer.padding_side = "left"

        if dtype is None:
            dtype = torch.bfloat16

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
            use_cache=self.use_cache,
            attn_implementation="flash_attention_2",
        )
        self.model.eval()
        self.model_dtype = dtype
        self.device = (
            torch.device(input_device)
            if input_device is not None
            else self._infer_input_device()
        )

        is_sharded = isinstance(device_map, str) and device_map in {
            "auto",
            "balanced",
            "balanced_low_0",
            "sequential",
        }
        if compile_model and not is_sharded:
            self.model = torch.compile(self.model, mode="max-autotune-no-cudagraphs")
        elif compile_model and is_sharded:
            logger.warning(
                "Skipping torch.compile because device_map=%r may shard the model",
                device_map,
            )

        logger.info(
            "Model loaded with device_map=%r, input_device=%s (dtype=%s, padding=%s)",
            self.model_device_map,
            self.device,
            self.model_dtype,
            self.tokenizer.padding_side,
        )

    # ------------------------------------------------------------------
    # Weight extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_weight(
        weight: Optional[torch.Tensor],
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Detach, cast to *dtype*, and ensure contiguity."""
        if weight is None:
            return None
        tensor = weight.detach()
        if tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        return tensor.contiguous()

    def extract_layer_weights(self) -> Dict[int, LayerWeights]:
        """Extract projection weights needed for RACE analysis.

        Returns:
            Mapping from layer index to :class:`LayerWeights`.
        """
        if self.model is None:
            raise RuntimeError("Model must be loaded first")

        logger.info("Extracting projection weights ...")
        backbone = get_language_model(self.model)
        layers = get_decoder_layers(backbone)

        layer_weights: Dict[int, LayerWeights] = {}
        for idx, layer in enumerate(layers):
            layer_weights[idx] = LayerWeights(
                W_O=self._normalize_weight(
                    extract_attention_out_proj_weight(layer), self.model_dtype
                ),
                W_down=self._normalize_weight(
                    extract_mlp_down_proj_weight(layer), self.model_dtype
                ),
            )

        logger.info(
            "Extracted weights for %d layers (dtype=%s)",
            len(layer_weights),
            self.model_dtype,
        )
        return layer_weights

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def register_hooks(self) -> None:
        """Register forward-pre hooks on attention and MLP projection modules."""
        if self.model is None:
            raise RuntimeError("Model must be loaded first")

        self.remove_hooks()

        backbone = get_language_model(self.model)
        layers = get_decoder_layers(backbone)

        for layer_idx, layer in enumerate(layers):
            for finder_fn, act_type in [
                (find_attention_out_proj, "attn_pre_out"),
                (find_mlp_down_proj, "mlp_pre_down"),
            ]:
                module = finder_fn(layer)
                if module is not None:
                    hook = self._make_pre_forward_hook(layer_idx, act_type)
                    self.hooks.append(module.register_forward_pre_hook(hook))

        logger.info("Registered %d hooks on %d layers", len(self.hooks), len(layers))

    def _make_pre_forward_hook(self, layer_idx: int, activation_type: str):
        """Return a pre-forward hook that captures activations.

        Hook behaviour during ``model.generate()`` with KV cache:

        - **Prefill phase**: one call with ``[batch, prompt_len, dim]`` — skipped
        - **Generation phase**: one call per new token with ``[batch, 1, dim]`` — captured
        """
        storage_device = self.activation_storage_device
        store = (
            self.attn_pre_out_activations
            if activation_type == "attn_pre_out"
            else self.mlp_pre_down_activations
        )

        def hook(module, inputs):
            if not inputs:
                return
            activation = inputs[0]
            if not isinstance(activation, torch.Tensor):
                return
            # Prefill produces seq_len > 1; decode steps always produce seq_len == 1.
            # Skip prefill to avoid storing large prompt-length tensors.
            if activation.ndim == 3 and activation.shape[1] > 1:
                return
            activation = activation.detach()
            if activation.device != storage_device:
                activation = activation.to(device=storage_device, non_blocking=True)
            store.setdefault(layer_idx, []).append(activation)

        return hook

    def _make_prefill_hook(self, layer_idx: int, activation_type: str):
        """Return a hook that captures the prefill activation (seq_len > 1).

        Used by :meth:`record_sample_teacher_forcing`.  Each token position
        is stored as a separate ``[batch, 1, dim]`` slice so that
        ``iter_sample_activations`` can consume them without modification.
        """
        storage_device = self.activation_storage_device
        store = (
            self.attn_pre_out_activations
            if activation_type == "attn_pre_out"
            else self.mlp_pre_down_activations
        )

        def hook(module, inputs):
            if not inputs:
                return
            activation = inputs[0]
            if not isinstance(activation, torch.Tensor):
                return
            if activation.ndim != 3 or activation.shape[1] <= 1:
                return
            activation = activation.detach()
            if activation.device != storage_device:
                activation = activation.to(device=storage_device, non_blocking=True)
            # Split [B, T, dim] → T tensors of [B, 1, dim]
            for t in range(activation.shape[1]):
                store.setdefault(layer_idx, []).append(activation[:, t : t + 1, :])

        return hook

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        super().remove_hooks()
        logger.debug("Removed all hooks")

    # ------------------------------------------------------------------
    # Activation management
    # ------------------------------------------------------------------

    def clear_activations(self) -> None:
        """Release all stored activation tensors."""
        self.attn_pre_out_activations.clear()
        self.mlp_pre_down_activations.clear()

    def _reset_activations(self) -> None:
        """Alias for :meth:`clear_activations`."""
        self.clear_activations()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_sample(
        self,
        prompts: Union[str, List[str]],
        sample_indices: Union[int, List[int]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        do_sample: Optional[bool] = None,
        top_p: Optional[float] = None,
        is_correct: Union[bool, List[bool]] = True,
    ) -> LLMSampleResult:
        """Run generation and capture activations for a batch of prompts.

        Inputs are normalised to batch format (single strings / ints are
        wrapped in lists).  Activations remain in this recorder; call
        :meth:`stream_evidence` or iterate via
        :meth:`iter_sample_activations` to process and free them.

        Args:
            prompts: One or more input prompts.
            sample_indices: Corresponding sample indices.
            max_new_tokens: Maximum tokens to generate per sample. ``None`` keeps
                the model generation config value.
            temperature: Sampling temperature. ``None`` keeps the model
                generation config value.
            do_sample: Use stochastic sampling (``False`` = greedy). ``None``
                keeps the model generation config value.
            top_p: Nucleus-sampling threshold. ``None`` keeps the model
                generation config value.
            is_correct: Correctness label(s) for RACE.

        Returns:
            :class:`LLMSampleResult` containing generation metadata
            (activations are **not** included — they stay in the recorder).
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded")

        self._reset_activations()

        # --- normalise inputs to batch format ---
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(sample_indices, int):
            sample_indices = [sample_indices]
        if isinstance(is_correct, bool):
            is_correct = [is_correct] * len(prompts)

        batch_size = len(prompts)
        if len(sample_indices) != batch_size:
            raise ValueError(
                f"prompts ({batch_size}) and sample_indices "
                f"({len(sample_indices)}) length mismatch"
            )

        # --- first-token-only: limit generation to 2 tokens ---
        # We need max_new_tokens >= 2 (not 1) because with KV cache:
        #   step 0 = prefill (prompt forward pass, no decode activations)
        #   step 1 = first decode step (processes the 1st generated token)
        # iter_sample_activations skips step 0, so step 1 must exist.
        # With max_new_tokens=1, only the prefill forward pass occurs and
        # no decode-step activations are captured, resulting in empty evidence.
        if self.first_token_only and max_new_tokens != 2:
            logger.debug(
                "first_token_only=True: overriding max_new_tokens %s -> 2",
                max_new_tokens,
            )
            max_new_tokens = 2

        # --- tokenise ---
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        padded_prompt_len: int = inputs["input_ids"].shape[1]

        # --- build generation kwargs ---
        gen_kwargs: Dict[str, Any] = {}
        generation_config = getattr(self.model, "generation_config", None)
        if generation_config is None:
            generation_config = getattr(
                getattr(self.model, "_orig_mod", None),
                "generation_config",
                None,
            )
        if (
            getattr(generation_config, "pad_token_id", None) is None
            and self.tokenizer.pad_token_id is not None
        ):
            gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        if (
            getattr(generation_config, "eos_token_id", None) is None
            and self.tokenizer.eos_token_id is not None
        ):
            gen_kwargs["eos_token_id"] = self.tokenizer.eos_token_id
        if max_new_tokens is not None:
            gen_kwargs["max_new_tokens"] = max_new_tokens
        if do_sample is not None:
            gen_kwargs["do_sample"] = do_sample
        if do_sample is not False:
            if temperature is not None:
                gen_kwargs["temperature"] = temperature
            if top_p is not None:
                gen_kwargs["top_p"] = top_p

        # --- generate ---
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        # --- compute per-sample generated-token counts ---
        # Only search for EOS in the *generated* portion to avoid false
        # positives from left-padding (which may use eos_token as pad_token).
        eos_id = self.tokenizer.eos_token_id
        num_generated_tokens: List[int] = []
        hit_max_without_eos: List[int] = []
        for i in range(batch_size):
            gen_tokens = outputs[i, padded_prompt_len:]
            if eos_id is not None:
                eos_mask = gen_tokens == eos_id
                if eos_mask.any():
                    gen_len = int(eos_mask.nonzero(as_tuple=True)[0][0].item()) + 1
                else:
                    gen_len = gen_tokens.shape[0]
                    if max_new_tokens is not None and gen_len >= max_new_tokens:
                        hit_max_without_eos.append(i)
            else:
                gen_len = gen_tokens.shape[0]
                if max_new_tokens is not None and gen_len >= max_new_tokens:
                    hit_max_without_eos.append(i)
            num_generated_tokens.append(gen_len)

        if hit_max_without_eos and not self.first_token_only:
            # Generation hit max_new_tokens without EOS — likely truncated.
            # Log warning and mark these samples as incorrect so they are skipped
            # for evidence accumulation; continue processing the rest.
            sample_ids = [sample_indices[i] for i in hit_max_without_eos]
            fail_token_lists: List[List[int]] = []
            for i in hit_max_without_eos:
                n = num_generated_tokens[i]
                fail_token_lists.append(
                    outputs[i, padded_prompt_len : padded_prompt_len + n].tolist()
                )
            fail_texts: List[str]
            try:
                fail_texts = self.tokenizer.batch_decode(
                    fail_token_lists, skip_special_tokens=True
                )
            except Exception:
                fail_texts = ["<decode_failed>"] * len(hit_max_without_eos)

            max_preview_chars = 2000
            previews: List[str] = []
            for batch_pos, sample_id, text in zip(
                hit_max_without_eos, sample_ids, fail_texts
            ):
                txt = text or ""
                if len(txt) > max_preview_chars:
                    txt = txt[:max_preview_chars] + " ..."
                previews.append(
                    f"batch_pos={batch_pos}, sample_index={sample_id}, generated={txt}"
                )
            preview_text = "\n".join(previews)

            logger.warning(
                "Generation hit max_new_tokens without EOS (likely truncated), "
                f"skipping these samples for evidence: max_new_tokens={max_new_tokens}, "
                f"batch_positions={hit_max_without_eos}, sample_indices={sample_ids}\n"
                f"{preview_text}"
            )

            # Mark truncated samples as incorrect so accumulator skips them
            is_correct_list = list(is_correct)
            for i in hit_max_without_eos:
                is_correct_list[i] = False
            is_correct = is_correct_list
        elif hit_max_without_eos and self.first_token_only:
            # In first-token-only mode we intentionally cap generation at 2
            # decode steps to expose the first generated token activation.
            # Not reaching EOS here is expected and should not be treated
            # as truncation/failure.
            logger.debug(
                "first_token_only=True: hit max_new_tokens=2 without EOS is expected; "
                "keeping samples for evidence"
            )

        # Always keep generated token ids for downstream token-level selection.
        gen_id_lists: List[List[int]] = []
        for i in range(batch_size):
            n = num_generated_tokens[i]
            gen_id_lists.append(
                outputs[i, padded_prompt_len : padded_prompt_len + n].tolist()
            )

        # Decode generated text only in debug mode to avoid extra overhead.
        if logger.isEnabledFor(logging.DEBUG):
            generated_texts = self.tokenizer.batch_decode(
                gen_id_lists, skip_special_tokens=True
            )
        else:
            generated_texts = None

        return LLMSampleResult(
            sample_indices=sample_indices,
            prompts=prompts,
            generated_texts=generated_texts,
            generated_token_ids=gen_id_lists,
            is_correct=is_correct,
            batch_size=batch_size,
            num_generated_tokens=num_generated_tokens,
        )

    def record_sample_teacher_forcing(
        self,
        prompts: Union[str, List[str]],
        sample_indices: Union[int, List[int]],
        is_correct: Union[bool, List[bool]] = True,
    ) -> LLMSampleResult:
        """Run a single prefill forward pass and capture per-token activations.

        Unlike :meth:`record_sample`, no tokens are generated.  The input
        sequence itself is the evidence source (teacher-forcing / raw-text
        corpora such as WikiText-2).

        Activations are stored as T individual ``[B, 1, dim]`` slices so that
        :meth:`iter_sample_activations` works without modification.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded")

        self._reset_activations()

        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(sample_indices, int):
            sample_indices = [sample_indices]
        if isinstance(is_correct, bool):
            is_correct = [is_correct] * len(prompts)

        batch_size = len(prompts)

        # Swap generation hooks for prefill-capture hooks.
        self.remove_hooks()
        backbone = get_language_model(self.model)
        layers = get_decoder_layers(backbone)
        for layer_idx, layer in enumerate(layers):
            for finder_fn, act_type in [
                (find_attention_out_proj, "attn_pre_out"),
                (find_mlp_down_proj, "mlp_pre_down"),
            ]:
                module = finder_fn(layer)
                if module is not None:
                    hook = self._make_prefill_hook(layer_idx, act_type)
                    self.hooks.append(module.register_forward_pre_hook(hook))

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        seq_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            self.model(**inputs)

        # Restore generation hooks for subsequent record_sample calls.
        self.remove_hooks()
        self.register_hooks()

        # Each sample contributes seq_len "generated" tokens (all input positions).
        num_tokens = [seq_len] * batch_size

        return LLMSampleResult(
            sample_indices=sample_indices,
            prompts=prompts,
            generated_texts=None,
            generated_token_ids=None,
            is_correct=list(is_correct),
            batch_size=batch_size,
            num_generated_tokens=num_tokens,
        )

    # ------------------------------------------------------------------
    # Streaming evidence
    # ------------------------------------------------------------------

    def iter_sample_activations(
        self,
        result: LLMSampleResult,
        token_masks: Optional[List[List[bool]]] = None,
    ) -> Iterator[Tuple[int, Dict[str, Dict[int, torch.Tensor]]]]:
        """Yield per-sample activations from the stored batch tensors.

        For each sample *i* in the batch, the activations dict has the
        format::

            {
                "attn_pre_out": {layer_idx: tensor},   # [T, d_attn]
                "mlp_pre_down": {layer_idx: tensor},   # [T, d_ffn]
            }

        where T is the number of generated tokens (or 1 for
        ``first_token_only`` mode).  All decode steps for a layer are
        pre-concatenated into a single ``[T, dim]`` tensor so that the
        downstream evidence matmul operates on the full token block at
        once rather than on T separate tiny tensors.

        After the iterator is fully consumed (or if an exception occurs),
        **all stored activations are released**.

        Yields:
            ``(batch_index, activations_dict)``
        """
        try:
            for i in range(result.batch_size):
                num_gen = result.num_generated_tokens[i]
                max_step = 1 if self.first_token_only else num_gen
                mask_i = token_masks[i] if token_masks is not None else None

                sample_acts: Dict[str, Dict[int, torch.Tensor]] = {
                    "attn_pre_out": {},
                    "mlp_pre_down": {},
                }

                for act_key, act_store in (
                    ("attn_pre_out", self.attn_pre_out_activations),
                    ("mlp_pre_down", self.mlp_pre_down_activations),
                ):
                    for layer_idx, step_list in act_store.items():
                        upper = min(max_step, len(step_list))
                        if upper == 0:
                            continue
                        # step_list[s]: [B, 1, dim] per decode step (prefill excluded).
                        # cat along the token dim → [B, T, dim], then index
                        # sample i → [T, dim].  One kernel launch instead of T
                        # separate slice ops followed by a cat in the accumulator.
                        stacked = torch.cat(step_list[:upper], dim=1)  # [B, T, dim]
                        sample_tensor = stacked[i]  # [T, dim]
                        if mask_i is not None:
                            if len(mask_i) < sample_tensor.shape[0]:
                                # Allow shorter masks by padding with False.
                                padded = mask_i + [False] * (
                                    sample_tensor.shape[0] - len(mask_i)
                                )
                                mask_i_t = padded
                            else:
                                mask_i_t = mask_i[: sample_tensor.shape[0]]
                            mask_tensor = torch.as_tensor(
                                mask_i_t,
                                dtype=torch.bool,
                                device=sample_tensor.device,
                            )
                            sample_tensor = sample_tensor[mask_tensor]
                        sample_acts[act_key][layer_idx] = sample_tensor

                yield i, sample_acts
        finally:
            self.clear_activations()

    def stream_evidence(
        self,
        result: LLMSampleResult,
        accumulator: Any,
        domain: str,
        token_masks: Optional[List[List[bool]]] = None,
    ) -> int:
        """Process RACE evidence sample-by-sample, then release activations.

        This is the recommended way to consume activations after
        :meth:`record_sample`.  Each sample's activations are extracted
        on-the-fly from the stored batch tensors, fed to *accumulator*,
        and allowed to be garbage-collected.

        Args:
            result: Metadata returned by :meth:`record_sample`.
            accumulator: An ``OnlineRACEAccumulator`` (or compatible object
                with an ``accumulate_from_activations`` method).
            domain: RACE domain label (e.g. ``"code_generation"``).

        Returns:
            Number of samples processed.
        """
        count = 0
        for i, sample_acts in self.iter_sample_activations(
            result,
            token_masks=token_masks,
        ):
            accumulator.accumulate_from_activations(
                domain=domain,
                activations=sample_acts,
                is_correct=result.is_correct[i],
                instance_idx=result.sample_indices[i],
            )
            count += 1
        return count
