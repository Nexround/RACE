"""Prefill-based activation recorder for RACE analysis.

Instead of capturing activations during autoregressive decode (one token at a
time with T sequential forward passes), this recorder performs a **single
forward pass** on the concatenated ``[prompt_tokens + generated_tokens]``
sequence and extracts activations only at the generated-token positions.

Mathematical validity
---------------------
With causal attention masking, position *t* can only attend to positions
``0 … t``.  Whether we compute position *t* in a single batched prefill or in
an incremental decode step *t*, the hidden state is **identical** — the same
set of earlier positions contributes via the same causal mask.  Therefore the
activations captured here are equivalent to those that the decode-based
:class:`~race.llm.recording.activation.LLMActivationRecorder` would have
captured during generation.

Design
------
- ``record_from_token_ids()`` accepts pre-generated token IDs (produced by
  Stage 1 / ``vllm_generate.py``) and runs a single ``model.forward()`` call
  with right-padded sequences and a proper ``attention_mask``.
- Forward-pre hooks on each ``o_proj`` / ``down_proj`` module slice out the
  generated-token portion ``activation[:, prompt_len_i : prompt_len_i + T_i, :]``
  for every sample *i* in the batch.
- The storage layout is ``{layer_idx: [tensor_sample_0, …, tensor_sample_{B-1}]}``
  where each tensor has shape ``[T_i, dim]`` — directly compatible with the
  ``accumulate_from_activations`` downstream interface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

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
class PrefillSampleResult:
    """Metadata for a batch processed by :class:`PrefillActivationRecorder`.

    Activations are **not** stored here; they remain inside the recorder
    until consumed by :meth:`PrefillActivationRecorder.stream_evidence`.
    """

    sample_indices: List[int]
    prompts: List[str]
    generated_texts: List[str]
    generated_token_ids: List[List[int]]
    is_correct: List[bool]
    batch_size: int
    num_generated_tokens: List[int]


# ======================================================================
# Recorder
# ======================================================================


class PrefillActivationRecorder(BaseActivationRecorder):
    """Record LLM activations via single-pass prefill for RACE analysis.

    Workflow per batch:

    1. Build full token sequences: ``[prompt_tokens_i] + [generated_tokens_i]``
       for each sample *i*.
    2. Right-pad the batch to a uniform length and create an ``attention_mask``.
    3. Run one ``model.forward()`` call with ``use_cache=False``.
    4. Forward-pre hooks capture ``activation[:, start_i:end_i, :]`` for the
       generated positions of each sample.
    5. :meth:`stream_evidence` consumes the captured tensors and feeds them to
       the RACE accumulator, then frees GPU/CPU memory.

    The hook storage format ``{layer_idx: [tensor[T_i, dim] for i in batch]}``
    is equivalent to what
    :class:`~race.llm.recording.activation.LLMActivationRecorder` produces
    after concatenating all decode steps, so both recorders use the same
    downstream accumulator interface.
    """

    def __init__(
        self,
        model_name: str,
        output_dir: str,
        activation_device: str = "cuda",
    ):
        """Initialise the prefill activation recorder.

        Args:
            model_name: HuggingFace model identifier.
            output_dir: Directory used by the parent class for bookkeeping.
            activation_device: Device on which captured activations are stored.
                ``"cpu"`` saves GPU memory; ``"cuda"`` avoids CPU transfers.
        """
        super().__init__(model_name, output_dir)

        self.tokenizer: Optional[Any] = None
        self.model_dtype: torch.dtype = torch.bfloat16
        self.activation_storage_device = torch.device(activation_device)

        # Per-layer activation storage populated by hooks during forward pass.
        # Layout: {layer_idx: [tensor_sample_0[T_0, dim], ..., tensor_sample_{B-1}[T_{B-1}, dim]]}
        self.attn_pre_out_activations: Dict[int, List[torch.Tensor]] = {}
        self.mlp_pre_down_activations: Dict[int, List[torch.Tensor]] = {}

        # Injected before each forward call; hooks read this to know which
        # token positions to extract.  Always reset to None afterwards.
        self._current_gen_slices: Optional[List[Tuple[int, int]]] = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(
        self,
        device: str = "cuda",
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Load the language model and tokenizer.

        Uses ``flash_attention_2`` and ``use_cache=False`` (no KV cache needed
        for a single full-sequence forward pass).

        Args:
            device: Target device (``"cuda"`` or ``"cpu"``).
            dtype: Model precision (default: ``bfloat16``).
        """
        logger.info("Loading model '%s' ...", self.model_name)
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_fast=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Right-padding: position 0 = first real token.  Real tokens come
        # before padding so slicing [prompt_len : prompt_len + T] is correct
        # without any offset adjustment.
        self.tokenizer.padding_side = "right"

        if dtype is None:
            dtype = torch.bfloat16
        self.model_dtype = dtype

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=dtype,
            device_map=device,
            trust_remote_code=True,
            use_cache=False,
            attn_implementation="flash_attention_2",
        )
        self.model.eval()

        logger.info(
            "Model loaded on %s (dtype=%s, use_cache=False, flash_attention_2)",
            self.device,
            self.model_dtype,
        )

    # ------------------------------------------------------------------
    # Weight extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_weight(
        weight: Optional[torch.Tensor],
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Detach, cast, and make contiguous."""
        if weight is None:
            return None
        tensor = weight.detach()
        if tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        return tensor.contiguous()

    def extract_layer_weights(self) -> Dict[int, LayerWeights]:
        """Extract ``W_O`` and ``W_down`` projection weights for RACE.

        Returns:
            Mapping ``{layer_idx: LayerWeights}`` compatible with
            :class:`~race.core.online_accumulator.OnlineRACEAccumulator`.
        """
        if self.model is None:
            raise RuntimeError("Model must be loaded first via load_model()")

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
        """Register forward-pre hooks on all attention and MLP projection modules."""
        if self.model is None:
            raise RuntimeError("Model must be loaded first via load_model()")

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

        logger.info("Registered %d hooks on %d layers", len(self.hooks), len(layers))

    def _make_prefill_hook(self, layer_idx: int, activation_type: str):
        """Build a forward-pre hook that captures the generated-position slice.

        During the single full-sequence forward pass (``[B, S, dim]``):

        - ``S = max(prompt_len_i + gen_len_i)`` (right-padded to batch max).
        - For sample *i*, the generated tokens occupy positions
          ``[prompt_len_i, prompt_len_i + T_i)``.
        - The hook reads ``self._current_gen_slices`` (set in
          :meth:`record_from_token_ids`) to find these position ranges.

        Single-token steps (``seq_len == 1``) — which would occur if
        ``model.generate()`` were accidentally called with this recorder
        attached — are silently ignored.
        """
        storage_device = self.activation_storage_device
        store = (
            self.attn_pre_out_activations
            if activation_type == "attn_pre_out"
            else self.mlp_pre_down_activations
        )
        recorder_ref = self  # captured reference for closure

        def hook(module, inputs):
            if not inputs:
                return
            activation = inputs[0]
            if not isinstance(activation, torch.Tensor):
                return
            # Only capture full-sequence prefill passes (seq_len > 1).
            if activation.ndim != 3 or activation.shape[1] <= 1:
                return
            slices = recorder_ref._current_gen_slices
            if slices is None:
                return

            activation = activation.detach()
            layer_store = store.setdefault(layer_idx, [])

            for i, (start, end) in enumerate(slices):
                if end <= start:
                    # Empty generation — append a zero-sized tensor as placeholder
                    # so that sample indices stay aligned with layer_store[i].
                    placeholder = activation.new_empty(0, activation.shape[2])
                    layer_store.append(placeholder)
                    continue
                gen_act = activation[i, start:end, :]  # [T_i, dim]
                if gen_act.device != storage_device:
                    gen_act = gen_act.to(device=storage_device, non_blocking=True)
                else:
                    gen_act = gen_act.clone()
                layer_store.append(gen_act)

        return hook

    # ------------------------------------------------------------------
    # Activation management
    # ------------------------------------------------------------------

    def clear_activations(self) -> None:
        """Release all stored activation tensors."""
        self.attn_pre_out_activations.clear()
        self.mlp_pre_down_activations.clear()

    def _reset_activations(self) -> None:
        self.clear_activations()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_from_token_ids(
        self,
        prompt_token_ids: List[List[int]],
        generated_token_ids: List[List[int]],
        sample_indices: List[int],
        prompts: Optional[List[str]] = None,
        generated_texts: Optional[List[str]] = None,
        is_correct: Optional[List[bool]] = None,
    ) -> PrefillSampleResult:
        """Run a single prefill forward pass and capture generated-position activations.

        Constructs full sequences ``[prompt_i + generated_i]`` for each sample,
        right-pads to batch maximum length, and runs one ``model.forward()``
        call.  Hooks slice out ``activation[i, prompt_len_i : prompt_len_i + T_i, :]``
        for every layer.

        Args:
            prompt_token_ids: Per-sample prompt token ID lists.
            generated_token_ids: Per-sample generated token ID lists.
            sample_indices: Sample index labels (for RACE bookkeeping).
            prompts: Original prompt strings (metadata only, may be empty).
            generated_texts: Generated text strings (metadata only).
            is_correct: Per-sample correctness labels (default: all ``True``).

        Returns:
            :class:`PrefillSampleResult` with metadata.  Activations remain in
            this recorder; consume them via :meth:`stream_evidence`.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Call load_model() before record_from_token_ids()")

        batch_size = len(prompt_token_ids)
        if is_correct is None:
            is_correct = [True] * batch_size
        if prompts is None:
            prompts = [""] * batch_size
        if generated_texts is None:
            generated_texts = [""] * batch_size

        self._reset_activations()

        num_gen = [len(g) for g in generated_token_ids]
        prompt_lens = [len(p) for p in prompt_token_ids]

        # Full sequences: [prompt_tokens] + [generated_tokens] per sample.
        full_seqs = [p + g for p, g in zip(prompt_token_ids, generated_token_ids)]
        seq_lens = [len(s) for s in full_seqs]
        max_len = max(seq_lens)

        pad_id = self.tokenizer.pad_token_id
        padded = [seq + [pad_id] * (max_len - len(seq)) for seq in full_seqs]
        attn_mask = [[1] * length + [0] * (max_len - length) for length in seq_lens]

        input_ids = torch.tensor(padded, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(attn_mask, dtype=torch.long, device=self.device)

        # Tell hooks where to find generated tokens for each sample.
        # With right-padding: real tokens are at positions [0, seq_len_i).
        # Generated tokens start at prompt_lens[i].
        self._current_gen_slices = [
            (prompt_lens[i], prompt_lens[i] + num_gen[i]) for i in range(batch_size)
        ]

        try:
            with torch.inference_mode():
                self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
        finally:
            self._current_gen_slices = None

        return PrefillSampleResult(
            sample_indices=sample_indices,
            prompts=prompts,
            generated_texts=generated_texts,
            generated_token_ids=generated_token_ids,
            is_correct=is_correct,
            batch_size=batch_size,
            num_generated_tokens=num_gen,
        )

    # ------------------------------------------------------------------
    # Streaming evidence
    # ------------------------------------------------------------------

    def iter_sample_activations(
        self,
        result: PrefillSampleResult,
    ) -> Iterator[Tuple[int, Dict[str, Dict[int, torch.Tensor]]]]:
        """Yield per-sample activations from the stored prefill tensors.

        For each sample *i* the yielded activations dict has the form::

            {
                "attn_pre_out": {layer_idx: tensor[T_i, d_attn]},
                "mlp_pre_down": {layer_idx: tensor[T_i, d_ffn]},
            }

        Unlike the decode-based recorder (which concatenates *T* separate
        ``[1, dim]`` tensors), each tensor here is **already** ``[T_i, dim]``
        because it was sliced directly from the prefill activation matrix.

        All stored tensors are released after the iterator is exhausted.

        Yields:
            ``(batch_index, activations_dict)``
        """
        try:
            for i in range(result.batch_size):
                sample_acts: Dict[str, Dict[int, torch.Tensor]] = {
                    "attn_pre_out": {},
                    "mlp_pre_down": {},
                }
                for act_key, act_store in (
                    ("attn_pre_out", self.attn_pre_out_activations),
                    ("mlp_pre_down", self.mlp_pre_down_activations),
                ):
                    for layer_idx, sample_list in act_store.items():
                        if i < len(sample_list):
                            tensor = sample_list[i]
                            if tensor.shape[0] > 0:  # skip empty placeholders
                                sample_acts[act_key][layer_idx] = tensor
                yield i, sample_acts
        finally:
            self.clear_activations()

    def stream_evidence(
        self,
        result: PrefillSampleResult,
        accumulator: Any,
        domain: str,
    ) -> int:
        """Process RACE evidence sample-by-sample and release activations.

        Mirrors the interface of
        :meth:`~race.llm.recording.activation.LLMActivationRecorder.stream_evidence`
        so that the two recorders are interchangeable in pipeline code.

        Args:
            result: Metadata from :meth:`record_from_token_ids`.
            accumulator: ``OnlineRACEAccumulator`` (or compatible object).
            domain: RACE domain label.

        Returns:
            Number of samples processed.
        """
        count = 0
        for i, sample_acts in self.iter_sample_activations(result):
            accumulator.accumulate_from_activations(
                domain=domain,
                activations=sample_acts,
                is_correct=result.is_correct[i],
                instance_idx=result.sample_indices[i],
            )
            count += 1
        return count
