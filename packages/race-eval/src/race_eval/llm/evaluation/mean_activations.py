"""Compute per-module mean activations on a calibration corpus.

Used for **mean ablation**: instead of zeroing out target neurons (which
breaks the residual stream norm and pushes the model OOD), we replace
them with their empirical expectation computed on a neutral corpus such
as WikiText-2.

The collection uses Welford's online algorithm to accumulate running means
without materialising all activations in memory.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from race.llm.model_utils import (
    find_attention_out_proj,
    find_mlp_down_proj,
    get_decoder_layers,
    get_language_model,
)

logger = logging.getLogger(__name__)


def _load_calibration_texts(
    corpus: str = "wikitext2",
    max_samples: int = 200,
) -> List[str]:
    """Load raw text passages from the calibration corpus."""
    if corpus == "wikitext2":
        from race.llm.dataset.wikitext2 import load_wikitext2

        dataset = load_wikitext2(max_samples=max_samples)
        return [row["text"] for row in dataset if row["text"].strip()]
    else:
        raise ValueError(f"Unsupported calibration corpus: {corpus!r}")


@torch.inference_mode()
def compute_mean_activations(
    model: nn.Module,
    tokenizer: Any,
    calibration_corpus: str = "wikitext2",
    max_samples: int = 200,
    max_length: int = 1024,
    batch_size: int = 4,
) -> Dict[str, torch.Tensor]:
    """Compute per-dimension mean activations at each hook point.

    Registers temporary ``forward_pre_hook`` on every attention output
    projection and MLP down projection, runs the calibration corpus
    through the model, then returns a dict mapping hook keys to their
    mean activation vectors.

    Keys follow the pattern ``"attn_layer_{i}"`` and ``"mlp_layer_{i}"``.

    Args:
        model: HuggingFace causal LM (eval mode, on device).
        tokenizer: Corresponding tokenizer.
        calibration_corpus: Name of calibration corpus (``"wikitext2"``).
        max_samples: Maximum number of calibration passages.
        max_length: Token-length cap per passage.
        batch_size: Forward-pass batch size.

    Returns:
        Dict mapping ``"attn_layer_{i}"`` / ``"mlp_layer_{i}"`` to 1-D
        mean-activation tensors (on CPU, float64 for precision).
    """
    texts = _load_calibration_texts(calibration_corpus, max_samples)
    if not texts:
        raise RuntimeError(f"No calibration texts from {calibration_corpus!r}")

    logger.info(
        "Computing mean activations on %d passages from %s ...",
        len(texts),
        calibration_corpus,
    )

    lm = get_language_model(model)
    layers = list(get_decoder_layers(lm))
    device = next(model.parameters()).device

    # Welford accumulators: running mean and count per hook point
    accumulators: Dict[str, Tuple[torch.Tensor, int]] = {}
    hooks: List[torch.utils.hooks.RemovableHandle] = []

    def _make_collector(key: str):
        """Return a hook that updates the Welford running mean for *key*."""

        def hook(module: nn.Module, inputs):
            if len(inputs) == 0:
                return
            hidden = inputs[0]
            if not isinstance(hidden, torch.Tensor):
                return

            # hidden: [batch, seq_len, dim] or [batch, dim]
            # Flatten to [N, dim] then compute batch mean
            flat = hidden.reshape(-1, hidden.shape[-1]).to(torch.float64)
            batch_mean = flat.mean(dim=0)  # [dim]
            batch_n = flat.shape[0]

            if key not in accumulators:
                accumulators[key] = (batch_mean.cpu(), batch_n)
            else:
                prev_mean, prev_n = accumulators[key]
                new_n = prev_n + batch_n
                # Welford update: combined_mean = prev + (batch_mean - prev) * batch_n / new_n
                delta = batch_mean.cpu() - prev_mean
                accumulators[key] = (prev_mean + delta * (batch_n / new_n), new_n)

        return hook

    # Register collectors on every layer
    for layer_idx, layer in enumerate(layers):
        attn_proj = find_attention_out_proj(layer)
        if attn_proj is not None:
            key = f"attn_layer_{layer_idx}"
            h = attn_proj.register_forward_pre_hook(_make_collector(key))
            hooks.append(h)

        mlp_proj = find_mlp_down_proj(layer)
        if mlp_proj is not None:
            key = f"mlp_layer_{layer_idx}"
            h = mlp_proj.register_forward_pre_hook(_make_collector(key))
            hooks.append(h)

    # Run calibration forward passes
    try:
        _run_calibration_forward(
            model, tokenizer, texts, max_length, batch_size, device
        )
    finally:
        for h in hooks:
            h.remove()

    # Extract final means
    result: Dict[str, torch.Tensor] = {}
    for key, (mean_tensor, count) in accumulators.items():
        result[key] = mean_tensor  # float64 on CPU
        logger.debug("  %s: dim=%d, tokens=%d", key, mean_tensor.shape[0], count)

    logger.info(
        "Mean activations computed for %d hook points (%d layers).",
        len(result),
        len(layers),
    )
    return result


def _run_calibration_forward(
    model: nn.Module,
    tokenizer: Any,
    texts: List[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> None:
    """Run teacher-forcing forward passes on calibration texts."""
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]

        encodings = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        model(input_ids=input_ids, attention_mask=attention_mask)

        if (start // batch_size + 1) % 10 == 0:
            logger.info(
                "  Calibration: %d/%d passages",
                min(start + batch_size, len(texts)),
                len(texts),
            )
