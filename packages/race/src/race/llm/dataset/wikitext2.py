"""WikiText-2 dataset loader for LLM perplexity evaluation.

Dataset: https://huggingface.co/datasets/Salesforce/wikitext
Subset:  wikitext-2-raw-v1

WikiText-2 is the standard benchmark for language-model perplexity.  Unlike
the other corpora in this package it is **raw prose text** — there is no
question/answer structure and no model generation is required.  The text
passages are fed directly into the model for teacher-forcing evaluation.

Loading strategy:
- Non-empty, non-header lines (len > ``min_chars``) are treated as individual
  samples.
- Consecutive short lines are merged into paragraphs up to ``max_chars``
  characters so that sequences are long enough to be meaningful for PPL.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)


WIKITEXT2_DEFAULTS: dict[str, Any] = {
    "instruction_prefix": "",
    "response_prefix": "",
}

_HF_DATASET = "Salesforce/wikitext"
_HF_SUBSET = "wikitext-2-raw-v1"

# Lines shorter than this (after stripping) are skipped as noise/headers.
_MIN_LINE_CHARS = 80

# Merge consecutive lines into paragraphs up to this many characters.
_MAX_PARAGRAPH_CHARS = 2000


def _build_paragraphs(
    raw_lines: list[str], min_chars: int, max_chars: int
) -> list[str]:
    """Merge raw WikiText lines into coherent paragraphs.

    WikiText-2 raw format has many empty lines and short section-header lines
    (e.g. ``" = Homarus americanus = "``).  We:
    1. Drop empty lines and header-style lines (all ``=`` signs).
    2. Accumulate normal lines into paragraphs capped at ``max_chars`` chars.
    3. Only keep paragraphs that exceed ``min_chars`` when finalised.
    """
    paragraphs: list[str] = []
    buf: list[str] = []
    buf_len: int = 0

    def _flush():
        text = " ".join(buf).strip()
        if len(text) >= min_chars:
            paragraphs.append(text)
        buf.clear()
        nonlocal buf_len
        buf_len = 0

    for line in raw_lines:
        line = line.strip()
        if not line:
            if buf:
                _flush()
            continue
        # Skip section headers like " = Title = " or " = = Sub = = "
        stripped = line.replace("=", "").strip()
        if not stripped:
            if buf:
                _flush()
            continue

        if buf_len + len(line) > max_chars and buf:
            _flush()

        buf.append(line)
        buf_len += len(line)

    if buf:
        _flush()

    return paragraphs


def load_wikitext2(
    max_samples: Optional[int] = None,
    dataset_name: str = _HF_DATASET,
    subset: str = _HF_SUBSET,
    split: str = "test",
    min_chars: int = _MIN_LINE_CHARS,
    max_chars: int = _MAX_PARAGRAPH_CHARS,
) -> Dataset:
    """Load WikiText-2 as a collection of prose paragraphs.

    Args:
        max_samples: Maximum number of paragraphs to return.
        dataset_name: HuggingFace dataset identifier.
        subset: Dataset config / subset name.
        split: Dataset split (``"test"`` for standard PPL evaluation).
        min_chars: Minimum paragraph length in characters.
        max_chars: Maximum paragraph length before splitting.

    Returns:
        HuggingFace :class:`Dataset` with ``text`` and ``prompt`` fields.
        ``prompt`` is always an empty string — the full ``text`` is the
        evaluation sequence.
    """
    logger.info("Loading %s/%s (split=%s) ...", dataset_name, subset, split)
    raw = load_dataset(dataset_name, subset, split=split)

    raw_lines: list[str] = [row["text"] for row in raw]
    paragraphs = _build_paragraphs(raw_lines, min_chars=min_chars, max_chars=max_chars)

    if max_samples is not None:
        paragraphs = paragraphs[:max_samples]

    dataset = Dataset.from_dict(
        {
            "text": paragraphs,
            "prompt": [""] * len(paragraphs),
        }
    )

    logger.info(
        "Loaded %d paragraphs from %s/%s (split=%s)",
        len(dataset),
        dataset_name,
        subset,
        split,
    )
    return dataset


def format_prompt_for_model(
    example: dict[str, Any],
    tokenizer: Any,
    split: str = "instruct",
    instruction_prefix: str = WIKITEXT2_DEFAULTS["instruction_prefix"],
    response_prefix: str = WIKITEXT2_DEFAULTS["response_prefix"],
    prefill: bool = False,
    direct_completion: bool = True,
) -> dict[str, Any]:
    # Raw text is fed directly as the sequence for teacher-forcing evidence.
    example["formatted_prompt"] = example["text"]
    return example
