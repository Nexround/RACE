"""Bounded streaming loader for the FineWeb corpus.

Dataset: https://huggingface.co/datasets/HuggingFaceFW/fineweb

FineWeb is far too large to materialise for an online-RACE run. This adapter
streams the already-randomised sample-10BT config, optionally applies a bounded
shuffle buffer, and stops once the requested number of documents is collected.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)


FINEWEB_DEFAULTS: dict[str, Any] = {
    "instruction_prefix": "",
    "response_prefix": "",
}

_HF_DATASET = "HuggingFaceFW/fineweb"
_HF_SUBSET = "sample-10BT"
_DEFAULT_SPLIT = "train"
_MIN_CHARS = 80
_MAX_CHARS = 2000

# The official subset is already a random sample of FineWeb. A second, modest
# shuffle buffer avoids taking a single Parquet/shard prefix while keeping the
# number of downloaded candidate documents and RAM use bounded. Collecting
# 10,000 rows needs roughly buffer_size + 10,000 streamed candidates (plus rows
# rejected by the minimum-length filter), rather than all 14.9M rows.
_SHUFFLE_BUFFER_SIZE = 10_000


def _truncate_text(text: str, max_chars: int) -> str:
    """Trim a document without leaving a short trailing word fragment."""
    text = text.strip()
    if len(text) <= max_chars:
        return text

    prefix = text[:max_chars]
    boundary = max(prefix.rfind(" "), prefix.rfind("\n"), prefix.rfind("\t"))
    if boundary >= int(max_chars * 0.8):
        prefix = prefix[:boundary]
    return prefix.rstrip()


def load_fineweb(
    max_samples: Optional[int] = None,
    dataset_name: str = _HF_DATASET,
    subset: str = _HF_SUBSET,
    split: Optional[str] = None,
    random_sample: bool = False,
    sample_seed: int = 42,
    shuffle_buffer_size: int = _SHUFFLE_BUFFER_SIZE,
    min_chars: int = _MIN_CHARS,
    max_chars: int = _MAX_CHARS,
) -> Dataset:
    """Stream and materialise a bounded FineWeb sample.

    sample-10BT is itself a random sample of FineWeb. With random_sample=True
    we additionally shuffle the stream using a finite buffer before taking
    documents. This avoids downloading the 27.6 GB sample while also avoiding
    a simple dataset prefix.

    Args:
        max_samples: Exact number of usable documents to collect. Required as
            a safety guard against accidentally materialising FineWeb.
        dataset_name: Hugging Face dataset identifier.
        subset: Dataset config. Defaults to the smallest official sample.
        split: Dataset split. Defaults to train.
        random_sample: Apply a seeded bounded-buffer shuffle before taking.
        sample_seed: Seed for the streaming shuffle.
        shuffle_buffer_size: Number of streamed rows held by the shuffle. The
            stream reads approximately this many extra candidate documents.
        min_chars: Skip documents shorter than this after stripping.
        max_chars: Truncate each selected document to at most this many chars.

    Returns:
        In-memory datasets.Dataset containing exactly max_samples rows and
        only compact provenance metadata.
    """
    if max_samples is None:
        raise ValueError(
            "FineWeb requires max_samples to bound streaming downloads; "
            "pass --max-samples (for example, 10000)."
        )
    if max_samples < 0:
        raise ValueError("max_samples must be >= 0")
    if min_chars < 0:
        raise ValueError("min_chars must be >= 0")
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    if max_chars < min_chars:
        raise ValueError("max_chars must be >= min_chars")
    if shuffle_buffer_size <= 0:
        raise ValueError("shuffle_buffer_size must be > 0")

    if max_samples == 0:
        return Dataset.from_dict(
            {
                "text": [],
                "prompt": [],
                "sample_index": [],
                "source_id": [],
                "dump": [],
                "url": [],
                "language_score": [],
                "source_token_count": [],
            }
        )

    effective_split = split or _DEFAULT_SPLIT
    logger.info(
        "Streaming %s/%s (split=%s, max_samples=%d) ...",
        dataset_name,
        subset,
        effective_split,
        max_samples,
    )
    stream = load_dataset(
        dataset_name,
        name=subset,
        split=effective_split,
        streaming=True,
    )
    if random_sample:
        stream = stream.shuffle(seed=sample_seed, buffer_size=shuffle_buffer_size)
        logger.info(
            "Using bounded streaming shuffle (seed=%d, buffer_size=%d)",
            sample_seed,
            shuffle_buffer_size,
        )

    columns: dict[str, list[Any]] = {
        "text": [],
        "prompt": [],
        "sample_index": [],
        "source_id": [],
        "dump": [],
        "url": [],
        "language_score": [],
        "source_token_count": [],
    }
    skipped_short = 0
    for row in stream:
        raw_text = row.get("text")
        if not isinstance(raw_text, str) or len(raw_text.strip()) < min_chars:
            skipped_short += 1
            continue

        text = _truncate_text(raw_text, max_chars=max_chars)
        sample_index = len(columns["text"])
        columns["text"].append(text)
        columns["prompt"].append("")
        columns["sample_index"].append(sample_index)
        columns["source_id"].append(str(row.get("id") or ""))
        columns["dump"].append(str(row.get("dump") or ""))
        columns["url"].append(str(row.get("url") or ""))
        columns["language_score"].append(row.get("language_score"))
        columns["source_token_count"].append(row.get("token_count"))

        if len(columns["text"]) >= max_samples:
            break

    if len(columns["text"]) != max_samples:
        raise RuntimeError(
            f"FineWeb stream ended after {len(columns['text'])} usable documents; "
            f"requested {max_samples}."
        )

    dataset = Dataset.from_dict(columns)
    logger.info(
        "Materialised %d FineWeb documents (skipped_short=%d, max_chars=%d)",
        len(dataset),
        skipped_short,
        max_chars,
    )
    return dataset


def format_prompt_for_model(
    example: dict[str, Any],
    tokenizer: Any,
    split: str = "instruct",
    instruction_prefix: str = FINEWEB_DEFAULTS["instruction_prefix"],
    response_prefix: str = FINEWEB_DEFAULTS["response_prefix"],
    prefill: bool = False,
    direct_completion: bool = True,
) -> dict[str, Any]:
    """Feed raw FineWeb text directly to teacher-forcing RACE."""
    _ = tokenizer
    _ = split
    _ = instruction_prefix
    _ = response_prefix
    _ = prefill
    _ = direct_completion
    example["formatted_prompt"] = example["text"]
    return example
