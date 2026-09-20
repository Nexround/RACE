"""GPQA Diamond dataset loader for LLM evaluation.

Dataset: https://huggingface.co/datasets/Idavidrein/gpqa
Subset:  gpqa_diamond (198 graduate-level science questions)

Columns (relevant): Question, Correct Answer, Incorrect Answer 1/2/3
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from datasets import Dataset, DatasetDict, load_dataset

logger = logging.getLogger(__name__)


GPQA_DIAMOND_DEFAULTS: dict[str, Any] = {
    "instruction_prefix": "",
    "response_prefix": "",
}

_MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"


def load_gpqa_diamond(
    max_samples: Optional[int] = None,
    dataset_name: str = "Idavidrein/gpqa",
    split: Optional[str] = "train",
    subset: str = "gpqa_diamond",
) -> Dataset:
    """Load GPQA Diamond dataset.

    Expected columns: ``Question``, ``Correct Answer``,
    ``Incorrect Answer 1``, ``Incorrect Answer 2``, ``Incorrect Answer 3``.

    Args:
        max_samples: Maximum number of samples to load.
        dataset_name: HuggingFace dataset identifier.
        split: Dataset split to use.
        subset: Dataset subset/config name.

    Returns:
        HuggingFace :class:`Dataset` with a ``prompt`` field.
    """
    logger.info("Loading %s/%s (split=%s) ...", dataset_name, subset, split or "<auto>")

    raw = load_dataset(dataset_name, subset, split=split)
    if isinstance(raw, DatasetDict):
        for key in ("train", "test", "validation"):
            if key in raw:
                raw = raw[key]
                break
        else:
            raw = raw[next(iter(raw.keys()))]

    if not isinstance(raw, Dataset):
        raise ValueError(f"Expected Dataset, got {type(raw)!r}")

    if max_samples is not None:
        raw = raw.select(range(min(max_samples, len(raw))))

    def _add_prompt(example: dict[str, Any]) -> dict[str, Any]:
        example["prompt"] = example.get("Question", "")
        return example

    raw = raw.map(_add_prompt)
    logger.info("Loaded %d samples from %s/%s", len(raw), dataset_name, subset)
    return raw


def format_prompt_for_model(
    example: dict[str, Any],
    tokenizer: Any,
    split: str = "instruct",
    instruction_prefix: str = GPQA_DIAMOND_DEFAULTS["instruction_prefix"],
    response_prefix: str = GPQA_DIAMOND_DEFAULTS["response_prefix"],
    prefill: bool = True,
    direct_completion: bool = False,
) -> dict[str, Any]:
    """Format prompt for model using chat template (for ``dataset.map``)."""
    _ = split

    question = example.get("prompt") or example.get("Question") or ""
    task_prompt = (
        f"{question.strip()}\n\n" "Please think step by step and provide your answer."
    )

    if tokenizer.chat_template is None or direct_completion:
        example["formatted_prompt"] = f"{task_prompt}\n"
        return example

    if prefill:
        response = _MAGIC_SPLITTER_
        formatted_prompt = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": task_prompt},
                {"role": "assistant", "content": response},
            ],
            tokenize=False,
        ).split(_MAGIC_SPLITTER_)[0]
    else:
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": task_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    example["formatted_prompt"] = formatted_prompt
    return example
