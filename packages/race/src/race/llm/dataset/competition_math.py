"""qwedsacf/competition_math dataset loader for LLM online-RACE analysis."""

from __future__ import annotations

import logging
from typing import Any, Optional

from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)

COMPETITION_MATH_DEFAULTS: dict[str, Any] = {
    "instruction_prefix": "",
    "response_prefix": "",
}

_MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"


def load_competition_math(
    max_samples: Optional[int] = None,
    dataset_name: str = "qwedsacf/competition_math",
    split: Optional[str] = None,
) -> Dataset:
    """Load competition_math dataset with a basic ``prompt`` field.

    Expected columns: ``problem``, ``solution``, ``level``, ``type``.
    """
    _split = split or "train"
    logger.info("Loading %s (split=%s) ...", dataset_name, _split)
    dataset = load_dataset(dataset_name, split=_split)
    if not isinstance(dataset, Dataset):
        raise ValueError(f"Expected Dataset, got {type(dataset)!r}")

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    dataset = dataset.map(lambda ex: {**ex, "prompt": ex.get("problem", "")})
    logger.info("Loaded %d samples", len(dataset))
    return dataset


def format_prompt_for_model(
    example: dict[str, Any],
    tokenizer: Any,
    split: str = "instruct",
    instruction_prefix: str = COMPETITION_MATH_DEFAULTS["instruction_prefix"],
    response_prefix: str = COMPETITION_MATH_DEFAULTS["response_prefix"],
    prefill: bool = True,
    direct_completion: bool = False,
) -> dict[str, Any]:
    _ = split
    question = example.get("prompt") or example.get("problem") or ""
    task_prompt = (
        f"{question.strip()}\n\n"
        "Please reason step by step, and put your final answer within \\boxed{{}}."
    )

    if tokenizer.chat_template is None or direct_completion:
        example["formatted_prompt"] = f"{task_prompt}\n"
        return example

    if prefill:
        formatted_prompt = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": task_prompt},
                {"role": "assistant", "content": _MAGIC_SPLITTER_},
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
