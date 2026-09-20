"""Python comprehension statements dataset loader for LLM-RACE analysis.

Dataset: https://huggingface.co/datasets/nexround/py_comprehension_statements_1k

This is a prompt-only corpus of Python statements containing list/set/dict
comprehensions or generator expressions.  It is intended for teacher-forcing
RACE runs over the code statements themselves, rather than generation-based
benchmarking.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from datasets import Dataset, DatasetDict, load_dataset

logger = logging.getLogger(__name__)


PY_COMPREHENSION_STATEMENTS_DEFAULTS: dict[str, Any] = {
    "instruction_prefix": "",
    "response_prefix": "",
}

_HF_DATASET = "nexround/py_comprehension_statements_1k"


def load_py_comprehension_statements(
    max_samples: Optional[int] = None,
    dataset_name: str = _HF_DATASET,
    split: Optional[str] = None,
) -> Dataset:
    """Load Python comprehension statements with a ``prompt`` field.

    Args:
        max_samples: Maximum number of samples to return.
        dataset_name: HuggingFace dataset identifier.
        split: Dataset split to load. Defaults to ``"train"``.

    Returns:
        HuggingFace :class:`~datasets.Dataset` with ``prompt`` copied from
        the source ``statement`` field.
    """
    effective_split = split or "train"
    logger.info("Loading %s (split=%s) ...", dataset_name, effective_split)

    raw = load_dataset(dataset_name, split=effective_split)
    if isinstance(raw, DatasetDict):
        if effective_split in raw:
            raw = raw[effective_split]
        elif "train" in raw:
            raw = raw["train"]
        else:
            raw = raw[next(iter(raw.keys()))]

    if not isinstance(raw, Dataset):
        raise ValueError(f"Expected Dataset after split selection, got {type(raw)!r}")

    if max_samples is not None:
        raw = raw.select(range(min(max_samples, len(raw))))

    def _add_prompt(example: dict[str, Any]) -> dict[str, Any]:
        example["prompt"] = example.get("statement", "")
        return example

    raw = raw.map(_add_prompt)
    logger.info("Loaded %d samples from %s", len(raw), dataset_name)
    return raw


def format_prompt_for_model(
    example: dict[str, Any],
    tokenizer: Any,
    split: str = "instruct",
    instruction_prefix: str = PY_COMPREHENSION_STATEMENTS_DEFAULTS[
        "instruction_prefix"
    ],
    response_prefix: str = PY_COMPREHENSION_STATEMENTS_DEFAULTS["response_prefix"],
    prefill: bool = False,
    direct_completion: bool = True,
) -> dict[str, Any]:
    """Format a comprehension statement for teacher-forcing RACE.

    The dataset is a raw code corpus, so chat templates and instruction text
    are intentionally ignored.  ``formatted_prompt`` is the Python statement
    itself.
    """
    _ = tokenizer
    _ = split
    _ = instruction_prefix
    _ = response_prefix
    _ = prefill
    _ = direct_completion

    example["formatted_prompt"] = example.get("prompt") or example.get("statement", "")
    return example


def main() -> None:
    """Load a few samples and print formatted statements."""
    import sys

    logging.basicConfig(level=logging.INFO)

    class _FakeTokenizer:
        chat_template = None

    ds = load_py_comprehension_statements(max_samples=3)
    tokenizer = _FakeTokenizer()

    for i in range(len(ds)):
        ex = format_prompt_for_model(dict(ds[i]), tokenizer)
        logger.info(
            "[%d] types=%s\n%s",
            i + 1,
            ex.get("comprehension_types"),
            ex["formatted_prompt"],
        )

    sys.exit(0)


if __name__ == "__main__":
    main()
