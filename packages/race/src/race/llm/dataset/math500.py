"""HuggingFaceH4/MATH-500 dataset loader for LLM online-RACE analysis."""

from __future__ import annotations

import logging
from typing import Any, Optional

from datasets import Dataset, DatasetDict, load_dataset

logger = logging.getLogger(__name__)


MATH500_DEFAULTS: dict[str, Any] = {
    # Required by the dataset adapter contract. The formatter builds the prompt.
    "instruction_prefix": "",
    "response_prefix": "",
}


_MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"


def _find_last_boxed_span(text: str) -> Optional[tuple[int, int]]:
    """Return the character span for the last ``\\boxed{...}`` content.

    The returned span excludes the surrounding ``\\boxed{`` and ``}``.
    Supports nested races inside the boxed payload.
    """
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None

    i = start + len(marker)
    depth = 1
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return (start + len(marker), i)
        i += 1
    return None


def build_boxed_token_mask(
    generated_token_ids: list[int],
    tokenizer: Any,
) -> list[bool]:
    """Build a token-level mask for the ``\\boxed{...}`` region.

    The mask length equals ``len(generated_token_ids)``. ``True`` means the
    token overlaps with the boxed payload characters; otherwise ``False``.
    If no boxed region is found or token alignment fails, returns all-False.
    """
    if not generated_token_ids:
        return []

    text = tokenizer.decode(
        generated_token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    span = _find_last_boxed_span(text)
    if span is None:
        return [False] * len(generated_token_ids)

    try:
        encoding = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = encoding.get("offset_mapping")
    except Exception:
        offsets = None

    if offsets is None:
        return [False] * len(generated_token_ids)

    if len(offsets) != len(generated_token_ids):
        # Conservative fallback: avoid accumulating evidence from misaligned tokens.
        return [False] * len(generated_token_ids)

    span_start, span_end = span
    mask: list[bool] = []
    for tok_start, tok_end in offsets:
        keep = tok_end > span_start and tok_start < span_end
        mask.append(bool(keep))
    return mask


def _load_split_auto(dataset_name: str, preferred_split: Optional[str]) -> Dataset:
    """Load a HF dataset, picking a split deterministically if needed."""
    if preferred_split is not None:
        ds = load_dataset(dataset_name, split=preferred_split)
        if not isinstance(ds, Dataset):
            raise ValueError("Expected Dataset, got DatasetDict")
        return ds

    dd = load_dataset(dataset_name)
    if isinstance(dd, Dataset):
        return dd
    if not isinstance(dd, DatasetDict):
        raise ValueError(f"Unexpected dataset type: {type(dd)!r}")

    # Common conventions first, otherwise pick the first split key.
    for key in ("test", "validation", "train"):
        if key in dd:
            return dd[key]
    first_key = next(iter(dd.keys()))
    return dd[first_key]


def load_math500(
    max_samples: Optional[int] = None,
    dataset_name: str = "HuggingFaceH4/MATH-500",
    split: Optional[str] = None,
) -> Dataset:
    """Load MATH-500 dataset with a basic ``prompt`` field.

    Expected columns include: ``problem``, ``solution``, ``level``, ``type``.
    """
    logger.info("Loading %s (split=%s) ...", dataset_name, split or "<auto>")
    dataset = _load_split_auto(dataset_name, split)

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    def _add_prompt(example: dict[str, Any]) -> dict[str, Any]:
        example["prompt"] = example.get("problem", "")
        return example

    dataset = dataset.map(_add_prompt)
    logger.info("Loaded %d samples", len(dataset))
    return dataset


def format_prompt_for_model(
    example: dict[str, Any],
    tokenizer: Any,
    split: str = "instruct",
    instruction_prefix: str = MATH500_DEFAULTS["instruction_prefix"],
    response_prefix: str = MATH500_DEFAULTS["response_prefix"],
    prefill: bool = True,
    direct_completion: bool = False,
) -> dict[str, Any]:
    """Format prompt for model using chat template (for ``dataset.map``)."""
    _ = split  # Required by the shared dataset formatter signature.

    question = example.get("prompt") or example.get("problem") or ""
    task_prompt = (
        f"{question.strip()}\n\n"
        "Please reason step by step, and put your final answer within \\boxed{{}}."
    )

    if tokenizer.chat_template is None or direct_completion:
        _ = instruction_prefix
        example["formatted_prompt"] = f"{task_prompt}\n"
        return example

    formatted_instruction = task_prompt

    if prefill:
        _ = response_prefix
        response = _MAGIC_SPLITTER_
        formatted_prompt = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": formatted_instruction},
                {"role": "assistant", "content": response},
            ],
            tokenize=False,
        ).split(_MAGIC_SPLITTER_)[0]
    else:
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": formatted_instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )

    example["formatted_prompt"] = formatted_prompt
    return example
