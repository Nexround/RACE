"""MBPP+ (evalplus/mbppplus) dataset loader for LLM-RACE analysis.

This module provides utilities to load and process the MBPP+ dataset
for RACE attribution analysis on code generation tasks.

Dataset: https://huggingface.co/datasets/evalplus/mbppplus
Columns: task_id (int), code, prompt, source_file, test_imports,
         test_list (list of assert strings), test (full test harness str)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from datasets import Dataset, DatasetDict, load_dataset

logger = logging.getLogger(__name__)


# ======================================================================
# MBPP+ default prompt parameters
# ======================================================================

MBPP_PLUS_DEFAULTS: dict[str, Any] = {
    # Required by the dataset adapter contract. _build_task_prompt uses the
    # fixed template below.
    "instruction_prefix": "",
    "response_prefix": "```python\n",
}

# Matches evalscope's MBPPplusAdapter prompt format exactly.
_PROMPT_TEMPLATE = (
    "You are an expert Python programmer, and here is your task: "
    "{question} Your code should pass these tests:\n\n{tests}"
)

# Magic splitter used to split assistant prefill from generation
_MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"


# ======================================================================
# Dataset loading
# ======================================================================


def load_mbpp_plus(
    max_samples: Optional[int] = None,
    dataset_name: str = "evalplus/mbppplus",
    split: Optional[str] = None,
) -> Dataset:
    """Load MBPP+ dataset with a ``prompt`` field.

    Args:
        max_samples: Maximum number of samples to load.
        dataset_name: HuggingFace dataset identifier.
        split: Dataset split to use (defaults to ``"test"``).

    Returns:
        HuggingFace :class:`Dataset` with a ``prompt`` field set to the
        problem description (English text from the original ``prompt`` column).
    """
    effective_split = split or "test"
    logger.info("Loading %s (split=%s) ...", dataset_name, effective_split)

    raw = load_dataset(dataset_name, split=effective_split)
    if isinstance(raw, DatasetDict):
        # Fallback: pick 'test', then first available split
        for key in ("test", "validation", "train"):
            if key in raw:
                raw = raw[key]
                break
        else:
            raw = raw[next(iter(raw.keys()))]

    if not isinstance(raw, Dataset):
        raise ValueError(f"Expected Dataset after split selection, got {type(raw)!r}")

    if max_samples is not None:
        raw = raw.select(range(min(max_samples, len(raw))))

    # The dataset already has a 'prompt' column (English task description).
    # We keep it as-is; no additional mapping needed.
    logger.info("Loaded %d samples from %s", len(raw), dataset_name)
    return raw


# ======================================================================
# Prompt formatting
# ======================================================================


def _build_task_prompt(example: dict[str, Any]) -> str:
    """Construct the user-facing task string from a dataset example.

    Uses the same prompt template as evalscope's ``MBPPplusAdapter``:

        You are an expert Python programmer, and here is your task: {question}
        Your code should pass these tests:

        {tests}

    ``{tests}`` is the full ``test_list`` joined by newlines (all assertions,
    not just a subset), matching evalscope's ``format_prompt_template``.
    """
    question = (example.get("prompt") or "").strip()
    test_list: list[str] = example.get("test_list") or []
    tests = "\n".join(test_list)
    return _PROMPT_TEMPLATE.format(question=question, tests=tests)


def format_prompt_for_model(
    example: dict[str, Any],
    tokenizer: Any,
    split: str = "instruct",
    instruction_prefix: str = MBPP_PLUS_DEFAULTS["instruction_prefix"],
    response_prefix: str = MBPP_PLUS_DEFAULTS["response_prefix"],
    prefill: bool = True,
    direct_completion: bool = False,
) -> dict[str, Any]:
    """Format prompt for model using chat template (for ``dataset.map``).

    Args:
        example: Dataset example (must contain ``prompt`` and ``test_list``).
        tokenizer: Tokenizer with optional ``chat_template``.
        split: Required by the shared formatter signature; unused here.
        instruction_prefix: Required by the adapter contract; unused here.
        response_prefix: Assistant response prefix (e.g. code-fence opener).
        prefill: Whether to prefill the response prefix into the prompt.
        direct_completion: Skip chat template for base / completion models.

    Returns:
        Example dict with ``formatted_prompt`` field added.
    """
    _ = split
    _ = instruction_prefix  # prompt is fully determined by _PROMPT_TEMPLATE

    task_prompt = _build_task_prompt(example)

    # Base models / no chat template: plain string
    if tokenizer.chat_template is None or direct_completion:
        example["formatted_prompt"] = f"{task_prompt}\n"
        return example

    # Chat-template path
    if prefill:
        assistant_content = f"{response_prefix}\n{_MAGIC_SPLITTER_}\n```\n"
        formatted_prompt = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": task_prompt},
                {"role": "assistant", "content": assistant_content},
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


# ======================================================================
# Quick smoke test
# ======================================================================


def main() -> None:
    """Load a few samples and print formatted prompts (no model needed)."""
    import sys

    logging.basicConfig(level=logging.INFO)

    class _FakeTokenizer:
        chat_template = None

    ds = load_mbpp_plus(max_samples=3)
    tokenizer = _FakeTokenizer()

    for i in range(len(ds)):
        ex = format_prompt_for_model(dict(ds[i]), tokenizer)
        logger.info(
            "[%d] task_id=%s\n%s\n%s",
            i + 1,
            ex.get("task_id", "N/A"),
            "-" * 60,
            ex["formatted_prompt"],
        )

    sys.exit(0)


if __name__ == "__main__":
    main()
