"""CodeAlpaca-20k (sahil2801/CodeAlpaca-20k) dataset loader for LLM-RACE analysis.

This module provides utilities to load and process the CodeAlpaca-20k dataset
for RACE attribution analysis on instruction-following code-generation tasks.

Dataset: https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k
Columns:
    instruction (str) – the coding task / question
    input       (str) – optional extra context / function signature (often empty)
    output      (str) – reference Python solution

The dataset has a single split (``"train"``), ~20 k samples covering a wide
range of Python programming tasks derived from the Self-Instruct procedure.

References:
    Chaudhary (2023) "Code Alpaca: An Instruction-following LLaMA model for
    code generation" https://github.com/sahil280114/codealpaca
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from datasets import Dataset, DatasetDict, load_dataset

logger = logging.getLogger(__name__)


# ======================================================================
# CodeAlpaca-20k default prompt parameters
# ======================================================================

CODE_ALPACA_DEFAULTS: dict[str, Any] = {
    # Required by the dataset adapter contract and embedded in the template.
    "instruction_prefix": "Below is an instruction that describes a coding task. Write a response that appropriately completes the request.",
    "response_prefix": "### Response:",
}

# Magic splitter for chat-template prefill split.
_MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"


# ======================================================================
# Helper utilities
# ======================================================================


def _build_prompt(example: dict[str, Any]) -> str:
    """Build the canonical task prompt from a raw dataset example.

    Follows the Alpaca prompt format:

        ### Instruction:
        <instruction>

        ### Input:          ← omitted when empty
        <input>

        ### Response:
    """
    instruction: str = (example.get("instruction") or "").strip()
    extra_input: str = (example.get("input") or "").strip()

    parts: list[str] = [f"### Instruction:\n{instruction}"]
    if extra_input:
        parts.append(f"\n### Input:\n{extra_input}")

    return "\n".join(parts)


# ======================================================================
# Dataset loading
# ======================================================================


def load_code_alpaca(
    max_samples: Optional[int] = None,
    dataset_name: str = "sahil2801/CodeAlpaca-20k",
    split: Optional[str] = None,
) -> Dataset:
    """Load CodeAlpaca-20k with a ``prompt`` field added.

    Args:
        max_samples: Maximum number of samples to return.
        dataset_name: HuggingFace dataset identifier.
        split: Dataset split to load (defaults to ``"train"`` — the only
            split in this dataset).

    Returns:
        HuggingFace :class:`~datasets.Dataset` with a ``prompt`` field.

    Raises:
        ValueError: If the dataset cannot be coerced to a single
            :class:`~datasets.Dataset`.
    """
    effective_split = split or "train"
    logger.info("Loading %s (split=%s) ...", dataset_name, effective_split)

    raw = load_dataset(dataset_name, split=effective_split)
    if isinstance(raw, DatasetDict):
        for key in ("train", "test", "validation"):
            if key in raw:
                raw = raw[key]
                break
        else:
            raw = raw[next(iter(raw.keys()))]

    if not isinstance(raw, Dataset):
        raise ValueError(f"Expected Dataset after split selection, got {type(raw)!r}")

    if max_samples is not None:
        raw = raw.select(range(min(max_samples, len(raw))))

    def _add_prompt(example: dict[str, Any]) -> dict[str, Any]:
        example["prompt"] = _build_prompt(example)
        return example

    raw = raw.map(_add_prompt)
    logger.info("Loaded %d samples from %s", len(raw), dataset_name)
    return raw


# ======================================================================
# Prompt formatting
# ======================================================================


def format_prompt_for_model(
    example: dict[str, Any],
    tokenizer: Any,
    split: str = "instruct",
    instruction_prefix: str = CODE_ALPACA_DEFAULTS["instruction_prefix"],
    response_prefix: str = CODE_ALPACA_DEFAULTS["response_prefix"],
    prefill: bool = True,
    direct_completion: bool = False,
) -> dict[str, Any]:
    """Format a CodeAlpaca prompt for the model (for ``dataset.map``).

    For **chat-template models** the instruction is placed in the ``user``
    turn; the ``assistant`` turn is optionally prefilled with a code-fence
    opener so the model begins generating code immediately.

    For **base / completion models** (``direct_completion=True`` or no
    ``chat_template``) the classic Alpaca plain-text format is used:

        Below is an instruction that describes a coding task …

        ### Instruction:
        …

        ### Input:      ← omitted when empty
        …

        ### Response:

    Args:
        example: Dataset example dict; must contain a ``prompt`` field.
        tokenizer: Tokenizer instance; ``chat_template`` attribute is
            inspected to choose the formatting path.
        split: Required by the shared formatter signature; unused here.
        instruction_prefix: System-level directive shown before the task.
        response_prefix: Response header / assistant prefill token.
        prefill: When ``True`` the assistant turn is prefilled up to the
            response prefix so the model starts generating directly.
        direct_completion: Skip the chat template even if available.

    Returns:
        Example dict with ``formatted_prompt`` field added.
    """
    _ = split

    task_prompt: str = (example.get("prompt") or "").strip()

    # ── Base / completion models ──────────────────────────────────────
    if tokenizer.chat_template is None or direct_completion:
        example["formatted_prompt"] = (
            f"{instruction_prefix}\n\n{task_prompt}\n\n{response_prefix}\n"
        )
        return example

    # ── Chat-template models ──────────────────────────────────────────
    user_message = f"{instruction_prefix}\n\n{task_prompt}"

    if prefill:
        assistant_content = f"{response_prefix}\n{_MAGIC_SPLITTER_}"
        formatted_prompt = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_content},
            ],
            tokenize=False,
        ).split(_MAGIC_SPLITTER_)[0]
    else:
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_message}],
            tokenize=False,
            add_generation_prompt=True,
        )

    example["formatted_prompt"] = formatted_prompt
    return example


# ======================================================================
# Quick smoke test (no model required)
# ======================================================================


def main() -> None:
    """Load a few CodeAlpaca samples and print formatted prompts."""
    import sys

    logging.basicConfig(level=logging.INFO)

    class _FakeTokenizer:
        chat_template = None

    ds = load_code_alpaca(max_samples=3)

    for i in range(len(ds)):
        ex = format_prompt_for_model(dict(ds[i]), _FakeTokenizer())
        logger.info(
            "[%d]\n%s\n%s\n",
            i + 1,
            "-" * 72,
            ex["formatted_prompt"][:600],
        )

    sys.exit(0)


if __name__ == "__main__":
    main()
