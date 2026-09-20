"""APPS (codeparrot/apps) dataset loader for LLM-RACE analysis.

This module provides utilities to load and process the APPS dataset
(Automated Programming Progress Standard) for RACE attribution analysis
on competitive-programming / interview code-generation tasks.

Dataset: https://huggingface.co/datasets/codeparrot/apps
Columns:
    problem_id   (int)    – unique identifier
    question     (str)    – problem description (may contain sample I/O)
    solutions    (str)    – JSON-encoded list of accepted Python solutions
    input_output (str)    – JSON-encoded dict with "inputs" / "outputs" lists
    difficulty   (str)    – "introductory" | "interview" | "competition"
    url          (str)    – source URL
    starter_code (str)    – optional function/class skeleton (may be empty)

References:
    Hendrycks et al. (2021) "Measuring Coding Challenge Competence With APPS"
    https://arxiv.org/abs/2105.09938
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from datasets import Dataset, DatasetDict, load_dataset

logger = logging.getLogger(__name__)


# ======================================================================
# APPS default prompt parameters
# ======================================================================

APPS_DEFAULTS: dict[str, Any] = {
    # Required by the dataset adapter contract and embedded in the template.
    "instruction_prefix": ("Write a Python solution for the following problem."),
    "response_prefix": "```python",
}

# Difficulty levels available in the dataset.
APPS_DIFFICULTIES = ("introductory", "interview", "competition")

# Magic splitter for chat-template prefill split.
_MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"

# Maximum number of sample I/O pairs to include in the prompt.
_MAX_SAMPLE_IO = 3


# ======================================================================
# Helper utilities
# ======================================================================


def _parse_json_field(raw: Any) -> Any:
    """Return parsed JSON if *raw* is a non-empty string, else *raw*."""
    if isinstance(raw, str) and raw.strip():
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return raw


def _build_sample_io_block(input_output: Any) -> str:
    """Return a formatted sample-I/O block (up to *_MAX_SAMPLE_IO* pairs).

    Args:
        input_output: Parsed ``input_output`` field (dict with ``"inputs"``
            and ``"outputs"`` lists), or an unparseable raw value.

    Returns:
        A multi-line string ready to be embedded in the problem description,
        or an empty string when no valid test cases are available.
    """
    if not isinstance(input_output, dict):
        return ""

    inputs: list[Any] = input_output.get("inputs") or []
    outputs: list[Any] = input_output.get("outputs") or []

    pairs = list(zip(inputs, outputs))[:_MAX_SAMPLE_IO]
    if not pairs:
        return ""

    lines: list[str] = ["", "Sample input/output:"]
    for idx, (inp, out) in enumerate(pairs, 1):
        lines.append(f"  Example {idx}:")
        lines.append(f"    Input:  {inp}")
        lines.append(f"    Output: {out}")
    return "\n".join(lines)


def _build_prompt(example: dict[str, Any]) -> str:
    """Construct the canonical problem prompt from a raw dataset example.

    The prompt consists of:
    1. The problem statement (``question`` field).
    2. An optional starter-code block (when ``starter_code`` is non-empty).
    3. An optional sample I/O section derived from ``input_output``.
    """
    question: str = (example.get("question") or "").strip()
    starter_code: str = (example.get("starter_code") or "").strip()
    input_output = _parse_json_field(example.get("input_output"))

    parts: list[str] = [question]

    if starter_code:
        parts.append(
            f"\nUse the following starter code:\n```python\n{starter_code}\n```"
        )

    io_block = _build_sample_io_block(input_output)
    if io_block:
        parts.append(io_block)

    return "\n".join(parts)


# ======================================================================
# Dataset loading
# ======================================================================


def load_apps(
    max_samples: Optional[int] = None,
    dataset_name: str = "codeparrot/apps",
    split: Optional[str] = None,
    difficulty: Optional[str] = None,
) -> Dataset:
    """Load the APPS dataset with a ``prompt`` field added.

    Args:
        max_samples: Maximum number of samples to return.  Applied *after*
            optional difficulty filtering so that each difficulty level is
            sampled proportionally (unless ``difficulty`` is set).
        dataset_name: HuggingFace dataset identifier.
        split: Dataset split to load (defaults to ``"test"``).
        difficulty: Optional difficulty filter – one of ``"introductory"``,
            ``"interview"``, or ``"competition"``.  When ``None`` all
            difficulties are included.

    Returns:
        HuggingFace :class:`~datasets.Dataset` with a ``prompt`` field.

    Raises:
        ValueError: If an unsupported *difficulty* value is provided, or if
            the dataset cannot be coerced to a single :class:`~datasets.Dataset`.
    """
    if difficulty is not None and difficulty not in APPS_DIFFICULTIES:
        raise ValueError(
            f"difficulty must be one of {APPS_DIFFICULTIES!r}, got {difficulty!r}"
        )

    effective_split = split or "test"
    logger.info(
        "Loading %s (split=%s, difficulty=%s) ...",
        dataset_name,
        effective_split,
        difficulty or "all",
    )

    raw = load_dataset(dataset_name, split=effective_split)
    if isinstance(raw, DatasetDict):
        for key in ("test", "train", "validation"):
            if key in raw:
                raw = raw[key]
                break
        else:
            raw = raw[next(iter(raw.keys()))]

    if not isinstance(raw, Dataset):
        raise ValueError(f"Expected Dataset after split selection, got {type(raw)!r}")

    # Optional difficulty filter.
    if difficulty is not None:
        raw = raw.filter(lambda ex: ex.get("difficulty") == difficulty)

    if max_samples is not None:
        raw = raw.select(range(min(max_samples, len(raw))))

    # Add unified `prompt` field.
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
    instruction_prefix: str = APPS_DEFAULTS["instruction_prefix"],
    response_prefix: str = APPS_DEFAULTS["response_prefix"],
    prefill: bool = True,
    direct_completion: bool = False,
) -> dict[str, Any]:
    """Format a prompt for the model (intended for use with ``dataset.map``).

    Supports two modes:

    * **Chat-template mode** (default for instruction-tuned models): wraps
      the problem in a ``user`` turn and optionally prefills the
      ``assistant`` turn with a Python code-fence opener.
    * **Direct-completion mode**: plain string suitable for base / fill-in-
      the-middle models.

    Args:
        example: Dataset example dict; must contain a ``prompt`` field
            (produced by :func:`load_apps`).
        tokenizer: Tokenizer instance; ``chat_template`` attribute is
            inspected to choose the formatting path.
        split: Required by the shared formatter signature; unused here.
        instruction_prefix: Instruction directive prepended to the problem.
        response_prefix: Assistant response prefix (e.g. code-fence opener).
        prefill: When ``True`` the assistant turn is prefilled up to the
            code-fence opener so that the model starts generating Python code
            directly.
        direct_completion: When ``True`` skip the chat template even if the
            tokenizer has one (useful for evaluating base models).

    Returns:
        Example dict with a ``formatted_prompt`` field added.
    """
    _ = split

    task_prompt: str = (example.get("prompt") or "").strip()
    user_message = f"{instruction_prefix}\n\n{task_prompt}"

    # ── Base / completion models ──────────────────────────────────────
    if tokenizer.chat_template is None or direct_completion:
        example["formatted_prompt"] = f"{user_message}\n\n{response_prefix}\n"
        return example

    # ── Chat-template models ──────────────────────────────────────────
    if prefill:
        # Prefill the assistant turn up to (but not including) the magic
        # splitter so the model continues right after the code-fence opener.
        assistant_content = f"{response_prefix}\n{_MAGIC_SPLITTER_}\n```\n"
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
    """Load a few APPS samples and print formatted prompts."""
    import sys

    logging.basicConfig(level=logging.INFO)

    class _FakeTokenizer:
        chat_template = None

    ds = load_apps(max_samples=3)

    for i in range(len(ds)):
        ex = format_prompt_for_model(dict(ds[i]), _FakeTokenizer())
        logger.info(
            "[%d] problem_id=%s  difficulty=%s\n%s\n%s\n",
            i + 1,
            ex.get("problem_id", "N/A"),
            ex.get("difficulty", "N/A"),
            "-" * 72,
            ex["formatted_prompt"][:600],
        )

    sys.exit(0)


if __name__ == "__main__":
    main()
