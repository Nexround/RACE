"""MMLU-Redux dataset loader for LLM evaluation.

Dataset: https://modelscope.cn/datasets/AI-ModelScope/mmlu-redux-2.0
Subsets: 57 academic subject configs (e.g. "abstract_algebra", "anatomy", ...)

Each row has:
    question  : str   — the question text
    choices   : list[str] — four answer options [A, B, C, D]
    answer    : int   — index of the correct choice (0-3)
    subject   : str   — subject name (present in some configs / merged splits)

The default source matches EvalScope's MMLU-Redux benchmark exactly:
``AI-ModelScope/mmlu-redux-2.0`` with 57 subjects and 100 test examples per
subject.  Individual subject datasets are merged in EvalScope subject order.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)

logger = logging.getLogger(__name__)


MMLU_REDUX_DEFAULTS: dict[str, Any] = {
    "instruction_prefix": "",
    "response_prefix": "",
}

_CHOICE_LABELS = ["A", "B", "C", "D"]

_MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"

# The Hugging Face source contains only the corrected subset of subjects.
# Callers may select it explicitly instead of the default ModelScope source.
_HF_DATASET = "edinburgh-dawg/mmlu-redux"

# This is the 57-subject source used by EvalScope's ``mmlu_redux`` benchmark.
_MODELSCOPE_DATASET = "AI-ModelScope/mmlu-redux-2.0"

# All 57 subject subsets in MMLU-Redux
_ALL_SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


def _modelscope_cache_root() -> Path:
    """Return ModelScope's dataset cache root without requiring ModelScope."""
    try:
        from modelscope.utils.config_ds import MS_DATASETS_CACHE

        return Path(MS_DATASETS_CACHE).expanduser()
    except ImportError:
        return Path("~/.cache/modelscope/hub/datasets").expanduser()


@lru_cache(maxsize=4)
def _cached_modelscope_subjects(dataset_name: str) -> dict[str, tuple[str, Path]]:
    """Index locally cached ModelScope subjects by their config name.

    EvalScope stores processed datasets below ``<cache>/datasets`` while the
    ModelScope SDK also keeps Arrow files in its own cache layout.  Supporting
    both layouts avoids a network metadata request when EvalScope has already
    downloaded the benchmark.
    """
    cache_root = _modelscope_cache_root()
    index: dict[str, tuple[str, Path]] = {}

    processed_prefix = dataset_name.replace("/", "_") + "-"
    processed_root = cache_root / "datasets"
    if processed_root.is_dir():
        for path in processed_root.glob(f"{processed_prefix}*"):
            info_path = path / "dataset_info.json"
            if not info_path.is_file():
                continue
            try:
                config_name = str(json.loads(info_path.read_text())["config_name"])
            except (KeyError, OSError, json.JSONDecodeError):
                continue
            index[config_name] = ("disk", path)

    raw_root = cache_root / dataset_name.replace("/", "___")
    if raw_root.is_dir():
        for info_path in raw_root.glob("*/**/dataset_info.json"):
            try:
                config_name = str(json.loads(info_path.read_text())["config_name"])
            except (KeyError, OSError, json.JSONDecodeError):
                continue
            arrow_paths = sorted(info_path.parent.glob("*.arrow"))
            if arrow_paths and config_name not in index:
                index[config_name] = ("arrow", arrow_paths[0])

    return index


def _load_modelscope_subject(
    dataset_name: str,
    subject: str,
    split: str,
) -> Dataset:
    cached = _cached_modelscope_subjects(dataset_name).get(subject)
    if cached is not None:
        cache_type, path = cached
        if cache_type == "disk":
            dataset = load_from_disk(str(path))
            if isinstance(dataset, Dataset):
                return dataset
        else:
            return Dataset.from_file(str(path))

    try:
        from modelscope.msdatasets import MsDataset
    except ImportError as exc:
        raise RuntimeError(
            f"Loading {dataset_name!r} requires the optional 'modelscope' package"
        ) from exc

    dataset = MsDataset.load(
        dataset_name=dataset_name,
        subset_name=subject,
        split=split,
    )
    if not isinstance(dataset, Dataset):
        dataset = dataset.to_hf_dataset()
    if not isinstance(dataset, Dataset):
        raise TypeError(
            f"Unexpected ModelScope dataset type for {subject!r}: {type(dataset)}"
        )
    return dataset


def _load_subject(dataset_name: str, subject: str, split: str) -> Dataset:
    if dataset_name == _MODELSCOPE_DATASET:
        return _load_modelscope_subject(dataset_name, subject, split)

    raw = load_dataset(dataset_name, subject, split=split)
    if isinstance(raw, DatasetDict):
        for key in (split, "test", "validation", "train"):
            if key in raw:
                raw = raw[key]
                break
        else:
            raw = raw[next(iter(raw.keys()))]
    if not isinstance(raw, Dataset):
        raise TypeError(f"Unexpected type for subject {subject!r}: {type(raw)}")
    return raw


def _build_question_text(example: dict[str, Any]) -> str:
    """Format a multiple-choice question with lettered options."""
    question = (example.get("question") or "").strip()
    choices: list[str] = example.get("choices") or []
    options_text = "\n".join(
        f"{label}. {choice}" for label, choice in zip(_CHOICE_LABELS, choices)
    )
    return f"{question}\n\n{options_text}"


def load_mmlu_redux(
    max_samples: Optional[int] = None,
    dataset_name: str = _MODELSCOPE_DATASET,
    subjects: Optional[list[str]] = None,
    split: str = "test",
) -> Dataset:
    """Load MMLU-Redux dataset with a ``prompt`` field.

    Args:
        max_samples: Maximum total number of samples to return.
        dataset_name: ModelScope dataset identifier by default. The compatible
            Hugging Face corrected-subset source can be supplied explicitly.
        subjects: List of subject subset names to load.  Defaults to all 57
                  subjects.  Pass a single-element list for a quick test.
        split: Dataset split (typically ``"test"``).

    Returns:
        HuggingFace :class:`Dataset` with a ``prompt`` field containing the
        formatted multiple-choice question text.
    """
    subjects = subjects or _ALL_SUBJECTS
    logger.info(
        "Loading %s (split=%s, %d subjects) ...", dataset_name, split, len(subjects)
    )

    parts: list[Dataset] = []
    failures: list[tuple[str, Exception]] = []
    for subject in subjects:
        try:
            raw = _load_subject(dataset_name, subject, split)
        except Exception as exc:
            failures.append((subject, exc))
            continue

        # Add subject column if missing (for traceability)
        if "subject" not in raw.column_names:
            raw = raw.add_column("subject", [subject] * len(raw))

        parts.append(raw)
        if max_samples is not None and sum(len(p) for p in parts) >= max_samples:
            break

    if failures and max_samples is None:
        details = "; ".join(
            f"{subject}: {type(exc).__name__}: {exc}" for subject, exc in failures[:5]
        )
        if len(failures) > 5:
            details += f"; ... and {len(failures) - 5} more"
        raise RuntimeError(
            f"Failed to load {len(failures)}/{len(subjects)} MMLU-Redux "
            f"subjects from {dataset_name!r}: {details}"
        )

    if not parts:
        raise RuntimeError(
            f"No MMLU-Redux subjects could be loaded from {dataset_name!r}."
        )

    dataset = concatenate_datasets(parts)

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    def _add_prompt(example: dict[str, Any]) -> dict[str, Any]:
        example["prompt"] = _build_question_text(example)
        return example

    dataset = dataset.map(_add_prompt)
    logger.info("Loaded %d samples from %s", len(dataset), dataset_name)
    return dataset


def format_prompt_for_model(
    example: dict[str, Any],
    tokenizer: Any,
    split: str = "instruct",
    instruction_prefix: str = MMLU_REDUX_DEFAULTS["instruction_prefix"],
    response_prefix: str = MMLU_REDUX_DEFAULTS["response_prefix"],
    prefill: bool = True,
    direct_completion: bool = False,
) -> dict[str, Any]:
    """Format prompt for model using chat template (for ``dataset.map``).

    Constructs a multiple-choice question prompt and wraps it with the model's
    chat template.  The system instruction asks the model to choose the best
    answer and explain its reasoning.
    """
    _ = split
    _ = instruction_prefix

    question = (example.get("question") or "").strip()
    choices: list[str] = example.get("choices") or []
    letters = ",".join(_CHOICE_LABELS[: len(choices)])
    choices_text = "\n".join(
        f"{label}) {choice}" for label, choice in zip(_CHOICE_LABELS, choices)
    )
    task_prompt = (
        "Answer the following multiple choice question. The last line of your "
        "response should be of the following format: 'ANSWER: [LETTER]' "
        f"(without quotes) where [LETTER] is one of {letters}. Think step by "
        f"step before answering.\n\n{question}\n\n{choices_text}"
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
