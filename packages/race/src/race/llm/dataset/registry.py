"""Dataset registry for LLM online-RACE pipelines.

The unified online-RACE pipeline only needs a minimal contract from datasets:

- a loader that returns a HuggingFace ``datasets.Dataset``
- a formatter that maps each example to a ``formatted_prompt`` string (via ``Dataset.map``)
- default prompt parameters

This registry keeps the pipeline decoupled from dataset-specific details.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    defaults: Dict[str, Any]
    load_fn: Callable[..., Any]
    format_fn: Callable[..., Dict[str, Any]]
    teacher_forcing: bool = False
    loader_handles_sampling: bool = False


_REGISTRY: Dict[str, DatasetSpec] = {}


def register_dataset(spec: DatasetSpec) -> None:
    key = spec.dataset_id.lower()
    if key in _REGISTRY:
        raise KeyError(f"Dataset already registered: {spec.dataset_id!r}")
    _REGISTRY[key] = spec


def get_dataset_spec(dataset_id: str) -> DatasetSpec:
    key = (dataset_id or "").lower()
    if key not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY.keys())) or "<empty>"
        raise KeyError(f"Unknown dataset {dataset_id!r}. Known: {known}")
    return _REGISTRY[key]


def ensure_builtin_datasets_registered() -> None:
    """Register built-in datasets lazily to avoid import side-effects."""
    if _REGISTRY:
        return

    # isort: off
    from race.llm.dataset.math500 import (  # local import by design
        MATH500_DEFAULTS,
        format_prompt_for_model as math_format,
        load_math500,
    )
    from race.llm.dataset.mbpp_plus import (  # local import by design
        MBPP_PLUS_DEFAULTS,
        format_prompt_for_model as mbpp_plus_format,
        load_mbpp_plus,
    )
    from race.llm.dataset.gpqa_diamond import (  # local import by design
        GPQA_DIAMOND_DEFAULTS,
        format_prompt_for_model as gpqa_diamond_format,
        load_gpqa_diamond,
    )
    from race.llm.dataset.mmlu_redux import (  # local import by design
        MMLU_REDUX_DEFAULTS,
        format_prompt_for_model as mmlu_redux_format,
        load_mmlu_redux,
    )
    from race.llm.dataset.competition_math import (  # local import by design
        COMPETITION_MATH_DEFAULTS,
        format_prompt_for_model as competition_math_format,
        load_competition_math,
    )
    from race.llm.dataset.wikitext2 import (  # local import by design
        WIKITEXT2_DEFAULTS,
        format_prompt_for_model as wikitext2_format,
        load_wikitext2,
    )
    from race.llm.dataset.fineweb import (  # local import by design
        FINEWEB_DEFAULTS,
        format_prompt_for_model as fineweb_format,
        load_fineweb,
    )
    from race.llm.dataset.apps import (  # local import by design
        APPS_DEFAULTS,
        format_prompt_for_model as apps_format,
        load_apps,
    )
    from race.llm.dataset.code_alpaca import (  # local import by design
        CODE_ALPACA_DEFAULTS,
        format_prompt_for_model as code_alpaca_format,
        load_code_alpaca,
    )
    from race.llm.dataset.py_comprehension_statements import (  # local import by design
        PY_COMPREHENSION_STATEMENTS_DEFAULTS,
        format_prompt_for_model as py_comprehension_statements_format,
        load_py_comprehension_statements,
    )

    # isort: on

    register_dataset(
        DatasetSpec(
            dataset_id="math500",
            defaults=MATH500_DEFAULTS,
            load_fn=load_math500,
            format_fn=math_format,
        )
    )
    register_dataset(
        DatasetSpec(
            dataset_id="mbpp_plus",
            defaults=MBPP_PLUS_DEFAULTS,
            load_fn=load_mbpp_plus,
            format_fn=mbpp_plus_format,
        )
    )
    register_dataset(
        DatasetSpec(
            dataset_id="gpqa_diamond",
            defaults=GPQA_DIAMOND_DEFAULTS,
            load_fn=load_gpqa_diamond,
            format_fn=gpqa_diamond_format,
        )
    )
    register_dataset(
        DatasetSpec(
            dataset_id="mmlu_redux",
            defaults=MMLU_REDUX_DEFAULTS,
            load_fn=load_mmlu_redux,
            format_fn=mmlu_redux_format,
        )
    )
    register_dataset(
        DatasetSpec(
            dataset_id="competition_math",
            defaults=COMPETITION_MATH_DEFAULTS,
            load_fn=load_competition_math,
            format_fn=competition_math_format,
        )
    )
    register_dataset(
        DatasetSpec(
            dataset_id="wikitext2",
            defaults=WIKITEXT2_DEFAULTS,
            load_fn=load_wikitext2,
            format_fn=wikitext2_format,
            teacher_forcing=True,
        )
    )
    register_dataset(
        DatasetSpec(
            dataset_id="fineweb",
            defaults=FINEWEB_DEFAULTS,
            load_fn=load_fineweb,
            format_fn=fineweb_format,
            teacher_forcing=True,
            loader_handles_sampling=True,
        )
    )
    register_dataset(
        DatasetSpec(
            dataset_id="apps",
            defaults=APPS_DEFAULTS,
            load_fn=load_apps,
            format_fn=apps_format,
        )
    )
    register_dataset(
        DatasetSpec(
            dataset_id="code_alpaca",
            defaults=CODE_ALPACA_DEFAULTS,
            load_fn=load_code_alpaca,
            format_fn=code_alpaca_format,
        )
    )
    register_dataset(
        DatasetSpec(
            dataset_id="py_comprehension_statements",
            defaults=PY_COMPREHENSION_STATEMENTS_DEFAULTS,
            load_fn=load_py_comprehension_statements,
            format_fn=py_comprehension_statements_format,
            teacher_forcing=True,
        )
    )
