"""LLM intervention and ablation utilities."""

from __future__ import annotations

__all__ = [
    "AblationPlan",
    "LLMNeuronModifier",
    "OperationMode",
    "create_ablation_plan",
    "create_random_ablation_plan",
    "load_top_k_neurons_from_h5",
]

_LAZY_IMPORT_MAP: dict[str, tuple[str, str]] = {
    "AblationPlan": ("race_eval.ablation", "AblationPlan"),
    "OperationMode": ("race_eval.ablation", "OperationMode"),
    "create_ablation_plan": ("race_eval.ablation", "create_ablation_plan"),
    "create_random_ablation_plan": (
        "race_eval.ablation",
        "create_random_ablation_plan",
    ),
    "load_top_k_neurons_from_h5": (
        "race_eval.ablation",
        "load_top_k_neurons_from_h5",
    ),
    "LLMNeuronModifier": (".ablation", "LLMNeuronModifier"),
}


def __getattr__(name: str):
    from race._lazy import load_lazy_attr

    return load_lazy_attr(globals(), __name__, _LAZY_IMPORT_MAP, name)


def __dir__():
    return sorted(set(globals()) | set(__all__))
