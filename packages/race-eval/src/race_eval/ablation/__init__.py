"""Ablation plan construction and neuron intervention primitives."""

from __future__ import annotations

__all__ = [
    "AblationPlan",
    "BaseNeuronModifier",
    "OperationMode",
    "build_ablation_plan",
    "create_ablation_plan",
    "create_reference_filtered_ablation_plan",
    "create_random_ablation_plan",
    "list_llm_h5_layer_indices",
    "load_top_k_neurons_from_h5",
    "select_bottom_indices",
    "select_top_indices",
    "select_top_negative_indices",
    "select_top_positive_indices",
]

_LAZY_IMPORT_MAP: dict[str, tuple[str, str]] = {
    "AblationPlan": (".plan", "AblationPlan"),
    "OperationMode": (".plan", "OperationMode"),
    "build_ablation_plan": (".plan", "build_ablation_plan"),
    "create_ablation_plan": (".plan", "create_ablation_plan"),
    "create_reference_filtered_ablation_plan": (
        ".plan",
        "create_reference_filtered_ablation_plan",
    ),
    "create_random_ablation_plan": (".plan", "create_random_ablation_plan"),
    "list_llm_h5_layer_indices": (".plan", "list_llm_h5_layer_indices"),
    "load_top_k_neurons_from_h5": (".plan", "load_top_k_neurons_from_h5"),
    "select_bottom_indices": (".plan", "select_bottom_indices"),
    "select_top_indices": (".plan", "select_top_indices"),
    "select_top_negative_indices": (".plan", "select_top_negative_indices"),
    "select_top_positive_indices": (".plan", "select_top_positive_indices"),
    "BaseNeuronModifier": (".intervention", "BaseNeuronModifier"),
}


def __getattr__(name: str):
    from race._lazy import load_lazy_attr

    return load_lazy_attr(globals(), __name__, _LAZY_IMPORT_MAP, name)


def __dir__():
    return sorted(set(globals()) | set(__all__))
