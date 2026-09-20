"""LLM evaluation, ablation, serving, and perturbed-model utilities."""

from __future__ import annotations

__all__ = ["evaluation", "interventions", "pipelines", "serving", "LLMNeuronModifier"]

_LAZY_IMPORT_MAP: dict[str, tuple[str, str]] = {
    "LLMNeuronModifier": (".interventions.ablation", "LLMNeuronModifier"),
}


def __getattr__(name: str):
    if name in ("evaluation", "interventions", "pipelines", "serving"):
        import importlib

        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod

    from race._lazy import load_lazy_attr

    return load_lazy_attr(globals(), __name__, _LAZY_IMPORT_MAP, name)


def __dir__():
    return sorted(set(globals()) | set(__all__))
