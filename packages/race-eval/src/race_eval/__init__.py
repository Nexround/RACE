"""Evaluation and ablation utilities for RACE/RACE experiments."""

from __future__ import annotations

__all__ = ["ablation", "llm"]


def __getattr__(name: str):
    if name in __all__:
        import importlib

        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
