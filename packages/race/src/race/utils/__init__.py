"""General-purpose RACE utilities.

Exports are lazy. Importing ``race.utils`` does not load PyTorch until an
exported name is accessed.
"""

from __future__ import annotations

__all__ = ["resolve_device"]

_LAZY_IMPORT_MAP: dict[str, tuple[str, str]] = {
    "resolve_device": (".device", "resolve_device"),
}


def __getattr__(name: str):
    from race._lazy import load_lazy_attr

    return load_lazy_attr(globals(), __name__, _LAZY_IMPORT_MAP, name)


def __dir__():
    return sorted(set(globals()) | set(__all__))
