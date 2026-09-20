"""I/O utilities for RACE.

Provides HDF5 schema definitions and data serialization helpers.

Exports are lazy. Importing ``race.io`` does not load h5py until an exported
name is accessed.
"""

from __future__ import annotations

__all__ = [
    "get_axis_metadata_lookup",
    "iter_axis_reports",
    "read_instance_metadata",
    "read_json_dataset",
]

_LAZY_IMPORT_MAP: dict[str, tuple[str, str]] = {
    "get_axis_metadata_lookup": (".h5_schema", "get_axis_metadata_lookup"),
    "iter_axis_reports": (".h5_schema", "iter_axis_reports"),
    "read_instance_metadata": (".h5_schema", "read_instance_metadata"),
    "read_json_dataset": (".h5_schema", "read_json_dataset"),
}


def __getattr__(name: str):
    from race._lazy import load_lazy_attr

    return load_lazy_attr(globals(), __name__, _LAZY_IMPORT_MAP, name)


def __dir__():
    return sorted(set(globals()) | set(__all__))
