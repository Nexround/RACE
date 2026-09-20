"""Utilities for unified activation/RACE HDF5 schema.

This module centralizes helper functions for constructing the shared
HDF5 layout used by the RACE pipelines. The helpers focus on:

- Meta information (`/meta`) including schema versioning and run metadata.
- Axis management (`/meta/axes/value_xxx`) with membership lists.
- Common datasets such as JSON payload blobs stored as UTF-8 strings.
- Consistent instance naming (`/instances/instance_XXXXXX`).

Writers are expected to call the helpers before appending instances or reports.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple

import h5py
import numpy as np

SCHEMA_VERSION = "1.0.0"
STRING_DTYPE = h5py.string_dtype("utf-8")


def _now_iso() -> str:
    return datetime.now().isoformat()


def _sanitize_identifier(raw: str) -> str:
    """Return filesystem-safe identifier derived from raw string."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", raw).strip("_").lower()
    if slug:
        return slug[:48]
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return digest[:16]


def _value_token(value: Any) -> str:
    if isinstance(value, int):
        return f"{value:06d}"
    if isinstance(value, float):
        return f"{value:.4f}".replace(".", "_")
    if value is None:
        return "none"
    return _sanitize_identifier(str(value))


def _ensure_group(parent: h5py.Group, name: str) -> h5py.Group:
    if name in parent:
        return parent[name]
    return parent.create_group(name)


def _ensure_dataset(
    parent: h5py.Group,
    name: str,
    shape: Tuple[int, ...],
    *,
    dtype: Any,
    maxshape: Optional[Tuple[Optional[int], ...]] = None,
    chunks: Optional[Tuple[int, ...]] = None,
) -> h5py.Dataset:
    if name in parent:
        return parent[name]
    kwargs = {}
    if maxshape is not None:
        kwargs["maxshape"] = maxshape
    if chunks is not None:
        kwargs["chunks"] = chunks
    return parent.create_dataset(name, shape=shape, dtype=dtype, **kwargs)


def write_json_blob(
    group: h5py.Group, name: str, payload: Optional[Dict[str, Any]]
) -> None:
    """Store JSON payload under the given group."""
    data = json.dumps(payload or {}, ensure_ascii=False)
    encoded = np.asarray([data], dtype=STRING_DTYPE)
    if name in group:
        ds = group[name]
        ds.resize((1,))
        ds[0] = encoded[0]
    else:
        group.create_dataset(
            name,
            data=encoded,
            dtype=STRING_DTYPE,
            maxshape=(None,),
        )


def format_instance_key(instance_idx: int) -> str:
    return f"instance_{instance_idx:06d}"


def ensure_meta(
    h5f: h5py.File,
    *,
    target_model_family: str,
    model_name: str,
    device: str,
    run_id: Optional[str] = None,
    model_revision: Optional[str] = None,
    created_at: Optional[str] = None,
    extra_payload: Optional[Dict[str, Any]] = None,
) -> Tuple[h5py.Group, h5py.Group]:
    """Ensure `/meta` structure exists and return `(meta_group, axes_group)`."""
    meta_group = _ensure_group(h5f, "meta")

    schema_group = _ensure_group(meta_group, "schema")
    schema_group.attrs["version"] = SCHEMA_VERSION
    schema_group.attrs["target_model_family"] = target_model_family

    run_group = _ensure_group(meta_group, "run")
    if created_at is None:
        created_at = _now_iso()
    if run_id is None:
        run_id = created_at.replace(":", "").replace("-", "")

    run_group.attrs["model_name"] = model_name
    run_group.attrs["model_revision"] = model_revision or ""
    run_group.attrs["run_id"] = run_id
    run_group.attrs["created_at"] = created_at
    run_group.attrs["device"] = device
    if "total_instances" not in run_group.attrs:
        run_group.attrs["total_instances"] = 0

    if extra_payload is not None or "json_payload" not in run_group:
        write_json_blob(run_group, "json_payload", extra_payload or {})

    axes_group = _ensure_group(meta_group, "axes")
    return meta_group, axes_group


def update_total_instances(h5f: h5py.File, total_instances: int) -> None:
    meta_group = h5f.get("meta")
    if meta_group is None:
        return
    run_group = meta_group.get("run")
    if run_group is None:
        return
    run_group.attrs["total_instances"] = int(total_instances)


def ensure_axis_value(
    axes_group: h5py.Group,
    *,
    axis_type: str,
    value_key: Any,
    display_name: Optional[str] = None,
    raw_value: Optional[Any] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> Tuple[h5py.Group, str]:
    """Ensure the specified axis value group exists directly under axes/.

    The ``axes`` group contains ``value_XXXXXX`` groups directly.
    ``value_key`` determines the numeric suffix.

    Args:
        axes_group: The /meta/axes group
        axis_type: Type of axis (e.g., "predicted_label", "domain")
        value_key: Value key for generating value_XXXXXX (must be integer or convertible to 6-digit numeric)
        display_name: Human-readable name
        raw_value: Raw value dict (stored as JSON)
        payload: Additional metadata (stored as JSON dataset)

    Returns:
        (value_group, path) where path is relative to /meta (e.g., "axes/value_000207")
    """
    # Axis value names require a six-digit numeric suffix.
    value_token = _value_token(value_key)
    if not value_token.isdigit() or len(value_token) != 6:
        raise ValueError(
            f"value_key must generate a 6-digit numeric token, got '{value_token}' from {value_key}. "
            f"Use integer value_key (e.g., label ID or domain ID from domain_mapping)."
        )

    value_group = _ensure_group(axes_group, f"value_{value_token}")

    existing_type = value_group.attrs.get("axis_type")
    if existing_type and existing_type != axis_type:
        raise ValueError(
            f"Value '{value_token}' already exists with type '{existing_type}', expected '{axis_type}'"
        )
    value_group.attrs["axis_type"] = axis_type

    if display_name is not None:
        value_group.attrs["display_name"] = display_name
    else:
        value_group.attrs.setdefault("display_name", str(value_key))

    if raw_value is not None:
        value_group.attrs["raw_value"] = json.dumps(raw_value, ensure_ascii=False)
    else:
        value_group.attrs.setdefault(
            "raw_value", json.dumps(value_key, ensure_ascii=False)
        )

    if payload is not None:
        write_json_blob(value_group, "json_payload", payload)

    members_ds = value_group.get("members")
    if members_ds is None:
        members_ds = value_group.create_dataset(
            "members",
            shape=(0,),
            maxshape=(None,),
            dtype="i8",
            chunks=(1024,),
        )

    path = f"axes/value_{value_token}"
    return value_group, path


def append_axis_members(
    value_group: h5py.Group,
    members: Iterable[int],
) -> None:
    """Append instance indices to axis value membership dataset."""
    to_add = [int(m) for m in members]
    if not to_add:
        return
    members_ds = value_group.get("members")
    if members_ds is None:
        members_ds = value_group.create_dataset(
            "members",
            shape=(0,),
            maxshape=(None,),
            dtype="i8",
            chunks=(1024,),
        )
    current = members_ds.shape[0]
    new_size = current + len(to_add)
    members_ds.resize((new_size,))
    members_ds[current:new_size] = np.asarray(to_add, dtype="i8")
    value_group.attrs["member_count"] = new_size


def ensure_instances_group(h5f: h5py.File) -> h5py.Group:
    return _ensure_group(h5f, "instances")


def ensure_race_meta(
    h5f: h5py.File,
    *,
    target_model_family: str,
    model_name: str,
    device: str,
    race_config: Dict[str, Any],
    run_id: Optional[str] = None,
    created_at: Optional[str] = None,
    model_revision: Optional[str] = None,
    extra_run_payload: Optional[Dict[str, Any]] = None,
) -> Tuple[h5py.Group, h5py.Group]:
    """Ensure meta structure for RACE results and store config."""
    meta_group, axes_group = ensure_meta(
        h5f,
        target_model_family=target_model_family,
        model_name=model_name,
        device=device,
        run_id=run_id,
        model_revision=model_revision,
        created_at=created_at,
        extra_payload=extra_run_payload,
    )

    race_group = _ensure_group(meta_group, "race_config")
    for key, value in race_config.items():
        race_group.attrs[key] = value

    return meta_group, axes_group


def ensure_reports_group(h5f: h5py.File) -> h5py.Group:
    return _ensure_group(h5f, "reports")


def ensure_metrics_group(parent: h5py.Group) -> h5py.Group:
    return _ensure_group(parent, "metrics")


def record_summary_metric(metrics_group: h5py.Group, name: str, value: float) -> None:
    metrics_group.attrs[name] = float(value)


def iter_instances(h5f: h5py.File) -> Iterator[Tuple[str, h5py.Group]]:
    instances_group = h5f.get("instances")
    if instances_group is None:
        return
    keys = sorted(instances_group.keys())
    for key in keys:
        yield key, instances_group[key]


def read_json_dataset(group: h5py.Group, name: str) -> Dict[str, Any]:
    dataset = group.get(name)
    if dataset is None or dataset.size == 0:
        return {}
    raw = dataset[0]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def read_instance_metadata(metadata_group: h5py.Group) -> Dict[str, Any]:
    payload = read_json_dataset(metadata_group, "json_payload")
    axis_refs_raw = metadata_group.attrs.get("source_axis_value_refs", "[]")
    if isinstance(axis_refs_raw, bytes):
        axis_refs_raw = axis_refs_raw.decode("utf-8")
    try:
        axis_refs = json.loads(axis_refs_raw) if axis_refs_raw else []
    except json.JSONDecodeError:
        axis_refs = []

    info = {
        "instance_idx": int(metadata_group.attrs.get("instance_idx", -1)),
        "source_id": metadata_group.attrs.get("source_id", ""),
        "task_family": metadata_group.attrs.get("task_family", ""),
        "status": metadata_group.attrs.get("status", "success"),
        "error_message": metadata_group.attrs.get("error_message", ""),
        "axis_refs": axis_refs,
        "payload": payload,
    }
    return info


def _decode_raw_json_attr(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def get_axis_metadata_lookup(h5f: h5py.File) -> Dict[str, Dict[str, Any]]:
    """Return metadata for all axis values keyed by value token.

    The ``axes`` group contains ``value_XXXXXX`` groups directly.

    Returns:
        Dict mapping value_XXXXXX to metadata dict with:
        - axis_type, display_name, raw_value, member_count, members_path, payload
    """
    axes_root = h5f.get("meta/axes")
    if axes_root is None:
        return {}

    lookup: Dict[str, Dict[str, Any]] = {}
    for value_name, value_group in axes_root.items():
        if not value_name.startswith("value_"):
            continue

        entry: Dict[str, Any] = {
            "axis_type": value_group.attrs.get("axis_type", ""),
            "display_name": value_group.attrs.get("display_name", str(value_name)),
            "raw_value": _decode_raw_json_attr(value_group.attrs.get("raw_value")),
            "member_count": int(value_group.attrs.get("member_count", 0)),
            "members_path": (
                f"meta/axes/{value_name}/members" if "members" in value_group else None
            ),
            "payload": read_json_dataset(value_group, "json_payload"),
        }
        lookup[value_name] = entry
    return lookup


def iter_axis_reports(
    h5f: h5py.File,
) -> Iterator[Tuple[str, h5py.Group, Dict[str, Any]]]:
    """Iterate over all report entries.

    The ``reports/axes`` group contains ``value_XXXXXX`` groups directly.

    Yields:
        (value_token, report_group, axis_metadata_dict)
    """
    reports_root = h5f.get("reports/axes")
    if reports_root is None:
        return

    lookup = get_axis_metadata_lookup(h5f)

    for value_name, value_group in reports_root.items():
        if not value_name.startswith("value_"):
            continue
        metadata = lookup.get(value_name, {})
        yield value_name, value_group, metadata
