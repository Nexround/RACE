"""RACE/RACE results loading utilities for the unified schema."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

import h5py
import numpy as np

from race.core.constants import LCB_DATASET_ALIASES, MODULE_ALIASES
from race.io.h5_schema import get_axis_metadata_lookup

LCB_DATASET_PRIORITY = ("cam_score", "credible_lcb", "lcb")
_MODULE_TO_SCHEMA = {
    "attn": "attn_pre_output",
    "attn_pre_output": "attn_pre_output",
    "ffn": "mlp_pre_down",
    "mlp": "mlp_pre_down",
    "mlp_pre_down": "mlp_pre_down",
}
_MODULE_TO_DISPLAY = {
    "attn_pre_output": "attn",
    "attn": "attn",
    "mlp_pre_down": "ffn",
    "ffn": "ffn",
    "mlp": "ffn",
}


@dataclass(frozen=True)
class RaceAxisValue:
    """Metadata for one value under ``/meta/axes`` and ``/reports/axes``."""

    value_name: str
    value_id: int
    axis_type: str
    display_name: str
    raw_value: Any
    payload: Dict[str, Any]
    member_count: int = 0
    members_path: Optional[str] = None

    def as_metadata(self) -> Dict[str, Any]:
        return {
            "axis_type": self.axis_type,
            "display_name": self.display_name,
            "raw_value": self.raw_value,
            "payload": self.payload,
            "member_count": self.member_count,
            "members_path": self.members_path,
        }


@dataclass(frozen=True)
class RaceLayerMetric:
    """One metric vector from one axis value, module, and layer."""

    axis: RaceAxisValue
    module: str
    schema_module: str
    layer_idx: int
    metric: str
    values: np.ndarray


AxisSelector = Union[RaceAxisValue, int, str]


def decode_h5_attr(value):
    """Decode HDF5 attribute value to Python native types."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"S", "U"}:
            return (
                value.tobytes().decode("utf-8")
                if value.dtype.kind == "S"
                else value.tolist()
            )
        if value.shape == ():
            return value.item()
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _parse_value_id(value_name: str) -> int:
    try:
        return int(value_name.split("_")[-1])
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"Invalid axis value name: {value_name!r}") from exc


def _canonical_module_name(module: str) -> str:
    return _MODULE_TO_DISPLAY.get(module, MODULE_ALIASES.get(module, module))


def _schema_module_name(module: str) -> str:
    return _MODULE_TO_SCHEMA.get(module, module)


def resolve_metric_dataset_name(layer_group: h5py.Group, metric: str) -> str:
    """Resolve a user-facing metric name to the dataset stored in a layer group."""
    metric_key = metric.lower()
    if metric_key == "lcb_neg":
        if "negative_cam_score" in layer_group:
            return "negative_cam_score"
        raise KeyError(
            f"Layer group {layer_group.name!r} has no negative CAM dataset. "
            "Regenerate the H5 with this paper-aligned implementation."
        )
    if metric_key in LCB_DATASET_ALIASES or metric_key in {"lcb_pos", "lcb_neg"}:
        if metric_key in layer_group:
            return metric_key
        for candidate in LCB_DATASET_PRIORITY:
            if candidate in layer_group:
                return candidate
        raise KeyError(
            f"Layer group {layer_group.name!r} has no LCB/CAM score dataset "
            f"(tried: {', '.join(LCB_DATASET_PRIORITY)})"
        )

    if metric_key in layer_group:
        return metric_key
    raise KeyError(
        f"Layer group {layer_group.name!r} has no dataset for metric {metric!r}"
    )


def read_layer_metric(
    layer_group: h5py.Group,
    metric: str,
    *,
    dtype: Optional[np.dtype] = np.float32,
) -> np.ndarray:
    """Read one per-neuron metric vector from a layer group."""
    dataset_name = resolve_metric_dataset_name(layer_group, metric)
    values = layer_group[dataset_name][()]
    array = np.asarray(values)
    if dtype is not None:
        array = array.astype(dtype)
    return array


class RaceResultsReader:
    """Structured reader for unified-schema RACE/RACE result H5 files."""

    def __init__(self, h5_path: str):
        self.h5_path = Path(h5_path)
        if not self.h5_path.exists():
            raise FileNotFoundError(f"H5 file not found: {h5_path}")
        self._handle: Optional[h5py.File] = None

    def __enter__(self) -> "RaceResultsReader":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def open(self) -> None:
        if self._handle is None:
            self._handle = h5py.File(self.h5_path, "r")

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def _require_handle(self) -> h5py.File:
        if self._handle is None:
            self.open()
        assert self._handle is not None
        return self._handle

    @property
    def schema_info(self) -> Dict[str, Any]:
        handle = self._require_handle()
        group = handle.get("meta/schema")
        if group is None:
            return {}
        return {key: decode_h5_attr(value) for key, value in group.attrs.items()}

    @property
    def run_info(self) -> Dict[str, Any]:
        handle = self._require_handle()
        group = handle.get("meta/run")
        if group is None:
            return {}
        return {key: decode_h5_attr(value) for key, value in group.attrs.items()}

    @property
    def race_config(self) -> Dict[str, Any]:
        handle = self._require_handle()
        group = handle.get("meta/race_config")
        if group is None:
            return {}
        return {key: decode_h5_attr(value) for key, value in group.attrs.items()}

    @property
    def summary_metrics(self) -> Dict[str, Any]:
        handle = self._require_handle()
        group = handle.get("reports/summary/metrics")
        if group is None:
            return {}
        return {key: decode_h5_attr(value) for key, value in group.attrs.items()}

    def list_axis_values(
        self,
        axis_type: Optional[str] = None,
        *,
        reports_only: bool = False,
    ) -> List[RaceAxisValue]:
        """Return axis values, optionally restricted to values with reports."""
        handle = self._require_handle()
        axis_lookup = get_axis_metadata_lookup(handle)

        value_names = set(axis_lookup)
        reports_root = handle.get("reports/axes")
        if reports_root is not None:
            report_names = {
                key for key in reports_root.keys() if key.startswith("value_")
            }
            value_names = report_names if reports_only else value_names | report_names
        elif reports_only:
            value_names = set()

        values: List[RaceAxisValue] = []
        for value_name in sorted(value_names):
            meta = axis_lookup.get(value_name, {})
            raw_value = decode_h5_attr(meta.get("raw_value"))
            display_name = str(decode_h5_attr(meta.get("display_name") or value_name))
            axis_type_value = str(decode_h5_attr(meta.get("axis_type") or ""))
            entry = RaceAxisValue(
                value_name=value_name,
                value_id=_parse_value_id(value_name),
                axis_type=axis_type_value,
                display_name=display_name,
                raw_value=raw_value,
                payload=dict(meta.get("payload", {}) or {}),
                member_count=int(meta.get("member_count", 0) or 0),
                members_path=decode_h5_attr(meta.get("members_path")),
            )
            if axis_type is not None and entry.axis_type != axis_type:
                continue
            values.append(entry)
        return values

    def resolve_axis_value(
        self,
        value: AxisSelector,
        *,
        axis_type: Optional[str] = None,
        reports_only: bool = False,
    ) -> RaceAxisValue:
        """Resolve ``value_XXXXXX``, numeric id, display name, or raw value."""
        if isinstance(value, RaceAxisValue):
            return value

        candidates = self.list_axis_values(
            axis_type=axis_type, reports_only=reports_only
        )
        if isinstance(value, int):
            value_name = f"value_{value:06d}"
            for candidate in candidates:
                if candidate.value_name == value_name:
                    return candidate
            raise KeyError(f"Axis value {value_name!r} not found")

        needle = str(value).strip()
        if needle.isdigit():
            token = f"value_{int(needle):06d}"
            for candidate in candidates:
                if candidate.value_name == token:
                    return candidate

        needle_lower = needle.lower()
        for candidate in candidates:
            if candidate.value_name.lower() == needle_lower:
                return candidate
            if candidate.display_name.lower() == needle_lower:
                return candidate
            raw_value = candidate.raw_value
            raw_matches = False
            if isinstance(raw_value, dict):
                raw_matches = any(
                    str(raw).lower() == needle_lower for raw in raw_value.values()
                )
            elif raw_value is not None:
                raw_matches = str(raw_value).lower() == needle_lower
            if raw_matches:
                return candidate
            if any(
                str(raw).lower() == needle_lower for raw in candidate.payload.values()
            ):
                return candidate

        available = [candidate.display_name for candidate in candidates]
        raise KeyError(f"Axis value {value!r} not found. Available: {available}")

    def _report_group(self, axis_value: AxisSelector) -> h5py.Group:
        handle = self._require_handle()
        axis = self.resolve_axis_value(axis_value, reports_only=True)
        path = f"reports/axes/{axis.value_name}"
        group = handle.get(path)
        if not isinstance(group, h5py.Group):
            raise KeyError(f"Report group {path!r} not found")
        return group

    def _modules_group(self, axis_value: AxisSelector) -> h5py.Group:
        report_group = self._report_group(axis_value)
        modules_group = report_group.get("modules")
        if not isinstance(modules_group, h5py.Group):
            raise KeyError(f"Report group {report_group.name!r} has no modules group")
        return modules_group

    def list_modules(
        self,
        axis_value: AxisSelector,
        *,
        canonical: bool = True,
    ) -> List[str]:
        modules_group = self._modules_group(axis_value)
        modules = sorted(modules_group.keys())
        if not canonical:
            return modules
        return list(dict.fromkeys(_canonical_module_name(module) for module in modules))

    def get_layer_group(
        self,
        axis_value: AxisSelector,
        module: str,
        layer_idx: int,
    ) -> h5py.Group:
        modules_group = self._modules_group(axis_value)
        schema_module = _schema_module_name(module)
        module_group = modules_group.get(schema_module)
        if not isinstance(module_group, h5py.Group):
            raise KeyError(
                f"Axis {axis_value!r} has no module {module!r} ({schema_module!r})"
            )
        layer_key = f"layer_{int(layer_idx):02d}"
        layer_group = module_group.get(layer_key)
        if not isinstance(layer_group, h5py.Group):
            raise KeyError(f"Module {schema_module!r} has no layer {int(layer_idx)}")
        return layer_group

    def list_layers(self, axis_value: AxisSelector, module: str) -> List[int]:
        modules_group = self._modules_group(axis_value)
        schema_module = _schema_module_name(module)
        module_group = modules_group.get(schema_module)
        if not isinstance(module_group, h5py.Group):
            return []
        layers = []
        for key in module_group.keys():
            if not key.startswith("layer_"):
                continue
            try:
                layers.append(int(key.split("_")[-1]))
            except ValueError:
                continue
        return sorted(layers)

    def list_metrics(
        self,
        axis_value: AxisSelector,
        module: str,
        layer_idx: int,
    ) -> List[str]:
        layer_group = self.get_layer_group(axis_value, module, layer_idx)
        return sorted(
            key for key, value in layer_group.items() if isinstance(value, h5py.Dataset)
        )

    def read_metric(
        self,
        axis_value: AxisSelector,
        module: str,
        layer_idx: int,
        metric: str,
        *,
        dtype: Optional[np.dtype] = np.float32,
    ) -> np.ndarray:
        layer_group = self.get_layer_group(axis_value, module, layer_idx)
        return read_layer_metric(layer_group, metric, dtype=dtype)

    def iter_layer_metrics(
        self,
        *,
        axis_values: Optional[Sequence[AxisSelector]] = None,
        modules: Optional[Sequence[str]] = None,
        layers: Optional[Sequence[int]] = None,
        metric: str = "posterior_mean",
        dtype: Optional[np.dtype] = np.float32,
    ) -> Iterator[RaceLayerMetric]:
        selected_axes = (
            [self.resolve_axis_value(axis, reports_only=True) for axis in axis_values]
            if axis_values is not None
            else self.list_axis_values(reports_only=True)
        )
        layer_filter = {int(layer) for layer in layers} if layers is not None else None

        for axis in selected_axes:
            selected_modules = (
                list(modules) if modules is not None else self.list_modules(axis)
            )
            for module in selected_modules:
                schema_module = _schema_module_name(module)
                for layer_idx in self.list_layers(axis, module):
                    if layer_filter is not None and layer_idx not in layer_filter:
                        continue
                    yield RaceLayerMetric(
                        axis=axis,
                        module=_canonical_module_name(module),
                        schema_module=schema_module,
                        layer_idx=layer_idx,
                        metric=metric,
                        values=self.read_metric(
                            axis, module, layer_idx, metric, dtype=dtype
                        ),
                    )

    def validate(
        self,
        *,
        required_metrics: Sequence[str] = ("posterior_mean",),
        require_modules: bool = True,
    ) -> List[str]:
        """Return schema/data issues found in the result file."""
        issues: List[str] = []
        handle = self._require_handle()
        if handle.get("meta") is None:
            issues.append("missing group: meta")
        if handle.get("reports/axes") is None:
            issues.append("missing group: reports/axes")
            return issues

        for axis in self.list_axis_values(reports_only=True):
            try:
                modules_group = self._modules_group(axis)
            except KeyError as exc:
                issues.append(str(exc))
                continue
            if require_modules and len(modules_group) == 0:
                issues.append(f"{axis.value_name}: modules group is empty")
            for module_name, module_group in modules_group.items():
                if not isinstance(module_group, h5py.Group):
                    continue
                layer_keys = sorted(
                    key for key in module_group.keys() if key.startswith("layer_")
                )
                if not layer_keys:
                    issues.append(f"{axis.value_name}/{module_name}: no layer groups")
                    continue
                for layer_key in layer_keys:
                    layer_group = module_group[layer_key]
                    if not isinstance(layer_group, h5py.Group):
                        continue
                    dim = int(layer_group.attrs.get("feature_dim", 0) or 0)
                    for metric in required_metrics:
                        try:
                            values = read_layer_metric(layer_group, metric, dtype=None)
                        except KeyError as exc:
                            issues.append(str(exc))
                            continue
                        if dim > 0 and values.shape[-1] != dim:
                            issues.append(
                                f"{layer_group.name}: metric {metric!r} has shape "
                                f"{values.shape}, expected last dimension {dim}"
                            )
        return issues
