"""Unified H5 activation reader utilities.

This module provides a lightweight interface for inspecting activation HDF5
files emitted by activation recorders. It understands the shared schema
implemented in :mod:`race.io.h5_schema` and exposes convenience helpers for
retrieving run metadata, instance metadata, and activation tensors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import h5py
import numpy as np

from race.io.h5_schema import read_instance_metadata, read_json_dataset
from race.core.constants import MODULE_ALIASES


@dataclass
class InstanceMetadata:
    """Container for instance-level metadata."""

    instance_idx: int
    source_id: str
    task_family: str
    status: str
    error_message: str
    axis_refs: Sequence[str]
    payload: Dict


class H5ActivationReader:
    """Reader for unified activation H5 files."""

    def __init__(self, h5_file_path: str):
        self.h5_file_path = Path(h5_file_path)
        if not self.h5_file_path.exists():
            raise FileNotFoundError(f"H5 file not found: {h5_file_path}")

        self._handle: Optional[h5py.File] = None
        self._schema_info: Dict[str, str] = {}
        self._run_info: Dict[str, str] = {}
        self._instance_indices: List[int] = []

    # ------------------------------------------------------------------
    # Context manager helpers
    # ------------------------------------------------------------------
    def __enter__(self) -> "H5ActivationReader":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def open(self) -> None:
        if self._handle is not None:
            return
        self._handle = h5py.File(self.h5_file_path, "r")
        self._load_meta()
        self._load_instance_indices()

    def close(self) -> None:
        if self._handle is None:
            return
        self._handle.close()
        self._handle = None

    # ------------------------------------------------------------------
    # Private loaders
    # ------------------------------------------------------------------
    def _load_meta(self) -> None:
        assert self._handle is not None
        meta_group = self._handle.get("meta")
        if meta_group is None:
            raise ValueError("H5 file is missing the 'meta' group")

        schema_group = meta_group.get("schema")
        if schema_group is not None:
            self._schema_info = dict(schema_group.attrs)
        else:
            self._schema_info = {}

        run_group = meta_group.get("run")
        if run_group is None:
            raise ValueError("H5 file is missing the 'meta/run' group")

        run_info = {k: v for k, v in run_group.attrs.items()}
        run_payload = read_json_dataset(run_group, "json_payload")
        if run_payload:
            run_info["payload"] = run_payload
        self._run_info = run_info

    def _load_instance_indices(self) -> None:
        assert self._handle is not None
        instances_group = self._handle.get("instances")
        if instances_group is None:
            self._instance_indices = []
            return

        indices: List[int] = []
        for key in instances_group.keys():
            if not key.startswith("instance_"):
                continue
            try:
                indices.append(int(key.split("_")[1]))
            except ValueError:
                continue
        indices.sort()
        self._instance_indices = indices

    # ------------------------------------------------------------------
    # Public metadata accessors
    # ------------------------------------------------------------------
    @property
    def schema_info(self) -> Dict[str, str]:
        if self._handle is None:
            self.open()
        return dict(self._schema_info)

    @property
    def run_info(self) -> Dict[str, str]:
        if self._handle is None:
            self.open()
        return dict(self._run_info)

    @property
    def num_instances(self) -> int:
        if self._handle is None:
            self.open()
        return len(self._instance_indices)

    @property
    def instance_indices(self) -> List[int]:
        if self._handle is None:
            self.open()
        return list(self._instance_indices)

    # ------------------------------------------------------------------
    # Instance helpers
    # ------------------------------------------------------------------
    def _resolve_instance_group(self, instance_idx: int) -> h5py.Group:
        if self._handle is None:
            self.open()
        assert self._handle is not None

        group_name = f"instance_{instance_idx:06d}"
        instances_group = self._handle.get("instances")
        if instances_group is None or group_name not in instances_group:
            raise KeyError(f"Instance {instance_idx} not found")
        return instances_group[group_name]

    def get_instance_metadata(self, instance_idx: int) -> InstanceMetadata:
        """Return structured metadata for the given instance."""
        instance_group = self._resolve_instance_group(instance_idx)
        metadata_group = instance_group.get("metadata")
        if metadata_group is None:
            raise ValueError(f"Instance {instance_idx} has no metadata group")

        meta = read_instance_metadata(metadata_group)
        axis_refs_raw = metadata_group.attrs.get("source_axis_value_refs", "[]")
        if isinstance(axis_refs_raw, bytes):
            axis_refs_raw = axis_refs_raw.decode("utf-8")
        try:
            axis_refs = json.loads(axis_refs_raw) if axis_refs_raw else []
        except json.JSONDecodeError:
            axis_refs = []

        return InstanceMetadata(
            instance_idx=int(metadata_group.attrs.get("instance_idx", instance_idx)),
            source_id=str(metadata_group.attrs.get("source_id", "")),
            task_family=str(metadata_group.attrs.get("task_family", "")),
            status=str(metadata_group.attrs.get("status", "")),
            error_message=str(metadata_group.attrs.get("error_message", "")),
            axis_refs=axis_refs,
            payload=meta.get("payload", {}),
        )

    def get_instance_activations(
        self,
        instance_idx: int,
        *,
        modules: Optional[Sequence[str]] = None,
        layer_idx: Optional[int] = None,
        record_idx: Optional[int] = None,
        dtype: np.dtype = np.float32,
    ) -> Dict[str, Dict[int, List[np.ndarray]]]:
        """Load activation tensors for an instance.

        Args:
            instance_idx: Target instance index.
            modules: Optional list of module identifiers to include. Use schema
                names such as ``"attn_pre_output"`` or friendly aliases
                (``"attn"``, ``"ffn"``). When omitted, all modules are loaded.
            layer_idx: Optional layer restriction.
            record_idx: Optional restriction that selects one record.
            dtype: NumPy dtype to cast the activations to (default: float32).
        """
        target_modules = None
        if modules is not None:
            target_modules = {MODULE_ALIASES.get(mod, mod) for mod in modules}

        instance_group = self._resolve_instance_group(instance_idx)
        payload_group = instance_group.get("payload")
        if payload_group is None:
            return {}
        activations_group = payload_group.get("activations")
        if activations_group is None:
            return {}

        activation_map: Dict[str, Dict[int, List[np.ndarray]]] = {}
        for module_name, module_group in activations_group.items():
            canonical_name = MODULE_ALIASES.get(module_name, module_name)
            if target_modules is not None and canonical_name not in target_modules:
                continue

            layer_map: Dict[int, List[np.ndarray]] = {}
            layer_keys = (
                [f"layer_{layer_idx:02d}"]
                if layer_idx is not None
                else sorted(k for k in module_group.keys() if k.startswith("layer_"))
            )

            for layer_key in layer_keys:
                if layer_key not in module_group:
                    continue
                layer_group = module_group[layer_key]
                layer_index = int(layer_key.split("_")[-1])

                record_keys = (
                    [f"record_{record_idx:03d}"]
                    if record_idx is not None
                    else sorted(
                        k for k in layer_group.keys() if k.startswith("record_")
                    )
                )

                records: List[np.ndarray] = []
                for record_key in record_keys:
                    if record_key not in layer_group:
                        continue
                    data = layer_group[record_key][:]
                    array = np.asarray(
                        data, dtype=np.float32 if dtype is None else dtype
                    )
                    records.append(array)

                if records:
                    layer_map[layer_index] = records

            if layer_map:
                activation_map[canonical_name] = layer_map

        return activation_map

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    def get_layer_statistics(
        self,
        instance_indices: Iterable[int],
        module: str,
        layer_idx: int,
    ) -> Dict[str, float]:
        """Compute simple statistics for a layer across instances."""
        canonical_module = MODULE_ALIASES.get(module, module)
        vectors: List[np.ndarray] = []

        for idx in instance_indices:
            acts = self.get_instance_activations(
                idx, modules=[canonical_module], layer_idx=layer_idx
            )
            module_map = acts.get(canonical_module)
            if module_map is None or layer_idx not in module_map:
                continue
            for record in module_map[layer_idx]:
                vectors.append(record.reshape(-1))

        if not vectors:
            return {}

        stacked = np.concatenate(vectors, axis=0)
        return {
            "mean": float(stacked.mean()),
            "std": float(stacked.std()),
            "min": float(stacked.min()),
            "max": float(stacked.max()),
        }


def load_activation_reader(h5_file_path: str) -> H5ActivationReader:
    """Convenience constructor."""
    return H5ActivationReader(h5_file_path)
