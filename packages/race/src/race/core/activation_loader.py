"""Shared utilities for loading and batching activation data from H5 files.

This module provides common functions for efficiently loading activation tensors
from H5 files, supporting both single-threaded and multi-process loading strategies.
"""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch


def collect_layer_vectors_as_tensor(
    layer_group: h5py.Group, device: torch.device
) -> Optional[torch.Tensor]:
    """Load all record_XXX datasets within a layer as a concatenated float32 tensor.

    Args:
        layer_group: H5 group containing record_XXX datasets
        device: Target device for the tensor

    Returns:
        Concatenated tensor of shape (total_records, feature_dim) or None if no data.
        If a record has shape (n, d), all records are concatenated along dim 0.
    """
    record_keys = [k for k in layer_group.keys() if k.startswith("record_")]
    if not record_keys:
        return None
    record_keys.sort()

    vectors: List[torch.Tensor] = []
    for key in record_keys:
        data = layer_group[key][:]
        if data.size == 0:
            continue
        # Convert numpy array to tensor directly
        tensor = torch.from_numpy(data).to(dtype=torch.float32, device=device)
        # Ensure 2D: (n_samples, feature_dim)
        if tensor.ndim > 2:
            tensor = tensor.reshape(-1, tensor.shape[-1])
        elif tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        vectors.append(tensor)

    if not vectors:
        return None
    return torch.cat(vectors, dim=0)


def collect_layer_vectors_as_numpy(layer_group: h5py.Group) -> Optional[np.ndarray]:
    """Load all record_XXX datasets within a layer as a stacked numpy array.

    Args:
        layer_group: H5 group containing record_XXX datasets

    Returns:
        Stacked array of shape (n_records, feature_dim) or None if no data
    """
    record_keys = sorted(k for k in layer_group.keys() if k.startswith("record_"))
    if not record_keys:
        return None

    vectors: List[np.ndarray] = []
    for key in record_keys:
        data = layer_group[key][:]
        if data.size == 0:
            continue
        arr = np.asarray(data, dtype=np.float32)
        # Handle different dimensionalities
        if arr.ndim > 2:
            arr = arr.reshape(-1, arr.shape[-1])
        elif arr.ndim == 1:
            arr = arr.reshape(1, -1)
        vectors.append(arr)

    if not vectors:
        return None
    return np.concatenate(vectors, axis=0)


def batch_collect_activations_as_tensors(
    instance_groups: List[h5py.Group],
    module_path: str,
    layer_indices: Iterable[int],
    device: torch.device,
) -> Dict[int, List[torch.Tensor]]:
    """Batch collect activations for multiple instances as tensors.

    Args:
        instance_groups: List of H5 instance groups to process
        module_path: Path to module activations (e.g., "attn_pre_output", "mlp_pre_down")
        layer_indices: Layer indices to collect
        device: Target device for tensors

    Returns:
        Dict mapping layer_idx -> List of tensors (one per instance)
    """
    layer_activations: Dict[int, List[torch.Tensor]] = {
        idx: [] for idx in layer_indices
    }

    for instance_group in instance_groups:
        payload_group = instance_group.get("payload")
        if payload_group is None:
            continue
        activations_group = payload_group.get("activations")
        if activations_group is None:
            continue
        module_group = activations_group.get(module_path)
        if module_group is None:
            continue

        for layer_idx in layer_indices:
            layer_key = f"layer_{layer_idx:02d}"
            if layer_key not in module_group:
                continue

            vectors = collect_layer_vectors_as_tensor(module_group[layer_key], device)
            if vectors is not None:
                layer_activations[layer_idx].append(vectors)

    return layer_activations


def read_instance_module_activations_numpy(
    args: Tuple[str, str, str, Tuple[int, ...]],
) -> Dict[int, np.ndarray]:
    """Worker helper to load activations for a single instance as numpy arrays.

    This function is designed for multiprocessing - it opens its own H5 file handle.

    Args:
        args: Tuple of (h5_path, instance_key, module_path, layer_indices)

    Returns:
        Mapping layer_idx -> numpy array stacked along record axis
    """
    h5_path, instance_key, module_path, layer_indices = args
    results: Dict[int, np.ndarray] = {}

    with h5py.File(h5_path, "r") as h5f:
        instances_group = h5f.get("instances")
        if instances_group is None:
            return results
        instance_group = instances_group.get(instance_key)
        if instance_group is None:
            return results

        payload_group = instance_group.get("payload")
        if payload_group is None:
            return results
        activations_group = payload_group.get("activations")
        if activations_group is None:
            return results
        module_group = activations_group.get(module_path)
        if module_group is None:
            return results

        for layer_idx in layer_indices:
            layer_key = f"layer_{layer_idx:02d}"
            if layer_key not in module_group:
                continue
            layer_group = module_group[layer_key]

            vectors = collect_layer_vectors_as_numpy(layer_group)
            if vectors is not None:
                results[layer_idx] = vectors

    return results


def _read_batch_instances_numpy(
    args: Tuple[str, Tuple[str, ...], str, Tuple[int, ...]],
) -> List[Dict[int, np.ndarray]]:
    """Worker helper to load activations for *multiple* instances in a single H5 open.

    This amortises the cost of opening the H5 file across many instances,
    which is significantly faster than opening once per instance.

    Args:
        args: Tuple of (h5_path, instance_keys, module_path, layer_indices)

    Returns:
        List of dicts, one per instance key, mapping layer_idx -> numpy array.
    """
    h5_path, instance_keys, module_path, layer_indices = args
    batch_results: List[Dict[int, np.ndarray]] = []

    with h5py.File(h5_path, "r") as h5f:
        instances_group = h5f.get("instances")
        if instances_group is None:
            return [{} for _ in instance_keys]

        for instance_key in instance_keys:
            results: Dict[int, np.ndarray] = {}
            instance_group = instances_group.get(instance_key)
            if instance_group is None:
                batch_results.append(results)
                continue

            payload_group = instance_group.get("payload")
            if payload_group is None:
                batch_results.append(results)
                continue
            activations_group = payload_group.get("activations")
            if activations_group is None:
                batch_results.append(results)
                continue
            module_group = activations_group.get(module_path)
            if module_group is None:
                batch_results.append(results)
                continue

            for layer_idx in layer_indices:
                layer_key = f"layer_{layer_idx:02d}"
                if layer_key not in module_group:
                    continue
                vectors = collect_layer_vectors_as_numpy(module_group[layer_key])
                if vectors is not None:
                    results[layer_idx] = vectors

            batch_results.append(results)

    return batch_results


def batch_collect_activations_multiprocess(
    pool,
    h5_path: str,
    instance_keys: Sequence[str],
    module_path: str,
    layer_indices: Iterable[int],
    device: torch.device,
    instances_per_task: int = 16,
) -> Dict[int, List[torch.Tensor]]:
    """Collect activations using multiple worker processes, then convert to tensors.

    Each worker opens the H5 file once and reads ``instances_per_task`` instances,
    amortising file-open overhead.

    Args:
        pool: multiprocessing.Pool instance
        h5_path: Path to H5 file
        instance_keys: List of instance keys to process
        module_path: Path to module activations
        layer_indices: Layer indices to collect
        device: Target device for final tensors
        instances_per_task: Number of instances each worker reads per H5 open

    Returns:
        Dict mapping layer_idx -> List of tensors (one per instance)
    """
    layer_indices_tuple = tuple(layer_indices)
    layer_activations: Dict[int, List[torch.Tensor]] = {
        idx: [] for idx in layer_indices_tuple
    }

    # Chunk instance keys so each worker handles multiple instances per file open
    chunked_keys: List[Tuple[str, ...]] = []
    for start in range(0, len(instance_keys), instances_per_task):
        chunk = tuple(instance_keys[start : start + instances_per_task])
        chunked_keys.append(chunk)

    tasks = [
        (h5_path, chunk, module_path, layer_indices_tuple) for chunk in chunked_keys
    ]

    # Workers load as numpy in batches, then we convert to tensors
    for batch_results in pool.imap_unordered(_read_batch_instances_numpy, tasks):
        for result in batch_results:
            for layer_idx, array in result.items():
                tensor = torch.from_numpy(array).to(dtype=torch.float32, device=device)
                layer_activations[layer_idx].append(tensor)

    return layer_activations
