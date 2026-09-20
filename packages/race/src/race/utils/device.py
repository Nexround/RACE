"""Shared utility functions for the RACE package.

This module centralizes common helpers (device resolution, etc.) that are
used across CV, LLM and VLM sub-packages so they are defined once.
"""

from __future__ import annotations

import torch


def resolve_device(device_spec: str = "auto") -> torch.device:
    """Resolve a user-supplied device specification to a concrete ``torch.device``.

    Supported values for *device_spec*:
    - ``"auto"`` (default) — pick CUDA > MPS > CPU in that order.
    - ``"cpu"``, ``"cuda"``, ``"cuda:0"``, ``"mps"`` — use the requested
      device, falling back to CPU when unavailable.

    Args:
        device_spec: Device string (case-insensitive).

    Returns:
        The resolved :class:`torch.device`.
    """
    device_spec = (device_spec or "auto").lower()

    if device_spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    requested = torch.device(device_spec)

    if requested.type == "cuda" and not torch.cuda.is_available():
        print(
            f"[RACE] Requested device '{device_spec}' is unavailable. "
            "Falling back to CPU."
        )
        return torch.device("cpu")

    if requested.type == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            print(
                f"[RACE] Requested device '{device_spec}' is unavailable. "
                "Falling back to CPU."
            )
            return torch.device("cpu")

    return requested
