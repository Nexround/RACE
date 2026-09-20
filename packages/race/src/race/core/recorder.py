"""Base activation recorder abstraction.

This module provides :class:`BaseActivationRecorder`, an abstract base class
that factors out the hook-lifecycle, model-management and activation-storage
patterns used by :class:`LLMActivationRecorder`.

Subclasses must implement:
- ``load_model()``  — model/tokenizer loading logic (architecture-specific).
- ``register_hooks()`` — hook placement on the correct submodules.
"""

from __future__ import annotations

import abc
import os
from typing import Any, List, Optional

import torch
from torch import nn


class BaseActivationRecorder(abc.ABC):
    """Abstract base class for activation recorders.

    Shared responsibilities:
    - Manage ``model``, ``device`` and ``output_dir`` attributes.
    - Maintain ``hooks`` list with ``remove_hooks()`` cleanup.
    - Provide ``clear_activations()`` that empties all activation dicts.
    """

    def __init__(self, model_name: str, output_dir: str):
        self.model_name = model_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.model: Optional[nn.Module] = None
        self.device: torch.device = torch.device("cpu")
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def load_model(self, device: str = "cuda", **kwargs: Any) -> None:
        """Load the model (and processor/tokenizer) onto *device*."""
        ...

    @abc.abstractmethod
    def register_hooks(self) -> None:
        """Register forward / forward-pre hooks on the model."""
        ...

    # ------------------------------------------------------------------
    # Shared hook lifecycle
    # ------------------------------------------------------------------

    def remove_hooks(self) -> None:
        """Detach all registered hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()

    @abc.abstractmethod
    def clear_activations(self) -> None:
        """Clear any cached activation tensors."""
        ...

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "BaseActivationRecorder":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.remove_hooks()
