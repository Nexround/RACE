"""Unit tests for race.utils.device — device resolution."""

import pytest
import torch

from race.utils.device import resolve_device


class TestResolveDevice:
    def test_auto(self):
        device = resolve_device("auto")
        assert isinstance(device, torch.device)
        # Should pick one of the available devices
        assert device.type in ("cuda", "mps", "cpu")

    def test_cpu(self):
        device = resolve_device("cpu")
        assert device == torch.device("cpu")

    def test_case_insensitive(self):
        device = resolve_device("CPU")
        assert device == torch.device("cpu")

    def test_none_defaults_to_auto(self):
        device = resolve_device(None)
        assert isinstance(device, torch.device)

    def test_empty_string_defaults_to_auto(self):
        device = resolve_device("")
        assert isinstance(device, torch.device)

    def test_cuda_fallback(self):
        """If CUDA is unavailable, should fall back to CPU."""
        if not torch.cuda.is_available():
            device = resolve_device("cuda")
            assert device == torch.device("cpu")

    def test_cuda_available(self):
        """If CUDA is available, should return cuda device."""
        if torch.cuda.is_available():
            device = resolve_device("cuda")
            assert device.type == "cuda"
