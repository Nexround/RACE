"""Core RACE algorithms, activation readers, and analysis utilities.

Exports are lazy. Importing ``race.core`` does not load heavyweight optional
dependencies such as PyTorch or h5py until their exported names are accessed.
"""

from __future__ import annotations

__all__ = [
    # RACE
    "RACEConfig",
    "RACEState",
    "NIGPosterior",
    "init_nig",
    "compute_attn_evidence_batch",
    "compute_ffn_evidence_batch",
    "LayerWeights",
    "H5ActivationReader",
    "load_activation_reader",
    # Online accumulator
    "OnlineRACEAccumulator",
    "OnlineRACEConfig",
    "OnlineRACEState",
    "DomainState",
    "SampleStreamContext",
    # Lightweight constants imported directly below
    "MODULE_ALIASES",
    "LCB_DATASET_ALIASES",
    # Recorder base class
    "BaseActivationRecorder",
    # Results loader helpers
    "RaceAxisValue",
    "RaceLayerMetric",
    "RaceResultsReader",
    "decode_h5_attr",
    "read_layer_metric",
    "resolve_metric_dataset_name",
    # Recompute posteriors
    "recompute_posteriors",
]

# ---------------------------------------------------------------------------
# Constants have no heavy deps – import eagerly so they are always available
# without any attribute access needed.
# ---------------------------------------------------------------------------
from .constants import MODULE_ALIASES, LCB_DATASET_ALIASES  # noqa: E402

# ---------------------------------------------------------------------------
# Lazy import map
# ---------------------------------------------------------------------------
_LAZY_IMPORT_MAP: dict[str, tuple[str, str]] = {
    # analyzer
    "RACEConfig": (".analyzer", "RACEConfig"),
    "RACEState": (".analyzer", "RACEState"),
    "NIGPosterior": (".analyzer", "NIGPosterior"),
    "init_nig": (".analyzer", "init_nig"),
    "compute_attn_evidence_batch": (".analyzer", "compute_attn_evidence_batch"),
    "compute_ffn_evidence_batch": (".analyzer", "compute_ffn_evidence_batch"),
    "LayerWeights": (".analyzer", "LayerWeights"),
    # reader
    "H5ActivationReader": (".reader", "H5ActivationReader"),
    "load_activation_reader": (".reader", "load_activation_reader"),
    # online_accumulator
    "OnlineRACEAccumulator": (".online_accumulator", "OnlineRACEAccumulator"),
    "OnlineRACEConfig": (".online_accumulator", "OnlineRACEConfig"),
    "OnlineRACEState": (".online_accumulator", "OnlineRACEState"),
    "DomainState": (".online_accumulator", "DomainState"),
    "SampleStreamContext": (".online_accumulator", "SampleStreamContext"),
    # recorder base class
    "BaseActivationRecorder": (".recorder", "BaseActivationRecorder"),
    # results_loader helpers
    "RaceAxisValue": (".results_loader", "RaceAxisValue"),
    "RaceLayerMetric": (".results_loader", "RaceLayerMetric"),
    "RaceResultsReader": (".results_loader", "RaceResultsReader"),
    "decode_h5_attr": (".results_loader", "decode_h5_attr"),
    "read_layer_metric": (".results_loader", "read_layer_metric"),
    "resolve_metric_dataset_name": (
        ".results_loader",
        "resolve_metric_dataset_name",
    ),
    # recompute_posteriors
    "recompute_posteriors": (".recompute_posteriors", "recompute_posteriors"),
}


def __getattr__(name: str):
    from race._lazy import load_lazy_attr

    return load_lazy_attr(globals(), __name__, _LAZY_IMPORT_MAP, name)


def __dir__():
    return sorted(set(globals()) | set(__all__))
