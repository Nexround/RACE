"""RACE: Residual Alignment for Consistency Estimation.

RACE records language-model activations, analyzes neuron consistency with a
Normal-Inverse-Gamma posterior, and supports multiple evaluation datasets.
"""

__version__ = "0.1.0"
__author__ = "RACE Team"

__all__ = [
    # Core analysis
    "RACEConfig",
    "RACEState",
    "NIGPosterior",
    "init_nig",
    "H5ActivationReader",
    "load_activation_reader",
    # Online accumulator
    "OnlineRACEAccumulator",
    "OnlineRACEConfig",
    # LLM
    "LLMActivationRecorder",
    "LLMSampleResult",
]

# ---------------------------------------------------------------------------
# Lazy imports keep optional heavyweight dependencies out of ``import race``.
# ---------------------------------------------------------------------------
_LAZY_IMPORT_MAP: dict[str, tuple[str, str]] = {
    # core
    "RACEConfig": (".core.analyzer", "RACEConfig"),
    "RACEState": (".core.analyzer", "RACEState"),
    "NIGPosterior": (".core.analyzer", "NIGPosterior"),
    "init_nig": (".core.analyzer", "init_nig"),
    "H5ActivationReader": (".core.reader", "H5ActivationReader"),
    "load_activation_reader": (".core.reader", "load_activation_reader"),
    # Online accumulator
    "OnlineRACEAccumulator": (".core.online_accumulator", "OnlineRACEAccumulator"),
    "OnlineRACEConfig": (".core.online_accumulator", "OnlineRACEConfig"),
    # LLM
    "LLMActivationRecorder": (".llm.recording.activation", "LLMActivationRecorder"),
    "LLMSampleResult": (".llm.recording.activation", "LLMSampleResult"),
}


def __getattr__(name: str):
    from race._lazy import load_lazy_attr

    return load_lazy_attr(globals(), __name__, _LAZY_IMPORT_MAP, name)


def __dir__():
    return sorted(set(globals()) | set(__all__))
