"""LLM-RACE: RACE analysis for decoder-only language models.

This package provides activation recording, evidence accumulation, and dataset
utilities for applying RACE to large language models.

Exports are lazy. Importing ``race.llm`` does not load PyTorch or Transformers
until an exported name is accessed.
"""

from __future__ import annotations

__all__ = [
    # Model introspection
    "get_language_model",
    "get_decoder_layers",
    "find_attention_out_proj",
    "find_mlp_down_proj",
    "extract_attention_out_proj_weight",
    "extract_mlp_down_proj_weight",
    # Recording
    "LLMActivationRecorder",
    "LLMSampleResult",
    # Prefill recorder
    "PrefillActivationRecorder",
    "PrefillSampleResult",
]

_LAZY_IMPORT_MAP: dict[str, tuple[str, str]] = {
    # model_utils
    "get_language_model": (".model_utils", "get_language_model"),
    "get_decoder_layers": (".model_utils", "get_decoder_layers"),
    "find_attention_out_proj": (".model_utils", "find_attention_out_proj"),
    "find_mlp_down_proj": (".model_utils", "find_mlp_down_proj"),
    "extract_attention_out_proj_weight": (
        ".model_utils",
        "extract_attention_out_proj_weight",
    ),
    "extract_mlp_down_proj_weight": (".model_utils", "extract_mlp_down_proj_weight"),
    # recording.activation
    "LLMActivationRecorder": (".recording.activation", "LLMActivationRecorder"),
    "LLMSampleResult": (".recording.activation", "LLMSampleResult"),
    # recording.prefill_recorder
    "PrefillActivationRecorder": (
        ".recording.prefill_recorder",
        "PrefillActivationRecorder",
    ),
    "PrefillSampleResult": (".recording.prefill_recorder", "PrefillSampleResult"),
}


def __getattr__(name: str):
    from race._lazy import load_lazy_attr

    return load_lazy_attr(globals(), __name__, _LAZY_IMPORT_MAP, name)


def __dir__():
    return sorted(set(globals()) | set(__all__))
