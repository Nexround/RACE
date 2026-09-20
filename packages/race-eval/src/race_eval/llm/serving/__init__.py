"""RACE LLM inference serving package.

Provides an OpenAI-compatible FastAPI server that loads HuggingFace causal LMs
with optional RACE neuron intervention hooks, supports multi-GPU data-parallel
inference, and dynamic request batching.

Entry point::

    python -m race_eval.llm.serving.server --help
"""

__all__ = ["RaceInferenceServer"]


def __getattr__(name: str):
    if name == "RaceInferenceServer":
        from race_eval.llm.serving.server import RaceInferenceServer  # noqa: PLC0415

        return RaceInferenceServer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
