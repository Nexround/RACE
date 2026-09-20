"""Core utilities for LLM model introspection and weight extraction.

This module provides functions to access language model components
and extract projection weights needed for RACE analysis.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
import torch.nn as nn


def get_language_model(model: nn.Module) -> nn.Module:
    """Extract the language model backbone from a model.

    Args:
        model: Full model (e.g., LlamaForCausalLM, Qwen2ForCausalLM)

    Returns:
        Language model backbone (e.g., LlamaModel, Qwen2Model)
    """
    # Common pattern: model.model contains the backbone
    if hasattr(model, "model") and isinstance(model.model, nn.Module):
        return model.model

    # Direct model (already the backbone)
    return model


def get_decoder_layers(model: nn.Module) -> Sequence[nn.Module]:
    """Extract decoder layers from language model.

    Args:
        model: Language model backbone

    Returns:
        Sequence of decoder layer modules
    """
    # Common pattern: model.layers
    if hasattr(model, "layers"):
        return model.layers

    # Alternative: model.decoder.layers
    if hasattr(model, "decoder") and hasattr(model.decoder, "layers"):
        return model.decoder.layers

    # GPT-2 style: model.h (transformer blocks)
    if hasattr(model, "h"):
        return model.h

    # Alternative: model.transformer.h
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h

    raise AttributeError("Could not find decoder layers in model")


def find_attention_out_proj(layer: nn.Module) -> Optional[nn.Module]:
    """Find attention output projection module in a decoder layer.

    Searches for common naming patterns:
    - self_attn.o_proj (Llama, Qwen2, Mistral)
    - attn.out_proj (GPT-2)
    - attention.wo (some architectures)

    Args:
        layer: Single decoder layer module

    Returns:
        Output projection module or None if not found
    """
    # Pattern 1: self_attn.o_proj (most common for modern models)
    if hasattr(layer, "self_attn"):
        if hasattr(layer.self_attn, "o_proj"):
            return layer.self_attn.o_proj
        if hasattr(layer.self_attn, "out_proj"):
            return layer.self_attn.out_proj

    # Pattern 2: attn.out_proj (GPT-2 style)
    if hasattr(layer, "attn"):
        if hasattr(layer.attn, "out_proj"):
            return layer.attn.out_proj
        if hasattr(layer.attn, "c_proj"):  # GPT-2 naming
            return layer.attn.c_proj

    # Pattern 3: attention.wo
    if hasattr(layer, "attention") and hasattr(layer.attention, "wo"):
        return layer.attention.wo

    return None


def find_mlp_down_proj(layer: nn.Module) -> Optional[nn.Module]:
    """Find MLP down projection module in a decoder layer.

    Searches for common naming patterns:
    - mlp.down_proj (Llama, Qwen2, Mistral)
    - mlp.c_proj (GPT-2)
    - feed_forward.w2 (some architectures)

    Args:
        layer: Single decoder layer module

    Returns:
        Down projection module or None if not found
    """
    # Pattern 1: mlp.down_proj (most common)
    if hasattr(layer, "mlp"):
        if hasattr(layer.mlp, "down_proj"):
            return layer.mlp.down_proj
        if hasattr(layer.mlp, "c_proj"):  # GPT-2 naming
            return layer.mlp.c_proj

    # Pattern 2: feed_forward.w2
    if hasattr(layer, "feed_forward"):
        if hasattr(layer.feed_forward, "w2"):
            return layer.feed_forward.w2
        if hasattr(layer.feed_forward, "down_proj"):
            return layer.feed_forward.down_proj

    return None


def extract_attention_out_proj_weight(layer: nn.Module) -> Optional[torch.Tensor]:
    """Extract W_O weight matrix from attention output projection.

    Args:
        layer: Decoder layer module

    Returns:
        W_O weight tensor [hidden_dim, hidden_dim] transposed for h @ W or None
    """
    module = find_attention_out_proj(layer)
    if module is None:
        return None

    if not hasattr(module, "weight"):
        return None

    # Transpose weight: PyTorch Linear stores [out_features, in_features]
    # but we need [in_features, out_features] for h @ W_O
    weight = module.weight.data
    return weight.t()


def extract_mlp_down_proj_weight(layer: nn.Module) -> Optional[torch.Tensor]:
    """Extract W_down weight matrix from MLP down projection.

    Args:
        layer: Decoder layer module

    Returns:
        W_down weight tensor [d_model, d_ff] or None
    """
    module = find_mlp_down_proj(layer)
    if module is None:
        return None

    if not hasattr(module, "weight"):
        return None

    return module.weight


def print_model_structure(model: nn.Module, max_layers: int = 3) -> None:
    """Print model structure for debugging.

    Args:
        model: Model to inspect
        max_layers: Maximum number of layers to print details for
    """
    print(f"\n{'='*80}")
    print("Model Structure")
    print(f"{'='*80}")

    # Try to get language model
    try:
        lm = get_language_model(model)
        print(f"✓ Language model: {type(lm).__name__}")
    except Exception as e:
        print(f"✗ Could not extract language model: {e}")
        return

    # Try to get layers
    try:
        layers = get_decoder_layers(lm)
        print(f"✓ Number of decoder layers: {len(layers)}")
    except Exception as e:
        print(f"✗ Could not extract decoder layers: {e}")
        return

    # Inspect first few layers
    print(f"\nInspecting first {min(max_layers, len(layers))} layers:")
    for i, layer in enumerate(layers[:max_layers]):
        print(f"\n  Layer {i}:")
        print(f"    Type: {type(layer).__name__}")

        # Check attention
        attn_proj = find_attention_out_proj(layer)
        if attn_proj is not None:
            weight = extract_attention_out_proj_weight(layer)
            print(
                f"    ✓ Attention out_proj: {type(attn_proj).__name__} {tuple(weight.shape) if weight is not None else 'N/A'}"
            )
        else:
            print(f"    ✗ Attention out_proj: not found")

        # Check MLP
        mlp_proj = find_mlp_down_proj(layer)
        if mlp_proj is not None:
            weight = extract_mlp_down_proj_weight(layer)
            print(
                f"    ✓ MLP down_proj: {type(mlp_proj).__name__} {tuple(weight.shape) if weight is not None else 'N/A'}"
            )
        else:
            print(f"    ✗ MLP down_proj: not found")

    print(f"\n{'='*80}")
