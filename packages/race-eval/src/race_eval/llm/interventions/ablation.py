"""Neuron ablation/modification utilities for LLM models.

This module provides tools to suppress or enhance specific neurons in
transformer-based language models, enabling causal intervention experiments
to validate RACE attribution results.

Supports zero ablation and mean ablation. Mean ablation avoids out-of-
distribution residual-stream collapse in LayerNorm-based architectures.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
from torch import nn

from race_eval.ablation import AblationPlan, OperationMode
from race_eval.ablation.intervention import BaseNeuronModifier
from race.llm.model_utils import (
    find_attention_out_proj,
    find_mlp_down_proj,
    get_decoder_layers,
    get_language_model,
)


class LLMNeuronModifier(BaseNeuronModifier):
    """Modify LLM neurons according to an ablation plan.

    This class registers forward hooks that intercept and modify neuron
    activations in real-time during model inference.

    Args:
        model: The language model to modify.
        plan: Ablation plan specifying which neurons to modify.
        mean_activations: Per-hook-point mean activation tensors, keyed as
            ``"attn_layer_{i}"`` / ``"mlp_layer_{i}"``.  Required when
            ``plan.operation_mode == "mean_ablation"``.
    """

    def __init__(
        self,
        model: nn.Module,
        plan: AblationPlan,
        mean_activations: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__(model, plan)
        self.mean_activations = mean_activations

    # ------------------------------------------------------------------
    # BaseNeuronModifier abstract methods
    # ------------------------------------------------------------------

    def _get_layers(self) -> Sequence[nn.Module]:
        lm = get_language_model(self.model)
        return list(get_decoder_layers(lm))

    def _find_attn_proj(self, layer: nn.Module) -> Optional[nn.Module]:
        return find_attention_out_proj(layer)

    def _find_ffn_proj(self, layer: nn.Module) -> Optional[nn.Module]:
        return find_mlp_down_proj(layer)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _resolve_hook(self, neuron_indices, op_mode, factor, mean_key: str):
        """Build the right hook depending on operation mode."""
        is_mean = op_mode in ("mean_ablation", OperationMode.MEAN_ABLATION.value)
        if is_mean:
            if self.mean_activations is None or mean_key not in self.mean_activations:
                raise RuntimeError(
                    f"Mean ablation requested but no mean_activations for {mean_key!r}. "
                    "Run compute_mean_activations() first."
                )
            return self._make_mean_ablation_hook(
                neuron_indices,
                self.mean_activations[mean_key],
            )
        return self._make_modification_hook(neuron_indices, op_mode, factor)

    def apply(self) -> None:
        """Apply the ablation plan by registering modification hooks."""
        self.remove()

        if self.model is None:
            raise RuntimeError(
                "Model must be provided before applying neuron modifiers"
            )

        decoder_layers = self._get_layers()

        op_mode = self.plan.operation_mode
        factor = self.plan.enhancement_factor

        # Register attention modification hooks
        for layer_idx, neuron_indices in self.plan.attn_neurons.items():
            if layer_idx >= len(decoder_layers):
                continue
            layer = decoder_layers[layer_idx]
            out_proj = self._find_attn_proj(layer)
            if out_proj is not None:
                hook = self._resolve_hook(
                    neuron_indices,
                    op_mode,
                    factor,
                    f"attn_layer_{layer_idx}",
                )
                self.hooks.append(out_proj.register_forward_pre_hook(hook))

        # Register MLP modification hooks
        for layer_idx, neuron_indices in self.plan.mlp_neurons.items():
            if layer_idx >= len(decoder_layers):
                continue
            layer = decoder_layers[layer_idx]
            down_proj = self._find_ffn_proj(layer)
            if down_proj is not None:
                hook = self._resolve_hook(
                    neuron_indices,
                    op_mode,
                    factor,
                    f"mlp_layer_{layer_idx}",
                )
                self.hooks.append(down_proj.register_forward_pre_hook(hook))

        print(
            f"Applied {len(self.hooks)} modification hooks for {self.plan.total_neurons} neurons"
        )

    def remove_hooks(self) -> None:
        """Remove all modification hooks."""
        self.remove()
