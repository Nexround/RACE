"""Save perturbed LLM models with RACE-identified neurons modified in weights.

Instead of using forward hooks at runtime (as in ``generate_ablation.py``),
this script **directly modifies the model weights** and saves the resulting
model to disk.  The saved model is a standard HuggingFace checkpoint that
can be evaluated by *any* downstream pipeline (vLLM, HF generate, etc.).

Mathematical equivalence
~~~~~~~~~~~~~~~~~~~~~~~~
The hook-based approach registers a ``forward_pre_hook`` on projection layers
(``o_proj`` / ``down_proj``) that multiplies the *input* activations by a mask::

    x_modified = x * mask          # mask[i]=0 zeroes neuron i
    y = x_modified @ W.T + b

This is algebraically equivalent to zeroing the corresponding **columns** of
the weight matrix::

    W_modified = W.clone()
    W_modified[:, i] = 0           # suppress neuron i
    y = x @ W_modified.T + b

For ``enhance`` mode the column is scaled by the enhancement factor instead.

Mean ablation
~~~~~~~~~~~~~
For ``mean_ablation`` the hook replaces neuron values with their calibration
mean rather than zeroing them.  The weight-surgery equivalent folds the
constant mean contribution into a **bias term** on the projection layer::

    delta_b = W[:, ablated_idx] @ mean_values[ablated_idx]
    W[:, ablated_idx] = 0
    b_new = b_old + delta_b        # bias is created if the layer has none

This eliminates *all* runtime hook overhead (no Python callbacks during
``model.generate()``).

When the architecture does not natively support bias on certain layers
(e.g. Qwen3's ``down_proj``), the checkpoint includes a generated
``modeling_race_perturbed.py`` that defines:

* ``RacePerturbedMLP`` — a subclass of the original MLP that re-creates
  ``down_proj`` with ``bias=True`` when ``config.mlp_bias`` is set.
* A thin causal-LM wrapper (e.g. ``RacePerturbedQwen3ForCausalLM``) that
  inherits from the base model and swaps in ``RacePerturbedMLP`` at
  ``__init__`` time so the safetensors state-dict loads correctly.

The ``config.json`` ``architectures`` / ``auto_map`` fields point at this
custom class so ``from_pretrained(..., trust_remote_code=True)`` resolves
it automatically.  No original modeling or configuration source files are
modified.

Supported modes
~~~~~~~~~~~~~~~
- **suppress**       — zero out top-k% important neuron columns
- **enhance**        — scale top-k% important neuron columns by a factor
- **keep_top**       — zero out *all but* top-k% important neurons
- **mean_ablation**  — replace with calibration mean (folded into weights)

Layer selection
~~~~~~~~~~~~~~
- **all layers** (default)
- **specific layers** via ``--layers 0 1 2``
- **layer-by-layer** via ``--layer-by-layer N`` — saves one model per
  N-layer group (storage-intensive for large models!)

Usage examples::

    # Suppress top 5% neurons across all layers
    uv run python -m race_eval.llm.pipelines.save_perturbed_model \\
        --model Qwen/Qwen3-4B-Instruct-2507 \\
        --race-h5 result/llm/online_race/mbpp_plus_online_race.h5 \\
        --concept mbpp_plus \\
        --operation suppress \\
        --top-k-percent 5.0 \\
        --output-dir result/llm/perturbed_models

    # Mean ablation (no runtime hooks needed)
    uv run python -m race_eval.llm.pipelines.save_perturbed_model \\
        --model Qwen/Qwen3-4B-Instruct-2507 \\
        --race-h5 result/llm/online_race/.../online_race.h5 \\
        --concept math500 \\
        --operation mean_ablation \\
        --top-k 8 --metric lcb_pos --lcb-min-score 0 \\
        --modules attn \\
        --output-dir result/llm/perturbed_models

    # Layer-by-layer (one model per layer)
    uv run python -m race_eval.llm.pipelines.save_perturbed_model \\
        --model Qwen/Qwen3-4B-Instruct-2507 \\
        --race-h5 result/llm/online_race/mbpp_plus_online_race.h5 \\
        --concept mbpp_plus \\
        --operation suppress \\
        --top-k-percent 5.0 \\
        --layer-by-layer 1 \\
        --output-dir result/llm/perturbed_models

    # Paper RSF: exclude the WikiText-2 Top-M, then select k target neurons
    uv run python -m race_eval.llm.pipelines.save_perturbed_model \\
        --model Qwen/Qwen3-4B-Instruct-2507 \\
        --race-h5 result/llm/online_race/mbpp_plus_online_race.h5 \\
        --concept mbpp_plus \\
        --operation suppress \\
        --top-k-percent 1.0 \\
        --general-race-h5 result/llm/online_race/wikitext2_online_race.h5 \\
        --general-concept wikitext2 \\
        --output-dir result/llm/perturbed_models

Loading mean-ablated models::

    # Standard HuggingFace loading — just add trust_remote_code=True
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        "result/llm/perturbed_models/Qwen3-4B--mean_ablation_math500_...",
        trust_remote_code=True,
    )

    # vLLM loading — use --trust-remote-code so vLLM resolves the custom
    # architecture via the Transformers modeling backend:
    #
    #   vllm serve <checkpoint> --trust-remote-code
    #
    # The generated ``config.json`` contains both mappings:
    #   "auto_map": {
    #       "AutoModel":           "modeling_race_perturbed.RacePerturbedXxxModel",
    #       "AutoModelForCausalLM": "modeling_race_perturbed.RacePerturbedXxxForCausalLM"
    #   }
    # vLLM's Transformers modeling backend looks up ``auto_map["AutoModel"]``
    # and checks that the class has ``_supports_attention_backend = True``
    # (which ``RacePerturbedXxxModel`` provides).  This allows vLLM to swap
    # in its own CUDA attention kernels transparently.
    #
    # For suppress / enhance / keep_top models the architecture name is
    # unchanged (e.g. "Qwen3ForCausalLM"), so vLLM loads them natively
    # without --trust-remote-code.

"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from race.llm.model_utils import (
    find_attention_out_proj,
    find_mlp_down_proj,
    get_decoder_layers,
    get_language_model,
)
from race_eval.ablation import (
    AblationPlan,
    OperationMode,
    create_ablation_plan,
    create_reference_filtered_ablation_plan,
    list_llm_h5_layer_indices,
)
from race_eval.llm.perturbed_model import materialize_race_perturbed_class

logger = logging.getLogger(__name__)


# ======================================================================
# Weight modification
# ======================================================================


def modify_weights_inplace(
    model: torch.nn.Module,
    plan: AblationPlan,
    mean_activations: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, int]:
    """Directly modify model weights according to the ablation plan.

    For each (layer, module, neuron_indices) entry in the plan, the
    corresponding **columns** of the projection weight matrix are modified:

    - suppress / keep_top  → ``W[:, idx] = 0``
    - enhance              → ``W[:, idx] *= factor``
    - mean_ablation        → ``W[:, idx] = 0`` + fold mean into bias

    Args:
        model: HuggingFace causal LM (e.g. ``AutoModelForCausalLM``).
        plan:  Ablation plan with ``attn_neurons`` / ``mlp_neurons``.
        mean_activations: Per-hook-point mean activation tensors, keyed as
            ``"attn_layer_{i}"`` / ``"mlp_layer_{i}"``.  Required when
            ``plan.operation_mode == "mean_ablation"``.

    Returns:
        Statistics dict with counts of modified neurons per module.
    """
    lm = get_language_model(model)
    decoder_layers = list(get_decoder_layers(lm))

    op = plan.operation_mode
    if isinstance(op, OperationMode):
        op = op.value
    factor = plan.enhancement_factor

    if op == "mean_ablation" and not mean_activations:
        raise ValueError(
            "mean_activations is required for mean_ablation weight surgery"
        )

    stats: Dict[str, int] = {
        "attn_layers_modified": 0,
        "attn_neurons_modified": 0,
        "mlp_layers_modified": 0,
        "mlp_neurons_modified": 0,
    }

    # --- Attention output projection ---
    for layer_idx, neuron_indices in plan.attn_neurons.items():
        if layer_idx >= len(decoder_layers):
            logger.warning(
                "Skipping attn layer %d (model has %d layers)",
                layer_idx,
                len(decoder_layers),
            )
            continue
        layer = decoder_layers[layer_idx]
        out_proj = find_attention_out_proj(layer)
        if out_proj is None:
            logger.warning("Layer %d: attention out_proj not found", layer_idx)
            continue

        idx = sorted(int(i) for i in neuron_indices)
        mean_key = f"attn_layer_{layer_idx}"
        mean_vals = mean_activations.get(mean_key) if mean_activations else None
        _apply_weight_modification(out_proj, idx, op, factor, mean_values=mean_vals)
        stats["attn_layers_modified"] += 1
        stats["attn_neurons_modified"] += len(idx)
        logger.info(
            "  attn layer %2d: modified %d neurons (%s)",
            layer_idx,
            len(idx),
            op,
        )

    # --- MLP down projection ---
    for layer_idx, neuron_indices in plan.mlp_neurons.items():
        if layer_idx >= len(decoder_layers):
            logger.warning(
                "Skipping mlp layer %d (model has %d layers)",
                layer_idx,
                len(decoder_layers),
            )
            continue
        layer = decoder_layers[layer_idx]
        down_proj = find_mlp_down_proj(layer)
        if down_proj is None:
            logger.warning("Layer %d: mlp down_proj not found", layer_idx)
            continue

        idx = sorted(int(i) for i in neuron_indices)
        mean_key = f"mlp_layer_{layer_idx}"
        mean_vals = mean_activations.get(mean_key) if mean_activations else None
        _apply_weight_modification(down_proj, idx, op, factor, mean_values=mean_vals)
        stats["mlp_layers_modified"] += 1
        stats["mlp_neurons_modified"] += len(idx)
        logger.info(
            "  mlp  layer %2d: modified %d neurons (%s)",
            layer_idx,
            len(idx),
            op,
        )

    return stats


def _apply_weight_modification(
    linear: torch.nn.Module,
    neuron_indices: List[int],
    operation: str,
    enhancement_factor: float,
    mean_values: Optional[torch.Tensor] = None,
) -> None:
    """Modify columns of a ``nn.Linear`` weight matrix in-place.

    ``nn.Linear`` stores weights as ``[out_features, in_features]``.
    Zeroing the input dimension *i* (as the hook does) is equivalent to
    zeroing column *i* of the weight: ``weight[:, i] = 0``.

    For ``mean_ablation`` the hook replaces ``input[..., idx]`` with the
    calibration mean *before* the linear layer.  Algebraically::

        y = x @ W^T + b
          = Σ_{i∉idx} x_i W[:,i]  +  Σ_{i∈idx} mean_i W[:,i]  +  b

    So we (1) fold the mean contribution into the bias:
    ``b_new = b + W[:, idx] @ mean_subset``, then (2) zero the columns:
    ``W[:, idx] = 0``.  A bias parameter is created when the layer has
    none (common for Qwen / Llama architectures).
    """
    assert hasattr(linear, "weight") and isinstance(linear.weight, torch.nn.Parameter)
    weight = linear.weight.data  # [out_features, in_features]
    idx_tensor = torch.tensor(neuron_indices, dtype=torch.long, device=weight.device)

    if operation in ("suppress", "keep_top"):
        weight[:, idx_tensor] = 0.0
    elif operation == "enhance":
        weight[:, idx_tensor] *= enhancement_factor
    elif operation == "mean_ablation":
        if mean_values is None:
            raise ValueError("mean_values required for mean_ablation")
        idx_cpu = torch.tensor(neuron_indices, dtype=torch.long)
        mean_subset = mean_values[idx_cpu].to(
            device=weight.device,
            dtype=weight.dtype,
        )
        # Constant contribution of ablated dims: delta_b = W[:, idx] @ mean
        delta_b = weight[:, idx_tensor] @ mean_subset
        weight[:, idx_tensor] = 0.0

        if linear.bias is None:
            linear.bias = torch.nn.Parameter(
                torch.zeros(
                    weight.shape[0],
                    device=weight.device,
                    dtype=weight.dtype,
                ),
            )
        linear.bias.data += delta_b
    else:
        raise ValueError(f"Unknown operation: {operation}")


def _patch_config_for_mean_ablation(
    model: torch.nn.Module,
    plan: AblationPlan,
) -> None:
    """Set config flags and ensure every projection layer has a bias parameter.

    This is a weight-surgery helper that runs **before** the model is saved.
    It does *not* write patched modeling source files — that responsibility
    has moved to :func:`~race_eval.llm.perturbed_model.materialize_race_perturbed_class`,
    which generates a self-contained ``modeling_race_perturbed.py`` subclass.

    1. **Attention** — sets ``config.attention_bias = True`` and adds zero
       biases to all q/k/v/o projections so the state dict is complete.
    2. **MLP** — sets ``config.mlp_bias = True`` and adds zero biases to
       every ``down_proj`` layer that weight surgery did not already equip
       with a bias.  The generated ``RacePerturbedMLP`` subclass (written by
       :func:`materialize_race_perturbed_class`) picks up this flag at
       ``__init__`` time and re-creates ``down_proj`` with ``bias=True``
       before the safetensors shards are loaded, so the bias tensors land in
       the right parameter slots.
    """
    lm = get_language_model(model)
    decoder_layers = list(get_decoder_layers(lm))

    # --- Attention: config.attention_bias + zero biases for q/k/v/o ---
    if plan.attn_neurons:
        if not getattr(model.config, "attention_bias", False):
            model.config.attention_bias = True
            logger.info(
                "Set config.attention_bias = True for %d attn layers",
                len(plan.attn_neurons),
            )
        for layer in decoder_layers:
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                continue
            for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                proj = getattr(attn, proj_name, None)
                if (
                    proj is not None
                    and isinstance(proj, torch.nn.Linear)
                    and proj.bias is None
                ):
                    proj.bias = torch.nn.Parameter(
                        torch.zeros(
                            proj.out_features,
                            device=proj.weight.device,
                            dtype=proj.weight.dtype,
                        ),
                    )

    # --- MLP: config.mlp_bias + zero biases for layers not yet patched ---
    if plan.mlp_neurons:
        model.config.mlp_bias = True
        # mlp_bias is global: ALL layers' down_proj must have a bias tensor in
        # the state dict so that the safetensors loader doesn't complain about
        # unexpected / missing keys.  Add a zero bias wherever weight surgery
        # didn't create one already.
        for layer in decoder_layers:
            mlp_proj = find_mlp_down_proj(layer)
            if (
                mlp_proj is not None
                and isinstance(mlp_proj, torch.nn.Linear)
                and mlp_proj.bias is None
            ):
                mlp_proj.bias = torch.nn.Parameter(
                    torch.zeros(
                        mlp_proj.out_features,
                        device=mlp_proj.weight.device,
                        dtype=mlp_proj.weight.dtype,
                    ),
                )
        logger.info(
            "Set config.mlp_bias = True; zero-biases added where needed "
            "(RacePerturbedMLP subclass handles down_proj at load time)",
        )


# ======================================================================
# Model name / path helpers
# ======================================================================


def _make_model_dir_name(
    model_name: str,
    concept: str,
    operation: str,
    top_k: float,
    modules: List[str],
    metric: str,
    layer_tag: str = "",
    *,
    top_k_count: Optional[int] = None,
    selection_tag: str = "",
) -> str:
    """Build a descriptive directory name for the saved model."""
    # Keep only the final segment so `org/model` becomes `model`.
    safe_model = model_name.rstrip("/").split("/")[-1]
    mod_str = "_".join(sorted(modules))
    if top_k_count is not None:
        top_tag = f"k{top_k_count}"
    else:
        top_tag = f"top{top_k}"
    return (
        f"{safe_model}--{operation}_{concept}_{metric}_{top_tag}_{mod_str}"
        f"{selection_tag}{layer_tag}"
    )


# ======================================================================
# Save one perturbed model
# ======================================================================


def _count_neurons_by_layer(plan: AblationPlan) -> Dict[str, Dict[str, int]]:
    """Return per-layer perturbation counts for metadata output."""
    layer_indices = sorted(set(plan.attn_neurons.keys()) | set(plan.mlp_neurons.keys()))
    return {
        str(layer_idx): {
            "attn": len(plan.attn_neurons.get(layer_idx, ())),
            "mlp": len(plan.mlp_neurons.get(layer_idx, ())),
            "total": (
                len(plan.attn_neurons.get(layer_idx, ()))
                + len(plan.mlp_neurons.get(layer_idx, ()))
            ),
        }
        for layer_idx in layer_indices
    }


def save_single_perturbed_model(
    model_name: str,
    plan: AblationPlan,
    output_dir: str,
    *,
    model_dir_name: str,
    trust_remote_code: bool = False,
    dtype: str = "bfloat16",
    extra_metadata: Optional[Dict] = None,
    calibration_corpus: str = "wikitext2",
    calibration_samples: int = 200,
    calibration_max_length: int = 2048,
    calibration_batch_size: int = 4,
    calibration_device: str = "cuda:0",
) -> str:
    """Load a model, modify weights, and save to disk.

    For ``suppress``, ``enhance``, and ``keep_top`` the model is loaded on
    CPU, weights are modified in place, and the checkpoint is saved without
    generating extra files.

    For ``mean_ablation`` the model is moved to GPU, calibration mean
    activations are computed, the mean contribution is folded into bias terms,
    config flags are patched, and a ``modeling_race_perturbed.py`` wrapper
    module is written alongside the checkpoint so the checkpoint can be loaded
    with ``AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)``.

    Args:
        model_name: HuggingFace model identifier or local path.
        plan: Ablation plan to apply.
        output_dir: Root output directory.
        model_dir_name: Subdirectory name for this variant.
        trust_remote_code: Trust remote code when loading model.
        dtype: Data type string (``"bfloat16"``, ``"float16"``, …).
        extra_metadata: Additional metadata to include in JSON.
        calibration_corpus: Corpus for mean-activation calibration
            (``mean_ablation`` only).
        calibration_samples: Number of calibration passages
            (``mean_ablation`` only).
        calibration_max_length: Token-length cap per calibration passage
            (``mean_ablation`` only).
        calibration_batch_size: Batch size for calibration forward passes
            (``mean_ablation`` only).
        calibration_device: Device for calibration forward passes
            (``mean_ablation`` only).

    Returns:
        Path to the saved model directory.
    """
    save_path = os.path.join(output_dir, model_dir_name)
    os.makedirs(save_path, exist_ok=True)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    model_dtype = dtype_map.get(dtype, torch.bfloat16)

    op = plan.operation_mode
    if isinstance(op, OperationMode):
        op = op.value
    is_mean_ablation = op == "mean_ablation"

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    logger.info("Loading model: %s (dtype=%s) ...", model_name, dtype)
    t0 = time.perf_counter()
    # Always load on CPU first for portability; move to the calibration device
    # afterwards when mean_ablation needs GPU forward passes.  Passing a bare
    # device string like "cuda:0" directly to ``device_map`` is not an
    # officially supported value and can break across Transformers versions or
    # with certain backends.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=model_dtype,
        trust_remote_code=trust_remote_code,
        device_map=None,
    )
    if is_mean_ablation:
        logger.info("  moving model to %s for calibration ...", calibration_device)
        model = model.to(calibration_device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dt_load = time.perf_counter() - t0
    logger.info("  loaded in %.1fs", dt_load)

    # ------------------------------------------------------------------
    # 2. Compute mean activations (mean_ablation only)
    # ------------------------------------------------------------------
    mean_activations: Optional[Dict[str, torch.Tensor]] = None
    dt_calibrate = 0.0
    if is_mean_ablation:
        from race_eval.llm.evaluation.mean_activations import compute_mean_activations

        logger.info(
            "Computing calibration means on %s (%d samples) ...",
            calibration_corpus,
            calibration_samples,
        )
        t0 = time.perf_counter()
        mean_activations = compute_mean_activations(
            model,
            tokenizer,
            calibration_corpus=calibration_corpus,
            max_samples=calibration_samples,
            max_length=calibration_max_length,
            batch_size=calibration_batch_size,
        )
        mean_activations = {k: v.cpu() for k, v in mean_activations.items()}
        dt_calibrate = time.perf_counter() - t0
        logger.info("  calibration done in %.1fs", dt_calibrate)

    # ------------------------------------------------------------------
    # 3. Apply weight modifications
    # ------------------------------------------------------------------
    logger.info("Applying weight modifications ...")
    t0 = time.perf_counter()
    stats = modify_weights_inplace(model, plan, mean_activations=mean_activations)
    dt_modify = time.perf_counter() - t0
    logger.info(
        "  modified %d attn + %d mlp neurons in %.1fs",
        stats["attn_neurons_modified"],
        stats["mlp_neurons_modified"],
        dt_modify,
    )

    # ------------------------------------------------------------------
    # 4 & 5. mean_ablation only: patch config + generate wrapper module
    # ------------------------------------------------------------------
    wrapper_class_name: Optional[str] = None
    if is_mean_ablation:
        # 4. Set config.attention_bias / config.mlp_bias and add zero-bias
        #    parameters to every projection layer so the state dict is complete.
        _patch_config_for_mean_ablation(model, plan)

        # 5. Write modeling_race_perturbed.py and update config auto_map so
        #    AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)
        #    resolves to the RacePerturbedMLP-aware subclass.
        wrapper_class_name = materialize_race_perturbed_class(model, save_path)

    # ------------------------------------------------------------------
    # 6. Save
    # ------------------------------------------------------------------
    logger.info("Saving perturbed model to: %s", save_path)
    t0 = time.perf_counter()
    model.save_pretrained(save_path, safe_serialization=True)
    tokenizer.save_pretrained(save_path)
    dt_save = time.perf_counter() - t0
    logger.info("  saved in %.1fs", dt_save)

    # ------------------------------------------------------------------
    # 7. Write metadata JSON
    # ------------------------------------------------------------------
    timing: Dict[str, float] = {
        "load_s": round(dt_load, 2),
        "modify_s": round(dt_modify, 2),
        "save_s": round(dt_save, 2),
    }
    if is_mean_ablation:
        timing["calibrate_s"] = round(dt_calibrate, 2)

    meta: Dict[str, object] = {
        "source_model": model_name,
        "concept": plan.concept_name,
        "operation_mode": op,
        "top_k_percent": plan.top_percent,
        "top_k_count": plan.top_k_count,
        "enhancement_factor": plan.enhancement_factor,
        "total_neurons_modified": plan.total_neurons,
        "neuron_counts_by_layer": _count_neurons_by_layer(plan),
        "modification_stats": stats,
        "timing": timing,
        "dtype": dtype,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "attn_neurons": {str(k): sorted(v) for k, v in plan.attn_neurons.items()},
        "mlp_neurons": {str(k): sorted(v) for k, v in plan.mlp_neurons.items()},
    }
    if is_mean_ablation:
        meta["wrapper_class"] = wrapper_class_name
        meta["calibration"] = {
            "corpus": calibration_corpus,
            "samples": calibration_samples,
            "max_length": calibration_max_length,
        }
    if extra_metadata:
        meta.update(extra_metadata)

    json_path = os.path.join(save_path, "perturbation_config.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    logger.info("  metadata -> %s", json_path)

    # Free memory
    del model
    torch.cuda.empty_cache()

    return save_path


# ======================================================================
# Main pipeline
# ======================================================================


def run_perturbation_pipeline(
    model_name: str,
    race_h5: str,
    concept: str,
    operation: str,
    top_k_percent: float,
    output_dir: str,
    *,
    metric: str = "lcb_pos",
    modules: Optional[List[str]] = None,
    layers: Optional[List[int]] = None,
    layer_by_layer: Optional[int] = None,
    enhancement_factor: float = 2.0,
    trust_remote_code: bool = False,
    dtype: str = "bfloat16",
    lcb_min_score: Optional[float] = None,
    top_k_count: Optional[int] = None,
    calibration_corpus: str = "wikitext2",
    calibration_samples: int = 200,
    calibration_max_length: int = 2048,
    calibration_batch_size: int = 4,
    calibration_device: str = "cuda:0",
    general_race_h5: Optional[str] = None,
    general_concept: Optional[str] = None,
    general_top_k_count: Optional[int] = None,
    general_top_k_percent: Optional[float] = None,
) -> List[str]:
    """Run the full perturbation → save pipeline.

    Args:
        model_name: HuggingFace model identifier.
        race_h5: Path to RACE results H5 file.
        concept: Concept name (e.g. ``"code_generation"``).
        operation: ``"suppress"``, ``"enhance"``, ``"keep_top"``, or
            ``"mean_ablation"``.
        top_k_percent: Percentage of top neurons to modify (unused for k when
            ``top_k_count`` is set).
        output_dir: Root output directory.
        metric: Importance metric (``"posterior_mean"``, ``"lcb"``,
            ``"lcb_pos"``, or ``"lcb_neg"``).
        modules: Modules to include (``["attn", "mlp"]``).
        layers: Specific layer indices to modify.
        layer_by_layer: If set, group size for layer-by-layer ablation.
        enhancement_factor: Factor for enhance mode.
        trust_remote_code: Trust remote code for HF model loading.
        dtype: Data type for model loading.
        lcb_min_score: For LCB-family metrics (``lcb``, ``lcb_pos``, ``lcb_neg``),
            only scores above this value are candidates for top-k%%;
            ``None`` uses the complete layer-local universe, as in the paper.
        top_k_count: If set, take this many top neurons per layer and per module
            (attn / mlp), capped by pool size; overrides percentage-based k.
        calibration_corpus: Corpus for mean-ablation calibration.
        calibration_samples: Number of calibration passages.
        calibration_max_length: Token-length cap per calibration passage.
        calibration_batch_size: Batch size for calibration forward passes.
        calibration_device: Device for calibration forward passes.
        general_race_h5: Optional path to a general-knowledge RACE H5 file.
            When provided, apply the paper's Reference-Set Filtering (RSF):
            traverse each complete target layer/module ranking, skip neurons in
            the corresponding reference Top-M set, and stop after selecting k.
            Target and reference use the same ``metric``.
        general_concept: Concept name to use when reading ``general_race_h5``.
            Required when ``general_race_h5`` is set.
        general_top_k_count: Reference exclusion budget M per layer/module.
            Defaults to the target ``top_k_count`` when neither reference budget
            option is set.
        general_top_k_percent: Reference exclusion budget as top-p percent.
            Defaults to ``top_k_percent``.

    Returns:
        List of paths to saved model directories.
    """
    if top_k_count is not None and top_k_count < 0:
        raise ValueError("top_k_count must be non-negative")
    if general_top_k_count is not None and general_top_k_count < 0:
        raise ValueError("general_top_k_count must be non-negative")
    if general_top_k_percent is not None and general_top_k_percent < 0:
        raise ValueError("general_top_k_percent must be non-negative")
    if general_top_k_count is not None and general_top_k_percent is not None:
        raise ValueError(
            "general_top_k_count and general_top_k_percent are mutually exclusive"
        )
    if (
        general_top_k_count is not None or general_top_k_percent is not None
    ) and not general_race_h5:
        raise ValueError("reference budget options require general_race_h5")
    if general_race_h5 and not general_concept:
        raise ValueError("general_concept is required when general_race_h5 is set")

    if modules is None:
        modules = ["attn", "mlp"]
    operation_mode = OperationMode(operation)
    os.makedirs(output_dir, exist_ok=True)
    saved_paths: List[str] = []

    extra_meta: Dict[str, object] = {
        "race_h5": os.path.abspath(race_h5),
        "metric": metric,
        "modules": modules,
    }
    if lcb_min_score is not None:
        extra_meta["lcb_min_score"] = lcb_min_score
        if metric not in ("lcb", "lcb_pos", "lcb_neg"):
            logger.warning(
                "--lcb-min-score is set but metric is %r; threshold applies only for lcb/lcb_pos/lcb_neg.",
                metric,
            )
    if top_k_count is not None:
        extra_meta["top_k_count"] = top_k_count
    if general_race_h5:
        extra_meta["general_race_h5"] = os.path.abspath(general_race_h5)
        extra_meta["general_concept"] = general_concept
        extra_meta["general_metric"] = metric
        extra_meta["selection_method"] = "reference_set_filtering"
        extra_meta["general_top_k_count"] = general_top_k_count
        extra_meta["general_top_k_percent"] = general_top_k_percent

    def _create_plan(layer_filter: Optional[List[int]]) -> AblationPlan:
        common = dict(
            operation_mode=operation_mode,
            top_k_percent=top_k_percent,
            metric=metric,
            enhancement_factor=enhancement_factor,
            modules=modules,
            layer_filter=layer_filter,
            lcb_min_score=lcb_min_score,
            top_k_count=top_k_count,
        )
        if general_race_h5:
            assert general_concept is not None
            return create_reference_filtered_ablation_plan(
                target_h5_path=race_h5,
                target_concept=concept,
                reference_h5_path=general_race_h5,
                reference_concept=general_concept,
                reference_top_k_count=general_top_k_count,
                reference_top_k_percent=general_top_k_percent,
                **common,
            )
        return create_ablation_plan(
            h5_path=race_h5,
            concept_name=concept,
            **common,
        )

    if layer_by_layer is not None:
        # ---- Layer-by-layer mode ----
        logger.info("=" * 70)
        logger.info("  Layer-by-layer perturbation (group_size=%d)", layer_by_layer)
        if top_k_count is not None:
            logger.info("  Top-k count: %d (per layer, per module)", top_k_count)
        if lcb_min_score is not None:
            logger.info(
                "  LCB min score: %s (candidates: score > threshold)",
                lcb_min_score,
            )
        logger.info("=" * 70)

        # Layer indices from H5 (not from the plan: thresholded LCB can omit layers).
        all_layer_indices = list_llm_h5_layer_indices(
            race_h5,
            concept,
            modules,
        )
        if not all_layer_indices:
            logger.warning("No layer groups in RACE H5 — nothing to do.")
            return saved_paths

        max_layer = max(all_layer_indices)
        num_layers = max_layer + 1

        layer_groups = []
        for start in range(0, num_layers, layer_by_layer):
            group = list(range(start, min(start + layer_by_layer, num_layers)))
            layer_groups.append(group)

        logger.info("Total layers: %d, groups: %d", num_layers, len(layer_groups))

        for gi, group in enumerate(layer_groups):
            if layer_by_layer == 1:
                layer_tag = f"_layer{group[0]}"
                tag_display = f"layer {group[0]}"
            else:
                layer_tag = f"_layers{group[0]}-{group[-1]}"
                tag_display = f"layers {group[0]}-{group[-1]}"

            logger.info(
                "\n[%d/%d] Processing %s ...",
                gi + 1,
                len(layer_groups),
                tag_display,
            )

            if general_race_h5:
                logger.info("  applying RSF for %s ...", tag_display)
            plan = _create_plan(group)

            domain_total = sum(len(v) for v in plan.attn_neurons.values()) + sum(
                len(v) for v in plan.mlp_neurons.values()
            )
            if domain_total == 0:
                logger.info("  (no neurons in %s — skipped)", tag_display)
                continue

            dir_name = _make_model_dir_name(
                model_name,
                concept,
                operation,
                top_k_percent,
                modules,
                metric,
                layer_tag=layer_tag,
                top_k_count=top_k_count,
                selection_tag="_rsf" if general_race_h5 else "",
            )
            path = save_single_perturbed_model(
                model_name,
                plan,
                output_dir,
                model_dir_name=dir_name,
                trust_remote_code=trust_remote_code,
                dtype=dtype,
                extra_metadata={**extra_meta, "layer_group": group},
                calibration_corpus=calibration_corpus,
                calibration_samples=calibration_samples,
                calibration_max_length=calibration_max_length,
                calibration_batch_size=calibration_batch_size,
                calibration_device=calibration_device,
            )
            saved_paths.append(path)

    else:
        # ---- Standard mode (all layers or specific layers) ----
        logger.info("=" * 70)
        logger.info("  Standard perturbation")
        logger.info("=" * 70)

        plan = _create_plan(layers)

        logger.info("Concept      : %s", concept)
        logger.info("Operation    : %s", operation)
        if top_k_count is not None:
            logger.info("Top-k count  : %d (per layer, per module)", top_k_count)
        else:
            logger.info("Top-k%%       : %.1f", top_k_percent)
        if lcb_min_score is not None:
            logger.info(
                "LCB min score: %s (candidates: score > threshold)", lcb_min_score
            )
        logger.info("Modules      : %s", modules)
        logger.info("Layers       : %s", layers or "all")

        domain_total = sum(len(v) for v in plan.attn_neurons.values()) + sum(
            len(v) for v in plan.mlp_neurons.values()
        )
        logger.info("Total neurons: %d", domain_total)

        if domain_total == 0:
            logger.warning("No neurons found — nothing to save.")
            return saved_paths

        layer_tag = ""
        if layers:
            layer_tag = f"_layers{'_'.join(map(str, layers))}"

        dir_name = _make_model_dir_name(
            model_name,
            concept,
            operation,
            top_k_percent,
            modules,
            metric,
            layer_tag=layer_tag,
            top_k_count=top_k_count,
            selection_tag="_rsf" if general_race_h5 else "",
        )
        path = save_single_perturbed_model(
            model_name,
            plan,
            output_dir,
            model_dir_name=dir_name,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            extra_metadata={**extra_meta, "layers": layers},
            calibration_corpus=calibration_corpus,
            calibration_samples=calibration_samples,
            calibration_max_length=calibration_max_length,
            calibration_batch_size=calibration_batch_size,
            calibration_device=calibration_device,
        )
        saved_paths.append(path)

    # ---- Summary ----
    logger.info("\n" + "=" * 70)
    logger.info("  Perturbation complete — %d model(s) saved", len(saved_paths))
    logger.info("=" * 70)
    for p in saved_paths:
        logger.info("  -> %s", p)

    return saved_paths


# ======================================================================
# CLI
# ======================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Save perturbed LLM model with RACE-identified neurons modified in weights",
    )

    # Model & data
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model identifier or local path",
    )
    parser.add_argument(
        "--race-h5",
        required=True,
        help="Path to RACE results H5 file",
    )
    parser.add_argument(
        "--output-dir",
        default="result/llm/perturbed_models",
        help="Root output directory for saved models",
    )

    # Perturbation config
    parser.add_argument(
        "--concept",
        default="code_generation",
        help="Concept name in RACE results",
    )
    parser.add_argument(
        "--operation",
        default="suppress",
        choices=["suppress", "enhance", "keep_top", "mean_ablation"],
        help="Neuron modification operation",
    )
    parser.add_argument(
        "--top-k-percent",
        type=float,
        default=5.0,
        help="Percentage of top neurons to modify (ignored if --top-k is set)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        dest="top_k_count",
        metavar="N",
        help=(
            "Take the top N neurons per decoder layer and per module (attn/mlp); "
            "overrides --top-k-percent. Use 0 to select none."
        ),
    )
    parser.add_argument(
        "--metric",
        default="lcb_pos",
        choices=[
            "posterior_mean",
            "lcb",
            "lcb_pos",
            "lcb_neg",
            "empirical_mean",
            "empirical_snr",
            "activation_mean",
        ],
        help=(
            "RACE importance metric for ranking neurons. "
            "'posterior_mean' and LCB-family metrics ('lcb'/'lcb_pos'/'lcb_neg') use the NIG Bayesian posterior; "
            "'lcb_pos' keeps only posterior_mean>0 neurons, 'lcb_neg' keeps only posterior_mean<0; "
            "'empirical_mean' is a pure arithmetic mean baseline; "
            "'empirical_snr' is a frequentist mean/std baseline; "
            "'activation_mean' uses raw neuron activation magnitude."
        ),
    )
    parser.add_argument(
        "--lcb-min-score",
        type=float,
        default=None,
        metavar="T",
        help=(
            "With --metric in {lcb,lcb_pos,lcb_neg}: only neurons with score > T are ranked; "
            "top-k%% applies within that subset. Omit to rank all neurons. "
            "Use 0 for strictly positive CAM/LCB values."
        ),
    )
    parser.add_argument(
        "--enhancement-factor",
        type=float,
        default=2.0,
        help="Multiplication factor for enhance mode",
    )

    # Module / layer selection
    parser.add_argument(
        "--modules",
        nargs="+",
        default=["attn", "mlp"],
        choices=["attn", "mlp"],
        help=(
            "Modules to modify. If both attn and mlp are passed, both "
            "perturbations are applied in the same checkpoint; top-k is selected "
            "separately per layer and per module, not pooled."
        ),
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Specific layer indices to modify (default: all)",
    )
    parser.add_argument(
        "--layer-by-layer",
        type=int,
        default=None,
        metavar="N",
        help="Save one model per N-layer group (storage-intensive!)",
    )

    # Model loading
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading model from HuggingFace",
    )
    parser.add_argument(
        "--dtype",
        "--torch-dtype",
        dest="dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Data type for model loading",
    )

    # Mean-ablation calibration
    parser.add_argument(
        "--calibration-corpus",
        default="wikitext2",
        help="Corpus for computing calibration mean activations (mean_ablation only)",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=200,
        help="Number of calibration passages for mean activations (mean_ablation only)",
    )
    parser.add_argument(
        "--calibration-max-length",
        type=int,
        default=2048,
        help="Max token length per calibration passage (mean_ablation only)",
    )
    parser.add_argument(
        "--calibration-batch-size",
        type=int,
        default=4,
        help="Batch size for calibration forward passes (mean_ablation only)",
    )
    parser.add_argument(
        "--calibration-device",
        default="cuda:0",
        help="Device for calibration forward passes (mean_ablation only)",
    )

    # Reference-Set Filtering (RSF)
    parser.add_argument(
        "--general-race-h5",
        default=None,
        help=(
            "Reference RACE H5 for paper RSF. Traverse each target ranking, "
            "skip the reference Top-M, and stop after selecting k neurons."
        ),
    )
    parser.add_argument(
        "--general-concept",
        default=None,
        help="Concept name for general-knowledge H5 (required when --general-race-h5 is set)",
    )
    parser.add_argument(
        "--general-top-k",
        type=int,
        default=None,
        dest="general_top_k_count",
        metavar="M",
        help=(
            "Reference exclusion budget M per layer/module. Defaults to the "
            "target --top-k budget."
        ),
    )
    parser.add_argument(
        "--general-top-k-percent",
        type=float,
        default=None,
        metavar="P",
        help=(
            "Reference exclusion budget as top-P percent per layer/module. "
            "Defaults to the target --top-k-percent budget."
        ),
    )

    args = parser.parse_args()
    if args.top_k_count is not None and args.top_k_count < 0:
        parser.error("--top-k must be non-negative")
    if args.general_race_h5 and not args.general_concept:
        parser.error("--general-concept is required when --general-race-h5 is provided")
    if (
        args.general_top_k_count is not None or args.general_top_k_percent is not None
    ) and not args.general_race_h5:
        parser.error(
            "--general-top-k / --general-top-k-percent require " "--general-race-h5"
        )
    if args.general_top_k_count is not None and args.general_top_k_count < 0:
        parser.error("--general-top-k must be non-negative")
    if args.general_top_k_percent is not None and args.general_top_k_percent < 0:
        parser.error("--general-top-k-percent must be non-negative")
    if args.general_top_k_count is not None and args.general_top_k_percent is not None:
        parser.error(
            "--general-top-k and --general-top-k-percent are mutually exclusive"
        )

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    run_perturbation_pipeline(
        model_name=args.model,
        race_h5=args.race_h5,
        concept=args.concept,
        operation=args.operation,
        top_k_percent=args.top_k_percent,
        output_dir=args.output_dir,
        metric=args.metric,
        modules=args.modules,
        layers=args.layers,
        layer_by_layer=args.layer_by_layer,
        enhancement_factor=args.enhancement_factor,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        lcb_min_score=args.lcb_min_score,
        top_k_count=args.top_k_count,
        calibration_corpus=args.calibration_corpus,
        calibration_samples=args.calibration_samples,
        calibration_max_length=args.calibration_max_length,
        calibration_batch_size=args.calibration_batch_size,
        calibration_device=args.calibration_device,
        general_race_h5=args.general_race_h5,
        general_concept=args.general_concept,
        general_top_k_count=args.general_top_k_count,
        general_top_k_percent=args.general_top_k_percent,
    )


if __name__ == "__main__":
    main()
