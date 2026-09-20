"""Shared utilities for PPL and KL divergence evaluation scripts.

Provides:
- ``add_common_args``  — shared argparse definitions (mirrors save_perturbed_model.py)
- ``load_model``       — model + tokenizer loading
- ``build_plan``       — AblationPlan construction from CLI args
- ``collect_logits``   — teacher-forcing forward pass
- ``pad_and_batch``    — collate variable-length input_ids into batches
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import tempfile
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from race_eval.ablation import AblationPlan, OperationMode, create_ablation_plan

logger = logging.getLogger(__name__)


# ======================================================================
# Argparse helpers
# ======================================================================


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add CLI arguments shared by ppl_eval and kl_eval.

    Neuron-selection args mirror ``save_perturbed_model.py`` exactly.
    """
    # --- Model ---
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "HuggingFace model identifier or local path for hook-based "
            "evaluation. Required unless --baseline-model/--perturbed-model "
            "are provided."
        ),
    )
    parser.add_argument(
        "--baseline-model",
        default=None,
        help=(
            "Baseline HuggingFace model identifier or local path for direct "
            "two-model comparison. Must be used with --perturbed-model."
        ),
    )
    parser.add_argument(
        "--perturbed-model",
        default=None,
        help=(
            "Already-perturbed HuggingFace model identifier or local path for "
            "direct two-model comparison. Must be used with --baseline-model."
        ),
    )
    parser.add_argument(
        "--dtype",
        "--torch-dtype",
        dest="dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Data type for model loading",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading model",
    )

    # --- RACE / neuron selection ---
    parser.add_argument(
        "--race-h5",
        default=None,
        help=(
            "Path to RACE results H5 file. Required for hook-based "
            "evaluation; not used for direct two-model comparison."
        ),
    )
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
        help="Take the top N neurons per decoder layer and per module; overrides --top-k-percent",
    )
    parser.add_argument(
        "--metric",
        default="posterior_mean",
        choices=[
            "posterior_mean",
            "lcb",
            "lcb_pos",
            "lcb_neg",
            "empirical_mean",
            "empirical_snr",
            "activation_mean",
        ],
        help="RACE importance metric for ranking neurons",
    )
    parser.add_argument(
        "--lcb-min-score",
        type=float,
        default=None,
        metavar="T",
        help="With --metric in {lcb,lcb_pos,lcb_neg}: only neurons with score > T are ranked",
    )
    parser.add_argument(
        "--modules",
        nargs="+",
        default=["attn", "mlp"],
        choices=["attn", "mlp"],
        help="Modules to modify",
    )
    parser.add_argument(
        "--enhancement-factor",
        type=float,
        default=2.0,
        help="Multiplication factor for enhance mode",
    )

    # --- Evaluation ---
    parser.add_argument(
        "--corpora",
        nargs="+",
        default=["math_500", "mbpp_plus", "gpqa_diamond"],
        help="Evaluation corpora to test",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Maximum samples per corpus",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum token length per sample",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for forward passes",
    )
    parser.add_argument(
        "--output-dir",
        default="result/llm/eval",
        help="Output directory for results JSON",
    )
    parser.add_argument(
        "--baseline-cache-dir",
        default="result/llm/baseline_eval_cache",
        help=(
            "Directory for direct-eval baseline metric caches. PPL stores "
            "small scalar results here to avoid recomputing the baseline."
        ),
    )
    parser.add_argument(
        "--no-baseline-cache",
        action="store_true",
        help="Disable direct-eval baseline metric cache.",
    )

    # --- Generation cache ---
    parser.add_argument(
        "--results-dir",
        default=None,
        metavar="DIR",
        help="Directory of pre-existing inference JSONL results "
        "(e.g. results/Qwen3-4B-Instruct-2507O). "
        "When set, skips live inference and generation-cache lookup. "
        "Files are matched by corpus name pattern "
        "(math_500_*.jsonl / mbpp_plus_*.jsonl / gpqa_diamond_*.jsonl).",
    )
    parser.add_argument(
        "--generation-cache-dir",
        default="result/llm/generation_cache",
        help="Directory for caching model generation outputs. "
        "Cached files are keyed by (model, corpus) and reused across runs.",
    )
    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=4,
        help="Batch size for generation (only used when cache is missing)",
    )

    # --- Mean ablation calibration ---
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
        "--prompt-only",
        action="store_true",
        help="Evaluate on input prompts only (no model generation); skips generation stage",
    )


def is_direct_model_comparison(args: argparse.Namespace) -> bool:
    """Return True when CLI args request baseline-vs-perturbed model eval."""
    return bool(
        getattr(args, "baseline_model", None) or getattr(args, "perturbed_model", None)
    )


def validate_common_args(
    args: argparse.Namespace,
    parser: Optional[argparse.ArgumentParser] = None,
) -> bool:
    """Validate common evaluation args.

    Returns:
        True for direct two-model comparison, or False for hook-based
        evaluation.
    """

    def fail(message: str) -> None:
        if parser is not None:
            parser.error(message)
        raise ValueError(message)

    direct = is_direct_model_comparison(args)
    if direct:
        if not getattr(args, "baseline_model", None) or not getattr(
            args, "perturbed_model", None
        ):
            fail("--baseline-model and --perturbed-model must be provided together")
        if getattr(args, "model", None):
            fail("--model cannot be combined with --baseline-model/--perturbed-model")
        return True

    if not getattr(args, "model", None):
        fail(
            "--model is required unless --baseline-model/--perturbed-model are provided"
        )
    if not getattr(args, "race_h5", None):
        fail(
            "--race-h5 is required unless --baseline-model/--perturbed-model are provided"
        )
    return False


# ======================================================================
# Model loading
# ======================================================================

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def load_model(
    model_name: str,
    dtype: str = "bfloat16",
    trust_remote_code: bool = False,
    device: str = "cuda",
) -> Tuple[nn.Module, object]:
    """Load model and tokenizer for evaluation.

    Returns:
        ``(model, tokenizer)`` tuple.
    """
    model_dtype = _DTYPE_MAP.get(dtype, torch.bfloat16)

    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    logger.info("Loading model: %s (dtype=%s)", model_name, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=model_dtype,
        device_map=device,
        trust_remote_code=trust_remote_code,
        attn_implementation="flash_attention_2",
    )
    model.eval()
    logger.info("Model loaded on %s", device)
    return model, tokenizer


# ======================================================================
# Ablation plan
# ======================================================================


def build_plan(args: argparse.Namespace) -> AblationPlan:
    """Create an ``AblationPlan`` from parsed CLI arguments."""
    return create_ablation_plan(
        h5_path=args.race_h5,
        concept_name=args.concept,
        operation_mode=OperationMode(args.operation),
        top_k_percent=args.top_k_percent,
        metric=args.metric,
        enhancement_factor=args.enhancement_factor,
        modules=args.modules,
        lcb_min_score=args.lcb_min_score,
        top_k_count=args.top_k_count,
    )


def compute_mean_activations_from_args(
    model: nn.Module,
    tokenizer,
    args: argparse.Namespace,
) -> Optional[Dict[str, torch.Tensor]]:
    """Compute per-neuron mean activations for mean-ablation mode.

    Returns ``None`` when ``args.operation != "mean_ablation"``.  Otherwise
    runs calibration on the corpus specified in *args* and returns a
    CPU-resident ``Dict[str, Tensor]`` (device-agnostic; individual hook
    instances move slices to their own device lazily).

    Keeping the result on CPU allows the same dict to be shared across
    multiple :class:`~race_eval.llm.interventions.ablation.LLMNeuronModifier`
    instances attached to replicas on different GPUs.
    """
    if args.operation != "mean_ablation":
        return None

    from race_eval.llm.evaluation.mean_activations import compute_mean_activations

    calibration_corpus = getattr(args, "calibration_corpus", "wikitext2")
    calibration_samples = getattr(args, "calibration_samples", 200)
    logger.info(
        "Computing calibration means on %s (%d samples) ...",
        calibration_corpus,
        calibration_samples,
    )
    mean_acts = compute_mean_activations(
        model,
        tokenizer,
        calibration_corpus=calibration_corpus,
        max_samples=calibration_samples,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    # Move to CPU so the dict can be shared across replicas on different GPUs.
    return {k: v.cpu() for k, v in mean_acts.items()}


def build_modifier(
    model: nn.Module,
    tokenizer,
    plan: AblationPlan,
    args: argparse.Namespace,
    mean_activations: Optional[Dict[str, torch.Tensor]] = None,
):
    """Construct a :class:`LLMNeuronModifier`, computing mean activations when required.

    When ``args.operation == "mean_ablation"`` and *mean_activations* is
    ``None``, the function calls :func:`compute_mean_activations_from_args`
    to compute calibration statistics.  Pass a pre-computed *mean_activations*
    dict (e.g. shared across GPU replicas) to skip the calibration step.

    Args:
        model: Loaded causal LM (eval mode, on device).
        tokenizer: Corresponding tokenizer.
        plan: Ablation plan from :func:`build_plan`.
        args: Parsed CLI namespace — must have ``operation``,
              ``calibration_corpus``, ``calibration_samples``,
              ``max_length``, and ``batch_size`` attributes.
        mean_activations: Pre-computed mean activations to reuse (optional).
            When ``None`` and ``operation == "mean_ablation"`` the activations
            are computed automatically.

    Returns:
        Configured :class:`~race_eval.llm.interventions.ablation.LLMNeuronModifier`.
    """
    from race_eval.llm.interventions.ablation import LLMNeuronModifier

    if mean_activations is None:
        mean_activations = compute_mean_activations_from_args(model, tokenizer, args)

    return LLMNeuronModifier(model, plan, mean_activations=mean_activations)


# ======================================================================
# Batching
# ======================================================================


def pad_and_batch(
    input_ids_list: List[torch.Tensor],
    pad_token_id: int,
    batch_size: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Collate variable-length 1-D tensors into left-padded batches.

    Returns:
        List of ``(input_ids, attention_mask)`` tuples, each with shape
        ``[B, max_len_in_batch]``.
    """
    batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for start in range(0, len(input_ids_list), batch_size):
        chunk = input_ids_list[start : start + batch_size]
        max_len = max(t.shape[0] for t in chunk)

        padded_ids = []
        masks = []
        for t in chunk:
            pad_len = max_len - t.shape[0]
            padded = F.pad(t, (pad_len, 0), value=pad_token_id)
            mask = F.pad(torch.ones_like(t), (pad_len, 0), value=0)
            padded_ids.append(padded)
            masks.append(mask)

        batches.append((torch.stack(padded_ids), torch.stack(masks)))
    return batches


# ======================================================================
# Teacher-forcing logit collection
# ======================================================================


@torch.inference_mode()
def collect_logits_batched(
    model: nn.Module,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Run teacher-forcing forward on pre-batched inputs.

    Args:
        model: Language model.
        batches: Output of :func:`pad_and_batch`.
        device: Device to run forward passes on.

    Returns:
        List of ``(logits_cpu, input_ids_cpu, attention_mask_cpu)`` tuples.
        Logits are moved to CPU immediately to save GPU memory.
    """
    results: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for input_ids, attention_mask in batches:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.cpu()
        results.append((logits, input_ids.cpu(), attention_mask.cpu()))

    return results


# ======================================================================
# Baseline metric cache for direct model comparison
# ======================================================================


def _safe_cache_component(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)


def _hash_batches(batches: List[Tuple[torch.Tensor, torch.Tensor]]) -> str:
    digest = hashlib.sha256()
    for input_ids, attention_mask in batches:
        for tensor in (input_ids, attention_mask):
            tensor_cpu = tensor.detach().cpu().contiguous()
            digest.update(str(tuple(tensor_cpu.shape)).encode("utf-8"))
            digest.update(str(tensor_cpu.dtype).encode("utf-8"))
            digest.update(tensor_cpu.numpy().tobytes())
    return digest.hexdigest()


def baseline_metric_cache_path(
    cache_dir: str,
    *,
    metric: str,
    model_name: str,
    corpus_name: str,
    dtype: str,
    batch_size: int,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
) -> str:
    """Return the cache path for a baseline metric on an exact tokenized corpus."""
    batch_hash = _hash_batches(batches)
    model_hash = hashlib.sha256(model_name.encode("utf-8")).hexdigest()[:12]
    safe_model = _safe_cache_component(
        os.path.basename(model_name.rstrip("/")) or "model"
    )
    safe_corpus = _safe_cache_component(corpus_name)
    safe_metric = _safe_cache_component(metric)
    filename = (
        f"{safe_model}_{model_hash}__{safe_corpus}__{safe_metric}__"
        f"bs{batch_size}__{dtype}__{batch_hash[:16]}.json"
    )
    return os.path.join(cache_dir, filename)


def load_baseline_metric_cache(
    cache_path: str,
) -> Optional[Dict]:
    """Load a cached baseline metric payload if present and valid."""
    if not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:  # pragma: no cover - defensive against partial files
        logger.warning("Failed to load baseline cache %s: %s", cache_path, exc)
        return None

    if not isinstance(payload, dict):
        logger.warning("Ignoring invalid baseline cache payload: %s", cache_path)
        return None

    logger.info("Loaded baseline metric cache: %s", cache_path)
    return payload


def save_baseline_metric_cache(
    cache_path: str,
    payload: Dict,
) -> None:
    """Atomically save a small baseline metric payload."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".tmp_baseline_", suffix=".json", dir=os.path.dirname(cache_path)
    )
    os.close(fd)
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, cache_path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    logger.info("Saved baseline metric cache: %s", cache_path)
