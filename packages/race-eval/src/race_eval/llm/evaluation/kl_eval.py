"""KL-Divergence evaluation for RACE neuron intervention experiments.

Measures how much the next-token probability distribution shifts when
RACE-identified neurons are suppressed, providing a continuous metric
to prove domain-specificity of identified neurons.

Supports both zero ablation (``--operation suppress``) and mean ablation
(``--operation mean_ablation``).  With mean ablation, calibration mean
activations are computed on ``--calibration-corpus`` (default: wikitext2)
before running the KL forward passes.

Usage::

    python -m race_eval.llm.evaluation.kl_eval \\
        --model Qwen/Qwen3-4B-Instruct-2507 \\
        --race-h5 result/llm/online_race/.../online_race.h5 \\
        --concept code_generation \\
        --operation mean_ablation \\
        --top-k 5 --metric lcb --lcb-min-score 0 \\
        --modules mlp \\
        --corpora math_500 mbpp_plus gpqa_diamond \\
        --calibration-corpus wikitext2 --calibration-samples 200
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from race_eval.llm.evaluation.eval_corpora import load_eval_corpus
from race_eval.llm.evaluation.utils import (
    add_common_args,
    build_modifier,
    build_plan,
    load_model,
    pad_and_batch,
    validate_common_args,
)

logger = logging.getLogger(__name__)


# ======================================================================
# KL divergence computation
# ======================================================================


def compute_kl_per_batch(
    logits_baseline: torch.Tensor,
    logits_ablated: torch.Tensor,
    attention_mask: torch.Tensor,
    direction: str = "forward",
) -> Tuple[float, int]:
    """Compute token-level KL divergence for one batch.

    Args:
        logits_baseline: ``[B, seq_len, vocab]`` original logits.
        logits_ablated:  ``[B, seq_len, vocab]`` ablated logits.
        attention_mask:  ``[B, seq_len]`` mask (1 = real token).
        direction: ``"forward"`` for KL(P_orig || P_ablated),
                   ``"reverse"`` for KL(P_ablated || P_orig).

    Returns:
        ``(sum_kl, n_tokens)`` — summed KL over valid tokens and count.
    """
    log_p = F.log_softmax(logits_baseline, dim=-1)
    log_q = F.log_softmax(logits_ablated, dim=-1)

    if direction == "reverse":
        log_p, log_q = log_q, log_p

    # KL(P||Q) = sum_x P(x) * (log P(x) - log Q(x))
    p = log_p.exp()
    kl_per_token = (p * (log_p - log_q)).sum(dim=-1)  # [B, seq_len]

    masked_kl = kl_per_token * attention_mask
    return masked_kl.sum().item(), int(attention_mask.sum().item())


def compute_kl_stats_per_batch(
    logits_baseline: torch.Tensor,
    logits_ablated: torch.Tensor,
    attention_mask: torch.Tensor,
    direction: str = "forward",
) -> List[float]:
    """Compute per-token KL values for statistics (std, max).

    Returns a flat list of KL values for all valid tokens in the batch.
    """
    log_p = F.log_softmax(logits_baseline, dim=-1)
    log_q = F.log_softmax(logits_ablated, dim=-1)

    if direction == "reverse":
        log_p, log_q = log_q, log_p

    p = log_p.exp()
    kl_per_token = (p * (log_p - log_q)).sum(dim=-1)  # [B, seq_len]

    mask_bool = attention_mask.bool()
    return kl_per_token[mask_bool].tolist()


def _make_empty_kl_result() -> Dict:
    return {
        "mean_kl": None,
        "std_kl": None,
        "max_kl": None,
        "num_samples": 0,
    }


def _print_and_write_summary(
    args: argparse.Namespace,
    results: Dict,
    elapsed: float,
    *,
    direct_comparison: bool,
    direction: str,
    plan=None,
) -> None:
    print("\n" + "=" * 70)
    print("  KL Divergence Evaluation Summary")
    print("=" * 70)
    if direct_comparison:
        print("  Mode:      direct model comparison")
        print(f"  Base:      {args.baseline_model}")
        print(f"  Pert:      {args.perturbed_model}")
    else:
        print(f"  Model:     {args.model}")
        print(f"  Concept:   {args.concept} ({args.operation})")
        if args.top_k_count is not None:
            print(f"  Top-k:     {args.top_k_count} per layer per module")
        else:
            print(f"  Top-k%%:    {args.top_k_percent}")
        print(f"  Modules:   {args.modules}")
        print(f"  Neurons:   {plan.total_neurons}")
    if direct_comparison:
        left = "baseline" if direction == "forward" else "perturbed"
        right = "perturbed" if direction == "forward" else "baseline"
    else:
        left = "orig" if direction == "forward" else "ablated"
        right = "ablated" if direction == "forward" else "orig"
    print(f"  Direction: KL(P_{left} || P_{right})")
    print("-" * 70)
    print(
        f"  {'Corpus':<20} {'Mean KL':>10} {'Std KL':>10} {'Max KL':>10} {'Samples':>8}"
    )
    print("-" * 70)
    for name, r in results["results"].items():
        mk = f"{r['mean_kl']:.6f}" if r["mean_kl"] is not None else "N/A"
        sk = f"{r['std_kl']:.6f}" if r["std_kl"] is not None else "N/A"
        xk = f"{r['max_kl']:.6f}" if r["max_kl"] is not None else "N/A"
        print(f"  {name:<20} {mk:>10} {sk:>10} {xk:>10} {r['num_samples']:>8}")
    print("=" * 70)
    print(f"  Elapsed: {elapsed:.1f}s")
    print("=" * 70 + "\n")

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, f"kl_eval_{results['timestamp']}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Results written to %s", json_path)


@torch.inference_mode()
def run_direct_kl_evaluation(args: argparse.Namespace, t_start: float) -> Dict:
    """Run KL with only one model resident on GPU at a time."""
    direction = getattr(args, "kl_direction", "forward")
    logger.info(
        "Running direct KL comparison: baseline=%s perturbed=%s",
        args.baseline_model,
        args.perturbed_model,
    )

    corpus_results: Dict[str, Dict] = {}

    for corpus_name in args.corpora:
        logger.info("=" * 60)
        logger.info("Evaluating corpus: %s", corpus_name)
        logger.info("=" * 60)

        baseline_model, tokenizer = load_model(
            args.baseline_model,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
        baseline_device = next(baseline_model.parameters()).device

        input_ids_list = load_eval_corpus(
            corpus_name,
            tokenizer,
            model=baseline_model,
            model_name=args.baseline_model,
            max_samples=args.max_samples,
            max_length=args.max_length,
            generation_cache_dir=args.generation_cache_dir,
            generation_batch_size=args.generation_batch_size,
            results_dir=args.results_dir,
            prompt_only=args.prompt_only,
        )
        if not input_ids_list:
            logger.warning("No valid samples for %s — skipping.", corpus_name)
            corpus_results[corpus_name] = _make_empty_kl_result()
            del baseline_model
            torch.cuda.empty_cache()
            continue

        batches = pad_and_batch(input_ids_list, tokenizer.pad_token_id, args.batch_size)

        baseline_logits: List[torch.Tensor] = []
        logger.info("  Baseline forward pass (%d batches) ...", len(batches))
        for batch_idx, (input_ids, attention_mask) in enumerate(batches):
            input_ids_dev = input_ids.to(baseline_device)
            attn_mask_dev = attention_mask.to(baseline_device)
            out_baseline = baseline_model(
                input_ids=input_ids_dev,
                attention_mask=attn_mask_dev,
            )
            baseline_logits.append(out_baseline.logits.cpu())
            if (batch_idx + 1) % 10 == 0 or batch_idx == len(batches) - 1:
                logger.info(
                    "  Baseline batch %d/%d done",
                    batch_idx + 1,
                    len(batches),
                )

        del baseline_model
        torch.cuda.empty_cache()

        perturbed_model, _ = load_model(
            args.perturbed_model,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
        perturbed_device = next(perturbed_model.parameters()).device

        total_kl = 0.0
        total_tokens = 0
        all_kl_values: List[float] = []
        logger.info("  Perturbed forward pass (%d batches) ...", len(batches))
        for batch_idx, ((input_ids, attention_mask), logits_bl) in enumerate(
            zip(batches, baseline_logits)
        ):
            input_ids_dev = input_ids.to(perturbed_device)
            attn_mask_dev = attention_mask.to(perturbed_device)
            out_perturbed = perturbed_model(
                input_ids=input_ids_dev,
                attention_mask=attn_mask_dev,
            )
            logits_ab = out_perturbed.logits.cpu()

            sum_kl, n_tok = compute_kl_per_batch(
                logits_bl,
                logits_ab,
                attention_mask,
                direction,
            )
            total_kl += sum_kl
            total_tokens += n_tok
            all_kl_values.extend(
                compute_kl_stats_per_batch(
                    logits_bl,
                    logits_ab,
                    attention_mask,
                    direction,
                )
            )
            del logits_ab

            if (batch_idx + 1) % 10 == 0 or batch_idx == len(batches) - 1:
                logger.info(
                    "  Batch %d/%d done (running mean KL: %.6f)",
                    batch_idx + 1,
                    len(batches),
                    total_kl / max(total_tokens, 1),
                )

        del perturbed_model, baseline_logits
        torch.cuda.empty_cache()

        mean_kl = total_kl / max(total_tokens, 1)
        if all_kl_values:
            kl_tensor = torch.tensor(all_kl_values)
            std_kl = float(kl_tensor.std().item())
            max_kl = float(kl_tensor.max().item())
        else:
            std_kl = 0.0
            max_kl = 0.0

        logger.info("  Mean KL: %.6f | Std: %.6f | Max: %.6f", mean_kl, std_kl, max_kl)
        corpus_results[corpus_name] = {
            "mean_kl": round(mean_kl, 6),
            "std_kl": round(std_kl, 6),
            "max_kl": round(max_kl, 6),
            "num_samples": len(input_ids_list),
            "num_tokens": total_tokens,
        }

    elapsed = time.perf_counter() - t_start
    results = {
        "evaluation_mode": "direct_model_comparison",
        "model": args.baseline_model,
        "baseline_model": args.baseline_model,
        "perturbed_model": args.perturbed_model,
        "concept": None,
        "operation": None,
        "top_k_percent": args.top_k_percent,
        "top_k_count": args.top_k_count,
        "metric": args.metric,
        "modules": args.modules,
        "lcb_min_score": args.lcb_min_score,
        "kl_direction": direction,
        "prompt_only": args.prompt_only,
        "total_neurons_modified": None,
        "results": corpus_results,
        "elapsed_s": round(elapsed, 2),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }
    _print_and_write_summary(
        args,
        results,
        elapsed,
        direct_comparison=True,
        direction=direction,
    )
    return results


# ======================================================================
# Main pipeline
# ======================================================================


@torch.inference_mode()
def run_kl_evaluation(args: argparse.Namespace) -> Dict:
    """Run the full KL divergence evaluation pipeline.

    For each corpus, processes batches one at a time to minimize GPU memory:
    1. Baseline forward pass (no hooks) -> logits_baseline
    2. Ablated forward pass (with hooks) -> logits_ablated
    3. Compute KL on CPU, release both logit tensors

    Returns:
        Results dictionary.
    """
    t_start = time.perf_counter()
    direct_comparison = validate_common_args(args)

    if direct_comparison:
        return run_direct_kl_evaluation(args, t_start)

    # 1. Load model
    baseline_model, tokenizer = load_model(
        args.model,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )
    baseline_device = next(baseline_model.parameters()).device
    perturbed_model = baseline_model
    perturbed_device = baseline_device
    generation_model = baseline_model
    generation_model_name = args.model

    # 2. Build ablation plan
    plan = build_plan(args)
    logger.info(
        "Ablation plan: concept=%s, operation=%s, total_neurons=%d, modules=%s",
        args.concept,
        args.operation,
        plan.total_neurons,
        args.modules,
    )

    # 2b. Build modifier (computes mean activations when operation=mean_ablation)
    modifier = build_modifier(baseline_model, tokenizer, plan, args)

    direction = getattr(args, "kl_direction", "forward")

    # 3. Evaluate each corpus
    corpus_results: Dict[str, Dict] = {}

    for corpus_name in args.corpora:
        logger.info("=" * 60)
        logger.info("Evaluating corpus: %s", corpus_name)
        logger.info("=" * 60)

        input_ids_list = load_eval_corpus(
            corpus_name,
            tokenizer,
            model=generation_model,
            model_name=generation_model_name,
            max_samples=args.max_samples,
            max_length=args.max_length,
            generation_cache_dir=args.generation_cache_dir,
            generation_batch_size=args.generation_batch_size,
            results_dir=args.results_dir,
            prompt_only=args.prompt_only,
        )
        if not input_ids_list:
            logger.warning("No valid samples for %s — skipping.", corpus_name)
            corpus_results[corpus_name] = {
                "mean_kl": None,
                "std_kl": None,
                "max_kl": None,
                "num_samples": 0,
            }
            continue

        batches = pad_and_batch(input_ids_list, tokenizer.pad_token_id, args.batch_size)

        # Process batch-by-batch: baseline + ablated in lock-step
        total_kl = 0.0
        total_tokens = 0
        all_kl_values: List[float] = []

        for batch_idx, (input_ids, attention_mask) in enumerate(batches):
            attn_mask_dev = attention_mask.to(baseline_device)

            # Baseline forward
            input_ids_dev = input_ids.to(baseline_device)
            out_baseline = baseline_model(
                input_ids=input_ids_dev,
                attention_mask=attn_mask_dev,
            )
            logits_bl = out_baseline.logits.cpu()

            if direct_comparison:
                input_ids_pert = input_ids.to(perturbed_device)
                attn_mask_pert = attention_mask.to(perturbed_device)
                out_ablated = perturbed_model(
                    input_ids=input_ids_pert,
                    attention_mask=attn_mask_pert,
                )
                logits_ab = out_ablated.logits.cpu()
            else:
                # Ablated forward
                modifier.apply()
                try:
                    out_ablated = perturbed_model(
                        input_ids=input_ids_dev,
                        attention_mask=attn_mask_dev,
                    )
                    logits_ab = out_ablated.logits.cpu()
                finally:
                    modifier.remove()

            # Compute KL on CPU
            sum_kl, n_tok = compute_kl_per_batch(
                logits_bl,
                logits_ab,
                attention_mask,
                direction,
            )
            total_kl += sum_kl
            total_tokens += n_tok

            batch_kl_values = compute_kl_stats_per_batch(
                logits_bl,
                logits_ab,
                attention_mask,
                direction,
            )
            all_kl_values.extend(batch_kl_values)

            del logits_bl, logits_ab

            if (batch_idx + 1) % 10 == 0 or batch_idx == len(batches) - 1:
                logger.info(
                    "  Batch %d/%d done (running mean KL: %.6f)",
                    batch_idx + 1,
                    len(batches),
                    total_kl / max(total_tokens, 1),
                )

        mean_kl = total_kl / max(total_tokens, 1)
        if all_kl_values:
            kl_tensor = torch.tensor(all_kl_values)
            std_kl = float(kl_tensor.std().item())
            max_kl = float(kl_tensor.max().item())
        else:
            std_kl = 0.0
            max_kl = 0.0

        logger.info("  Mean KL: %.6f | Std: %.6f | Max: %.6f", mean_kl, std_kl, max_kl)

        corpus_results[corpus_name] = {
            "mean_kl": round(mean_kl, 6),
            "std_kl": round(std_kl, 6),
            "max_kl": round(max_kl, 6),
            "num_samples": len(input_ids_list),
            "num_tokens": total_tokens,
        }

    elapsed = time.perf_counter() - t_start

    # 4. Assemble final results
    results = {
        "evaluation_mode": (
            "direct_model_comparison" if direct_comparison else "hook_ablation"
        ),
        "model": args.model if not direct_comparison else args.baseline_model,
        "baseline_model": args.baseline_model if direct_comparison else args.model,
        "perturbed_model": args.perturbed_model if direct_comparison else None,
        "concept": None if direct_comparison else args.concept,
        "operation": None if direct_comparison else args.operation,
        "top_k_percent": args.top_k_percent,
        "top_k_count": args.top_k_count,
        "metric": args.metric,
        "modules": args.modules,
        "lcb_min_score": args.lcb_min_score,
        "kl_direction": direction,
        "prompt_only": args.prompt_only,
        "total_neurons_modified": None if direct_comparison else plan.total_neurons,
        "results": corpus_results,
        "elapsed_s": round(elapsed, 2),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    # Print summary table
    print("\n" + "=" * 70)
    print("  KL Divergence Evaluation Summary")
    print("=" * 70)
    if direct_comparison:
        print("  Mode:      direct model comparison")
        print(f"  Base:      {args.baseline_model}")
        print(f"  Pert:      {args.perturbed_model}")
    else:
        print(f"  Model:     {args.model}")
        print(f"  Concept:   {args.concept} ({args.operation})")
        if args.top_k_count is not None:
            print(f"  Top-k:     {args.top_k_count} per layer per module")
        else:
            print(f"  Top-k%%:    {args.top_k_percent}")
        print(f"  Modules:   {args.modules}")
        print(f"  Neurons:   {plan.total_neurons}")
    if direct_comparison:
        left = "baseline" if direction == "forward" else "perturbed"
        right = "perturbed" if direction == "forward" else "baseline"
    else:
        left = "orig" if direction == "forward" else "ablated"
        right = "ablated" if direction == "forward" else "orig"
    print(f"  Direction: KL(P_{left} || P_{right})")
    print("-" * 70)
    print(
        f"  {'Corpus':<20} {'Mean KL':>10} {'Std KL':>10} {'Max KL':>10} {'Samples':>8}"
    )
    print("-" * 70)
    for name, r in corpus_results.items():
        mk = f"{r['mean_kl']:.6f}" if r["mean_kl"] is not None else "N/A"
        sk = f"{r['std_kl']:.6f}" if r["std_kl"] is not None else "N/A"
        xk = f"{r['max_kl']:.6f}" if r["max_kl"] is not None else "N/A"
        print(f"  {name:<20} {mk:>10} {sk:>10} {xk:>10} {r['num_samples']:>8}")
    print("=" * 70)
    print(f"  Elapsed: {elapsed:.1f}s")
    print("=" * 70 + "\n")

    # Write JSON
    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, f"kl_eval_{results['timestamp']}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Results written to %s", json_path)

    return results


# ======================================================================
# CLI
# ======================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate KL divergence after RACE neuron suppression",
    )
    add_common_args(parser)
    parser.add_argument(
        "--kl-direction",
        default="forward",
        choices=["forward", "reverse"],
        help="KL direction: forward = KL(P_orig || P_ablated), reverse = KL(P_ablated || P_orig)",
    )
    args = parser.parse_args()

    validate_common_args(args, parser)
    if args.top_k_count is not None and args.top_k_count < 0:
        parser.error("--top-k must be non-negative")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    run_kl_evaluation(args)


if __name__ == "__main__":
    main()
