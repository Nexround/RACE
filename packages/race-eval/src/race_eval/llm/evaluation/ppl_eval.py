"""Perplexity (PPL) evaluation for RACE neuron intervention experiments.

Measures how much perplexity changes on different domain corpora when
RACE-identified neurons are suppressed, providing a continuous metric
that complements discrete benchmark accuracy.

Supports both zero ablation (``--operation suppress``) and mean ablation
(``--operation mean_ablation``).  With mean ablation, calibration mean
activations are computed on ``--calibration-corpus`` (default: wikitext2)
before running the PPL forward passes.

Usage::

    python -m race_eval.llm.evaluation.ppl_eval \\
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
    baseline_metric_cache_path,
    build_modifier,
    build_plan,
    collect_logits_batched,
    load_baseline_metric_cache,
    validate_common_args,
    load_model,
    pad_and_batch,
    save_baseline_metric_cache,
)

logger = logging.getLogger(__name__)


# ======================================================================
# PPL computation
# ======================================================================


def compute_perplexity(
    batch_results: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> float:
    """Compute corpus-level perplexity from batched logits.

    Uses the standard shifted-label cross-entropy:
    ``PPL = exp(mean NLL)`` where NLL is computed only over non-padding
    positions.

    Args:
        batch_results: List of ``(logits, input_ids, attention_mask)``
            tuples (all on CPU).

    Returns:
        Corpus-level perplexity as a float.
    """
    total_loss = 0.0
    total_tokens = 0

    for logits, input_ids, attention_mask in batch_results:
        # Shift: logits[:, :-1] predicts input_ids[:, 1:]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()

        # Per-token cross-entropy (no reduction)
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        )
        loss = loss.view(shift_labels.shape)

        # Mask out padding positions
        masked_loss = (loss * shift_mask).sum()
        n_tokens = shift_mask.sum()

        total_loss += masked_loss.item()
        total_tokens += n_tokens.item()

    if total_tokens == 0:
        return float("inf")

    mean_nll = total_loss / total_tokens
    return float(torch.exp(torch.tensor(mean_nll)).item())


def _make_empty_corpus_result() -> Dict:
    return {
        "ppl_baseline": None,
        "ppl_ablated": None,
        "ppl_perturbed": None,
        "delta_pct": None,
        "num_samples": 0,
    }


def _print_and_write_summary(
    args: argparse.Namespace,
    results: Dict,
    elapsed: float,
    *,
    direct_comparison: bool,
    plan=None,
) -> None:
    print("\n" + "=" * 70)
    print("  PPL Evaluation Summary")
    print("=" * 70)
    if direct_comparison:
        print("  Mode:    direct model comparison")
        print(f"  Base:    {args.baseline_model}")
        print(f"  Pert:    {args.perturbed_model}")
    else:
        print(f"  Model:   {args.model}")
        print(f"  Concept: {args.concept} ({args.operation})")
        if args.top_k_count is not None:
            print(f"  Top-k:   {args.top_k_count} per layer per module")
        else:
            print(f"  Top-k%%:  {args.top_k_percent}")
        print(f"  Modules: {args.modules}")
        print(f"  Neurons: {plan.total_neurons}")
    print("-" * 70)
    comparison_label = "Perturbed" if direct_comparison else "Ablated"
    print(
        f"  {'Corpus':<20} {'Baseline':>10} "
        f"{comparison_label:>10} {'Delta%':>10} {'Samples':>8}"
    )
    print("-" * 70)
    for name, r in results["results"].items():
        bl = f"{r['ppl_baseline']:.2f}" if r["ppl_baseline"] is not None else "N/A"
        ab = f"{r['ppl_ablated']:.2f}" if r["ppl_ablated"] is not None else "N/A"
        dp = f"{r['delta_pct']:+.2f}%" if r["delta_pct"] is not None else "N/A"
        print(f"  {name:<20} {bl:>10} {ab:>10} {dp:>10} {r['num_samples']:>8}")
    print("=" * 70)
    print(f"  Elapsed: {elapsed:.1f}s")
    print("=" * 70 + "\n")

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, f"ppl_eval_{results['timestamp']}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Results written to %s", json_path)


def run_direct_ppl_evaluation(args: argparse.Namespace, t_start: float) -> Dict:
    """Run PPL by loading baseline and perturbed models sequentially.

    A 4B baseline plus a 4B perturbed checkpoint do not fit comfortably on a
    single 24GB GPU once lm_head logits are materialized.  This path keeps only
    one model on GPU at a time.
    """
    logger.info(
        "Running direct PPL comparison: baseline=%s perturbed=%s",
        args.baseline_model,
        args.perturbed_model,
    )

    baseline_model, tokenizer = load_model(
        args.baseline_model,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )
    baseline_device = next(baseline_model.parameters()).device

    corpus_results: Dict[str, Dict] = {}
    corpus_batches: Dict[str, Tuple[List[Tuple[torch.Tensor, torch.Tensor]], int]] = {}

    for corpus_name in args.corpora:
        logger.info("=" * 60)
        logger.info("Evaluating corpus: %s", corpus_name)
        logger.info("=" * 60)

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
            corpus_results[corpus_name] = _make_empty_corpus_result()
            continue

        batches = pad_and_batch(input_ids_list, tokenizer.pad_token_id, args.batch_size)
        corpus_batches[corpus_name] = (batches, len(input_ids_list))

        ppl_baseline = None
        if not args.no_baseline_cache:
            cache_path = baseline_metric_cache_path(
                args.baseline_cache_dir,
                metric="ppl",
                model_name=args.baseline_model,
                corpus_name=corpus_name,
                dtype=args.dtype,
                batch_size=args.batch_size,
                batches=batches,
            )
            baseline_cache = load_baseline_metric_cache(cache_path)
            if baseline_cache is not None:
                ppl_baseline = baseline_cache.get("ppl_baseline")

        if ppl_baseline is None:
            logger.info("  Baseline forward pass (%d batches) ...", len(batches))
            baseline_results = collect_logits_batched(
                baseline_model, batches, baseline_device
            )
            ppl_baseline = compute_perplexity(baseline_results)
            del baseline_results
            if not args.no_baseline_cache:
                save_baseline_metric_cache(
                    cache_path,
                    {
                        "metric": "ppl",
                        "model": args.baseline_model,
                        "corpus": corpus_name,
                        "dtype": args.dtype,
                        "batch_size": args.batch_size,
                        "num_samples": len(input_ids_list),
                        "ppl_baseline": round(ppl_baseline, 4),
                    },
                )
        else:
            logger.info("  Reusing cached baseline PPL: %.4f", ppl_baseline)

        corpus_results[corpus_name] = {
            "ppl_baseline": round(ppl_baseline, 4),
            "ppl_ablated": None,
            "ppl_perturbed": None,
            "delta_pct": None,
            "num_samples": len(input_ids_list),
        }
        logger.info("  PPL baseline: %.4f", ppl_baseline)

    del baseline_model
    torch.cuda.empty_cache()

    perturbed_model, _ = load_model(
        args.perturbed_model,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )
    perturbed_device = next(perturbed_model.parameters()).device

    for corpus_name, (batches, num_samples) in corpus_batches.items():
        logger.info(
            "  Perturbed forward pass for %s (%d batches) ...",
            corpus_name,
            len(batches),
        )
        perturbed_results = collect_logits_batched(
            perturbed_model,
            batches,
            perturbed_device,
        )
        ppl_perturbed = compute_perplexity(perturbed_results)
        del perturbed_results

        ppl_baseline = corpus_results[corpus_name]["ppl_baseline"]
        if ppl_baseline > 0 and ppl_baseline != float("inf"):
            delta_pct = (ppl_perturbed - ppl_baseline) / ppl_baseline * 100.0
        else:
            delta_pct = float("nan")

        corpus_results[corpus_name] = {
            "ppl_baseline": ppl_baseline,
            "ppl_ablated": round(ppl_perturbed, 4),
            "ppl_perturbed": round(ppl_perturbed, 4),
            "delta_pct": round(delta_pct, 2),
            "num_samples": num_samples,
        }
        logger.info("  PPL perturbed: %.4f", ppl_perturbed)
        logger.info("  Delta:        %.2f%%", delta_pct)

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
    )
    return results


# ======================================================================
# Main pipeline
# ======================================================================


def run_ppl_evaluation(args: argparse.Namespace) -> Dict:
    """Run the full PPL evaluation pipeline.

    1. Load model & tokenizer
    2. Build ablation plan from H5
    3. For each corpus: baseline forward -> ablated forward -> compute PPL
    4. Write JSON results

    Returns:
        Results dictionary.
    """
    t_start = time.perf_counter()
    direct_comparison = validate_common_args(args)

    if direct_comparison:
        return run_direct_ppl_evaluation(args, t_start)

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
    if plan.total_neurons == 0:
        logger.warning("Ablation plan has 0 neurons — ablated PPL will equal baseline.")

    # 2b. Build modifier (computes mean activations when operation=mean_ablation)
    modifier = build_modifier(baseline_model, tokenizer, plan, args)

    # 3. Evaluate each corpus
    corpus_results: Dict[str, Dict] = {}

    for corpus_name in args.corpora:
        logger.info("=" * 60)
        logger.info("Evaluating corpus: %s", corpus_name)
        logger.info("=" * 60)

        # Generate (or load from cache/results-dir) & tokenize
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
                "ppl_baseline": None,
                "ppl_ablated": None,
                "ppl_perturbed": None,
                "delta_pct": None,
                "num_samples": 0,
            }
            continue

        batches = pad_and_batch(input_ids_list, tokenizer.pad_token_id, args.batch_size)

        # Baseline forward
        logger.info("  Baseline forward pass (%d batches) ...", len(batches))
        baseline_results = collect_logits_batched(
            baseline_model, batches, baseline_device
        )
        ppl_baseline = compute_perplexity(baseline_results)
        del baseline_results
        logger.info("  PPL baseline: %.4f", ppl_baseline)

        if direct_comparison:
            logger.info("  Perturbed forward pass (%d batches) ...", len(batches))
            perturbed_results = collect_logits_batched(
                perturbed_model,
                batches,
                perturbed_device,
            )
            ppl_ablated = compute_perplexity(perturbed_results)
            del perturbed_results
            logger.info("  PPL perturbed: %.4f", ppl_ablated)
        else:
            # Ablated forward
            logger.info("  Ablated forward pass (%d batches) ...", len(batches))
            modifier.apply()
            try:
                ablated_results = collect_logits_batched(
                    perturbed_model,
                    batches,
                    perturbed_device,
                )
                ppl_ablated = compute_perplexity(ablated_results)
                del ablated_results
            finally:
                modifier.remove()
            logger.info("  PPL ablated:  %.4f", ppl_ablated)

        # Delta
        if ppl_baseline > 0 and ppl_baseline != float("inf"):
            delta_pct = (ppl_ablated - ppl_baseline) / ppl_baseline * 100.0
        else:
            delta_pct = float("nan")

        logger.info("  Delta:        %.2f%%", delta_pct)

        corpus_results[corpus_name] = {
            "ppl_baseline": round(ppl_baseline, 4),
            "ppl_ablated": round(ppl_ablated, 4),
            "ppl_perturbed": round(ppl_ablated, 4) if direct_comparison else None,
            "delta_pct": round(delta_pct, 2),
            "num_samples": len(input_ids_list),
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
        "prompt_only": args.prompt_only,
        "total_neurons_modified": None if direct_comparison else plan.total_neurons,
        "results": corpus_results,
        "elapsed_s": round(elapsed, 2),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    # Print summary table
    print("\n" + "=" * 70)
    print("  PPL Evaluation Summary")
    print("=" * 70)
    if direct_comparison:
        print("  Mode:    direct model comparison")
        print(f"  Base:    {args.baseline_model}")
        print(f"  Pert:    {args.perturbed_model}")
    else:
        print(f"  Model:   {args.model}")
        print(f"  Concept: {args.concept} ({args.operation})")
        if args.top_k_count is not None:
            print(f"  Top-k:   {args.top_k_count} per layer per module")
        else:
            print(f"  Top-k%%:  {args.top_k_percent}")
        print(f"  Modules: {args.modules}")
        print(f"  Neurons: {plan.total_neurons}")
    print("-" * 70)
    comparison_label = "Perturbed" if direct_comparison else "Ablated"
    print(
        f"  {'Corpus':<20} {'Baseline':>10} "
        f"{comparison_label:>10} {'Delta%':>10} {'Samples':>8}"
    )
    print("-" * 70)
    for name, r in corpus_results.items():
        bl = f"{r['ppl_baseline']:.2f}" if r["ppl_baseline"] is not None else "N/A"
        ab = f"{r['ppl_ablated']:.2f}" if r["ppl_ablated"] is not None else "N/A"
        dp = f"{r['delta_pct']:+.2f}%" if r["delta_pct"] is not None else "N/A"
        print(f"  {name:<20} {bl:>10} {ab:>10} {dp:>10} {r['num_samples']:>8}")
    print("=" * 70)
    print(f"  Elapsed: {elapsed:.1f}s")
    print("=" * 70 + "\n")

    # Write JSON
    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, f"ppl_eval_{results['timestamp']}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Results written to %s", json_path)

    return results


# ======================================================================
# CLI
# ======================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate perplexity change after RACE neuron suppression",
    )
    add_common_args(parser)
    args = parser.parse_args()

    validate_common_args(args, parser)
    if args.top_k_count is not None and args.top_k_count < 0:
        parser.error("--top-k must be non-negative")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    run_ppl_evaluation(args)


if __name__ == "__main__":
    main()
