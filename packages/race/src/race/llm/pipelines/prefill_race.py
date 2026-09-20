"""Stage 2: RACE analysis via prefill.

Loads pre-generated token IDs produced by Stage 1 (``vllm_generate.py``) and
runs RACE analysis using a single transformer forward pass per batch rather
than autoregressive decoding.

Pipeline
--------
::

    [JSONL from Stage 1]
          │
          ▼  load_generation_results()
    List[{sample_index, prompt_token_ids, generated_token_ids, is_correct, …}]
          │
          ▼  PrefillActivationRecorder.record_from_token_ids()
    Single model.forward() on [prompt + generated] with right-padding
          │  (hooks capture activation[:, prompt_len:prompt_len+T, :] per layer)
          ▼  stream_evidence()
    OnlineRACEAccumulator.accumulate_from_activations()
          │
          ▼  save_domain_to_h5()
    NIG posteriors → .h5 + run_config.json

Usage example::

    uv run python -m race.llm.pipelines.prefill_race \\
        --model  Qwen/Qwen3-4B-Instruct-2507 \\
        --generation-file  result/llm/vllm_generate/math500_*.jsonl \\
        --output-dir  result/llm/prefill_race
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from race.core.online_accumulator import OnlineRACEAccumulator, OnlineRACEConfig
from race.llm.recording.prefill_recorder import PrefillActivationRecorder

logger = logging.getLogger(__name__)

torch.set_grad_enabled(False)


# ======================================================================
# JSONL loader
# ======================================================================


def load_generation_results(
    path: str,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load per-sample generation results from a Stage-1 JSONL file.

    Each line must contain at minimum:
    ``sample_index``, ``prompt``, ``prompt_token_ids``,
    ``generated_token_ids``, ``generated_text``, ``is_correct``.

    Args:
        path: Path to the ``.jsonl`` file produced by ``vllm_generate.py``.
        max_samples: If set, only load the first *N* records.

    Returns:
        List of record dicts.
    """
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if max_samples is not None and len(records) >= max_samples:
                break

    logger.info("Loaded %d generation records from %s", len(records), path)
    return records


# ======================================================================
# Batch processing
# ======================================================================


def _process_batches(
    recorder: PrefillActivationRecorder,
    records: List[Dict[str, Any]],
    accumulator: OnlineRACEAccumulator,
    *,
    domain: str,
    batch_size: int,
    sort_by_length: bool = True,
) -> Dict[str, Any]:
    """Iterate over records in batches, prefill, and accumulate RACE evidence.

    Args:
        recorder: Initialised :class:`PrefillActivationRecorder` with hooks.
        records: Loaded generation records.
        accumulator: Initialised :class:`OnlineRACEAccumulator`.
        domain: RACE domain label.
        batch_size: Number of samples per forward pass.
        sort_by_length: Sort batches by total sequence length to reduce padding.

    Returns:
        Timing and throughput statistics dict.
    """
    total = len(records)
    t_start = time.perf_counter()
    t_fwd_total = 0.0
    t_evi_total = 0.0
    n_processed = 0

    # Optionally sort by total token count to minimise padding waste.
    if sort_by_length and batch_size > 1:
        order = sorted(
            range(total),
            key=lambda i: (
                len(records[i]["prompt_token_ids"])
                + len(records[i]["generated_token_ids"])
            ),
        )
        logger.info("Sorted %d records by total sequence length for batching", total)
    else:
        order = list(range(total))

    with torch.inference_mode():
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_num = batch_start // batch_size + 1

            logger.info(
                "[Batch %d] samples %d-%d / %d",
                batch_num,
                batch_start + 1,
                batch_end,
                total,
            )

            batch_records = [records[order[k]] for k in range(batch_start, batch_end)]

            prompt_token_ids = [r["prompt_token_ids"] for r in batch_records]
            generated_token_ids = [r["generated_token_ids"] for r in batch_records]
            sample_indices = [r["sample_index"] for r in batch_records]
            prompts = [r["prompt"] for r in batch_records]
            generated_texts = [r["generated_text"] for r in batch_records]
            is_correct = [bool(r.get("is_correct", True)) for r in batch_records]

            # Detect empty generations and mark those samples as incorrect so
            # the accumulator skips them without crashing.
            empty_mask = [len(g) == 0 for g in generated_token_ids]
            n_empty = sum(empty_mask)
            if n_empty == len(batch_records):
                logger.warning(
                    "  All %d samples in batch have empty generations — skipping batch",
                    n_empty,
                )
                continue
            if n_empty > 0:
                logger.warning(
                    "  %d / %d samples have empty generations — marked incorrect",
                    n_empty,
                    len(batch_records),
                )
                is_correct = [c and not e for c, e in zip(is_correct, empty_mask)]

            prompt_lens = [len(p) for p in prompt_token_ids]
            gen_lens = [len(g) for g in generated_token_ids]
            total_lens = [p + g for p, g in zip(prompt_lens, gen_lens)]

            t0 = time.perf_counter()
            result = recorder.record_from_token_ids(
                prompt_token_ids=prompt_token_ids,
                generated_token_ids=generated_token_ids,
                sample_indices=sample_indices,
                prompts=prompts,
                generated_texts=generated_texts,
                is_correct=is_correct,
            )
            dt_fwd = time.perf_counter() - t0
            t_fwd_total += dt_fwd

            logger.info(
                "  prefill %.2fs  total_len: min=%d max=%d avg=%.1f  "
                "gen_len: min=%d max=%d avg=%.1f",
                dt_fwd,
                min(total_lens),
                max(total_lens),
                sum(total_lens) / len(total_lens),
                min(gen_lens),
                max(gen_lens),
                sum(gen_lens) / len(gen_lens),
            )

            t0 = time.perf_counter()
            n = recorder.stream_evidence(result, accumulator, domain)
            dt_evi = time.perf_counter() - t0
            t_evi_total += dt_evi
            n_processed += n

            logger.info(
                "  evidence %.2fs  (%d samples)  progress %d/%d (%.1f%%)",
                dt_evi,
                n,
                batch_end,
                total,
                100.0 * batch_end / total,
            )

    elapsed = time.perf_counter() - t_start
    return {
        "total": total,
        "processed": n_processed,
        "elapsed": elapsed,
        "forward_s": t_fwd_total,
        "evidence_s": t_evi_total,
        "other_s": elapsed - t_fwd_total - t_evi_total,
        "throughput": total / elapsed if elapsed > 0 else 0.0,
    }


# ======================================================================
# Main analysis function
# ======================================================================


def run_prefill_race(
    *,
    model_name: str,
    generation_file: str,
    output_dir: str,
    domain: Optional[str] = None,
    config: Optional[OnlineRACEConfig] = None,
    batch_size: int = 4,
    sort_by_length: bool = True,
    max_samples: Optional[int] = None,
) -> str:
    """Run RACE analysis on pre-generated outputs via single-pass prefill.

    Args:
        model_name: HuggingFace model identifier (must match the model used in
            Stage 1 so that token IDs are consistent).
        generation_file: Path to JSONL from ``vllm_generate.py``.
        output_dir: Root output directory.  A timestamped subdirectory is
            created inside it for each run.
        domain: RACE domain label.  If ``None``, inferred from the JSONL
            filename stem (e.g. ``"math500_..."`` -> ``"math500"``).
        config: RACE hyperparameter config (default: :class:`OnlineRACEConfig`
            defaults).
        batch_size: Samples per forward pass.  Larger values fill the GPU
            better but require more VRAM for long sequences.
        sort_by_length: Sort batches by total sequence length to reduce
            right-padding waste.
        max_samples: Cap on records to process (useful for debugging).

    Returns:
        Absolute path to the saved ``.h5`` file.
    """
    t0_total = time.perf_counter()
    race_cfg = config or OnlineRACEConfig()

    records = load_generation_results(generation_file, max_samples=max_samples)
    if not records:
        logger.warning("No generation records found in %s — aborting.", generation_file)
        return ""

    # Infer domain from filename if not provided.
    # E.g. "math500_Qwen3-4B_20260319.jsonl" -> "math500"
    if domain is None:
        domain = Path(generation_file).stem.split("_")[0]
        logger.info("Domain inferred from filename: '%s'", domain)

    logger.info("=" * 70)
    logger.info("  Prefill RACE Analysis — Stage 2")
    logger.info("=" * 70)
    logger.info(
        "  model=%s  domain=%s  samples=%d  batch=%d",
        model_name,
        domain,
        len(records),
        batch_size,
    )
    logger.info(
        "  RACE: mu_0=%.1f  lambda_0=%.1f  alpha_0=%.1f  beta_0=%.1f  gamma=%.3f",
        race_cfg.mu_0,
        race_cfg.lambda_0,
        race_cfg.alpha_0,
        race_cfg.beta_0,
        race_cfg.gamma,
    )
    logger.info("  generation_file=%s", generation_file)
    logger.info("  output_dir=%s", output_dir)
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Stage 2-1: set up recorder
    # ------------------------------------------------------------------
    logger.info("[1/4] Loading model and registering hooks ...")
    recorder = PrefillActivationRecorder(
        model_name=model_name,
        output_dir=output_dir,
    )
    recorder.load_model(device="cuda")
    recorder.register_hooks()

    # ------------------------------------------------------------------
    # Stage 2-2: extract weights and init accumulator
    # ------------------------------------------------------------------
    logger.info("[2/4] Extracting layer weights and initialising accumulator ...")
    layer_weights = recorder.extract_layer_weights()
    accumulator = OnlineRACEAccumulator(
        model_name=model_name,
        layer_weights=layer_weights,
        config=race_cfg,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    h5_name = f"{domain}_prefill_race.h5"
    output_path = accumulator.init_h5_file(os.path.join(run_dir, h5_name))
    logger.info("[3/4] H5 initialised: %s", output_path)

    # ------------------------------------------------------------------
    # Stage 2-3: prefill + RACE evidence accumulation
    # ------------------------------------------------------------------
    logger.info(
        "[4/4] Processing %d samples (batch_size=%d) ...",
        len(records),
        batch_size,
    )
    stats = _process_batches(
        recorder,
        records,
        accumulator,
        domain=domain,
        batch_size=batch_size,
        sort_by_length=sort_by_length,
    )

    recorder.remove_hooks()

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    t_save = time.perf_counter()
    logger.info("Saving domain '%s' to H5 ...", domain)
    accumulator.save_domain_to_h5(output_path, domain)
    accumulator.finalize_h5_file(output_path)

    t_total = time.perf_counter() - t0_total

    meta = {
        "model_name": model_name,
        "generation_file": os.path.abspath(generation_file),
        "domain": domain,
        "timestamp": os.path.basename(run_dir),
        "config": {
            "mu_0": race_cfg.mu_0,
            "lambda_0": race_cfg.lambda_0,
            "alpha_0": race_cfg.alpha_0,
            "beta_0": race_cfg.beta_0,
            "gamma": race_cfg.gamma,
            "batch_size": batch_size,
            "sort_by_length": sort_by_length,
            "max_samples": max_samples,
        },
        "results": {
            "total_samples": stats["total"],
            "processed": stats["processed"],
            "unique_domains": len(accumulator.state.domains),
        },
        "timing": {
            "total_s": round(t_total, 2),
            "forward_s": round(stats["forward_s"], 2),
            "evidence_s": round(stats["evidence_s"], 2),
            "other_s": round(stats["other_s"], 2),
            "save_s": round(time.perf_counter() - t_save, 2),
            "throughput": round(stats["throughput"], 2),
        },
        "environment": {
            "device": "cuda",
            "torch_version": torch.__version__,
            "cuda_device": (
                torch.cuda.get_device_name() if torch.cuda.is_available() else None
            ),
        },
        "output_h5": os.path.basename(output_path),
    }

    json_path = os.path.join(run_dir, "run_config.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    logger.info("-" * 70)
    logger.info(
        "Done in %.1fs  (fwd %.1fs, evi %.1fs, other %.1fs)  %.2f samples/s",
        t_total,
        stats["forward_s"],
        stats["evidence_s"],
        stats["other_s"],
        stats["throughput"],
    )
    logger.info("H5   → %s", output_path)
    logger.info("JSON → %s", json_path)
    logger.info("-" * 70)

    return output_path


# ======================================================================
# CLI entry point
# ======================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2: Prefill-based RACE analysis on pre-generated outputs"
    )

    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="HuggingFace model identifier (must match Stage 1 model)",
    )
    parser.add_argument(
        "--generation-file",
        required=True,
        help="Path to the JSONL file produced by Stage 1 (vllm_generate.py)",
    )
    parser.add_argument(
        "--output-dir",
        default="result/llm/prefill_race",
        help="Root directory for output H5 and run_config.json",
    )
    parser.add_argument(
        "--domain",
        default=None,
        help="RACE domain label (default: inferred from JSONL filename stem)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Samples per prefill forward pass",
    )
    parser.add_argument(
        "--no-sort-by-length",
        action="store_true",
        help="Disable length-based batch sorting",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of records to process (useful for debugging)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.05,
        help="RACE credible-interval confidence level",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    os.makedirs(args.output_dir, exist_ok=True)

    race_config = OnlineRACEConfig(gamma=args.gamma, device="cuda", model_family="llm")

    run_prefill_race(
        model_name=args.model,
        generation_file=args.generation_file,
        output_dir=args.output_dir,
        domain=args.domain,
        config=race_config,
        batch_size=args.batch_size,
        sort_by_length=not args.no_sort_by_length,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
