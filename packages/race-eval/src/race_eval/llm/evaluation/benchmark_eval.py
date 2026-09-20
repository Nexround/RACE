"""Benchmark evaluation via an OpenAI-compatible RACE inference server.

Connects to a running RACE FastAPI server (``race_eval.llm.serving.server``)
using evalscope's native ``eval_type="openai_api"`` mode, then evaluates on
benchmarks such as ``math_500`` (accuracy) and ``mbpp_plus`` (pass@k).

Workflow::

    # 1. Start the RACE inference server (ablated model):
    python -m race_eval.llm.serving.server \\
        --model Qwen/Qwen3-4B-Instruct-2507 \\
        --device-ids 0 1 2 3 \\
        --race-h5 result/llm/online_race/.../online_race.h5 \\
        --concept math500 --operation mean_ablation \\
        --top-k 8 --metric lcb_pos --lcb-min-score 0 \\
        --modules attn --port 8000

    # 2. (Optional) Start a baseline server on a different port:
    python -m race_eval.llm.serving.server \\
        --model Qwen/Qwen3-4B-Instruct-2507 \\
        --device-ids 4 5 6 7 --no-race --port 8001

    # 3. Run evaluation:
    python -m race_eval.llm.evaluation.benchmark_eval \\
        --api-url http://127.0.0.1:8000/v1 \\
        --model-name Qwen--Qwen3-4B-Instruct-2507__ablated \\
        --benchmarks math_500 mbpp_plus \\
        --baseline-api-url http://127.0.0.1:8001/v1 \\
        --baseline-model-name Qwen--Qwen3-4B-Instruct-2507__baseline
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ======================================================================
# Helpers
# ======================================================================


def _wait_server_ready(
    api_url: str, timeout: float = 120.0, interval: float = 2.0
) -> None:
    """Poll ``/v1/models`` until the server responds or timeout expires."""
    models_url = api_url.rstrip("/") + "/models"
    deadline = time.monotonic() + timeout
    last_err: Optional[Exception] = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(models_url, timeout=10) as resp:
                if 200 <= resp.status < 300:
                    logger.info("Server ready at %s", api_url)
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_err = exc
        time.sleep(interval)
    raise RuntimeError(
        f"Server at {api_url} did not become ready within {timeout}s. "
        f"Last error: {last_err}"
    )


def _run_evalscope(
    api_url: str,
    model_name: str,
    args: argparse.Namespace,
    work_dir: str,
) -> Any:
    """Run evalscope benchmark evaluation against an OpenAI-compatible API."""
    from evalscope import TaskConfig, run_task

    dataset_args: Dict[str, Any] = {}
    for bench in args.benchmarks:
        if bench == "mbpp_plus":
            dataset_args["mbpp_plus"] = {"review_timeout": 30.0}

    generation_config: Dict[str, Any] = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }

    task_cfg = TaskConfig(
        model=model_name,
        api_url=api_url,
        api_key=args.api_key,
        eval_type="openai_api",
        datasets=list(args.benchmarks),
        dataset_args=dataset_args,
        eval_batch_size=args.eval_batch_size,
        generation_config=generation_config,
        work_dir=work_dir,
        limit=args.evalscope_limit,
        dataset_hub="huggingface",
    )

    logger.info(
        "Running evalscope: benchmarks=%s  model=%s  api_url=%s",
        args.benchmarks,
        model_name,
        api_url,
    )
    return run_task(task_cfg=task_cfg)


def _extract_scores(evalscope_output: Any) -> Dict[str, Any]:
    """Best-effort extraction of scores from evalscope run output."""
    if evalscope_output is None:
        return {}
    if isinstance(evalscope_output, dict):
        return evalscope_output
    if isinstance(evalscope_output, (list, tuple)):
        combined: Dict[str, Any] = {}
        for item in evalscope_output:
            if isinstance(item, dict):
                combined.update(item)
        return combined
    return {"raw": str(evalscope_output)}


def _print_summary(results: Dict[str, Any]) -> None:
    """Print a human-readable summary table."""
    print("\n" + "=" * 70)
    print("  Benchmark Evaluation Summary")
    print("=" * 70)
    print(f"  API URL:    {results['api_url']}")
    print(f"  Model:      {results['model_name']}")
    print(f"  Benchmarks: {results['benchmarks']}")
    print("-" * 70)

    if results.get("ablated_results"):
        print("  [Ablated]")
        for k, v in results["ablated_results"].items():
            print(f"    {k}: {v}")

    if results.get("baseline_results"):
        print("  [Baseline]")
        for k, v in results["baseline_results"].items():
            print(f"    {k}: {v}")

    print("=" * 70)
    print(f"  Elapsed: {results['elapsed_s']:.1f}s")
    print("=" * 70 + "\n")


# ======================================================================
# Main evaluation pipeline
# ======================================================================


def run_benchmark_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    """Evaluate benchmarks against a running RACE server via OpenAI API.

    Steps:
        1. Wait for the ablated server to be ready.
        2. Run evalscope on the ablated server (``eval_type="openai_api"``).
        3. Optionally run evalscope on a baseline server.
        4. Write summary JSON.
    """
    t_start = time.perf_counter()

    # 1. Wait for server readiness
    _wait_server_ready(args.api_url, timeout=args.server_wait_timeout)

    # 2. Run ablated evaluation
    ablated_results = _run_evalscope(
        api_url=args.api_url,
        model_name=args.model_name,
        args=args,
        work_dir=os.path.join(args.evalscope_work_dir, "ablated"),
    )

    # 3. Baseline evaluation (optional)
    baseline_results = None
    if args.baseline_api_url:
        _wait_server_ready(args.baseline_api_url, timeout=args.server_wait_timeout)
        baseline_model_name = args.baseline_model_name or args.model_name
        baseline_results = _run_evalscope(
            api_url=args.baseline_api_url,
            model_name=baseline_model_name,
            args=args,
            work_dir=os.path.join(args.evalscope_work_dir, "baseline"),
        )

    elapsed = time.perf_counter() - t_start

    # 4. Assemble and write summary
    results: Dict[str, Any] = {
        "api_url": args.api_url,
        "model_name": args.model_name,
        "benchmarks": list(args.benchmarks),
        "evalscope_limit": args.evalscope_limit,
        "ablated_results": _extract_scores(ablated_results),
        "baseline_results": (
            _extract_scores(baseline_results) if baseline_results else None
        ),
        "elapsed_s": round(elapsed, 2),
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    _print_summary(results)

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(
        args.output_dir,
        f"benchmark_eval_{results['timestamp']}.json",
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Results written to %s", json_path)

    return results


# ======================================================================
# CLI
# ======================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark evaluation (math_500 / mbpp_plus) via RACE OpenAI-compatible server. "
            "Start the server first with: python -m race_eval.llm.serving.server"
        ),
    )

    # --- Server connection ---
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8000/v1",
        help="Base URL of the RACE inference server (ablated model)",
    )
    parser.add_argument(
        "--api-key",
        default="EMPTY",
        help="API key (usually not required for local servers)",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Model name as registered on /v1/models of the ablated server",
    )
    parser.add_argument(
        "--baseline-api-url",
        default=None,
        help="Base URL of a second (baseline) server. If set, also runs baseline evaluation.",
    )
    parser.add_argument(
        "--baseline-model-name",
        default=None,
        help="Model name for the baseline server (defaults to --model-name)",
    )
    parser.add_argument(
        "--server-wait-timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for the server to become ready",
    )

    # --- Benchmark settings ---
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["math_500", "mbpp_plus"],
        help="Evalscope benchmarks to evaluate",
    )
    parser.add_argument(
        "--evalscope-limit",
        type=int,
        default=None,
        help="Limit number of evalscope samples per benchmark (for debugging)",
    )
    parser.add_argument(
        "--evalscope-work-dir",
        default="result/llm/benchmark_eval",
        help="Evalscope output directory",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=32,
        help="Evalscope concurrent request batch size",
    )

    # --- Generation config ---
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum tokens to generate per request",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.8,
        help="Top-p sampling parameter (default: 0.8)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-k sampling parameter (default: 20)",
    )

    # --- Output ---
    parser.add_argument(
        "--output-dir",
        default="result/llm/eval",
        help="Output directory for results JSON",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    run_benchmark_evaluation(args)


if __name__ == "__main__":
    main()
