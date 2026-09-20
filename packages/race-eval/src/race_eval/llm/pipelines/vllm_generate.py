"""Stage 1: vLLM inference pipeline — generate and save per-sample results.

Runs greedy generation on a dataset using vLLM (flash_attention_2) and writes
one JSON record per line to a JSONL output file.  Each record contains all
information required by Stage 2 (prefill_race.py) to reconstruct the full
token sequence and run RACE analysis without re-generating.

Output JSONL schema (one object per line):

    {
      "sample_index":        int,
      "prompt":              str,
      "prompt_token_ids":    List[int],
      "generated_text":      str,
      "generated_token_ids": List[int],
      "is_correct":          bool   # placeholder — True by default
    }

Usage example (standalone, no race package required)::

    python vllm_generate.py \\
        --model  Qwen/Qwen3-4B-Instruct-2507 \\
        --dataset math500 \\
        --output-dir result/llm/vllm_generate

    # or via uvx (only vllm + datasets needed):
    uvx --with vllm --with datasets python /path/to/vllm_generate.py ...
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ======================================================================
# Inline dataset registry (decoupled from race package)
# ======================================================================


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    defaults: Dict[str, Any]
    load_fn: Callable[..., Any]
    format_fn: Callable[..., Dict[str, Any]]


_REGISTRY: Dict[str, DatasetSpec] = {}

_DISALLOWED_DEFAULT_KEYS = {
    "do_sample",
    "eos_token_id",
    "length_penalty",
    "max_length",
    "max_new_tokens",
    "min_length",
    "min_new_tokens",
    "no_repeat_ngram_size",
    "num_beams",
    "pad_token_id",
    "repetition_penalty",
    "temperature",
    "top_k",
    "top_p",
}


def _register_dataset(spec: DatasetSpec) -> None:
    key = spec.dataset_id.lower()
    if key in _REGISTRY:
        raise KeyError(f"Dataset already registered: {spec.dataset_id!r}")
    disallowed = sorted(_DISALLOWED_DEFAULT_KEYS.intersection(spec.defaults))
    if disallowed:
        raise ValueError(
            f"Dataset {spec.dataset_id!r} defaults contain generation "
            f"parameters, which must be set by vLLM SamplingParams or CLI "
            f"flags instead: {', '.join(disallowed)}"
        )
    _REGISTRY[key] = spec


def _get_dataset_spec(dataset_id: str) -> DatasetSpec:
    key = (dataset_id or "").lower()
    if key not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY.keys())) or "<empty>"
        raise KeyError(f"Unknown dataset {dataset_id!r}. Known: {known}")
    return _REGISTRY[key]


# ======================================================================
# Inline MATH-500 loader & formatter
# ======================================================================

_MATH500_DEFAULTS: Dict[str, Any] = {
    "instruction_prefix": "",
    "response_prefix": "",
}

_MATH_MAGIC_SPLITTER = "-[[]]-this-is-really-our-highest-priority-[[]]-"


def _load_split_auto(dataset_name: str, preferred_split: Optional[str]) -> Any:
    from datasets import Dataset, DatasetDict, load_dataset  # type: ignore[import]

    if preferred_split is not None:
        ds = load_dataset(dataset_name, split=preferred_split)
        if not isinstance(ds, Dataset):
            raise ValueError("Expected Dataset, got DatasetDict")
        return ds

    dd = load_dataset(dataset_name)
    if isinstance(dd, Dataset):
        return dd
    if not isinstance(dd, DatasetDict):
        raise ValueError(f"Unexpected dataset type: {type(dd)!r}")

    for key in ("test", "validation", "train"):
        if key in dd:
            return dd[key]
    first_key = next(iter(dd.keys()))
    return dd[first_key]


def _load_math500(
    max_samples: Optional[int] = None,
    dataset_name: str = "HuggingFaceH4/MATH-500",
    split: Optional[str] = None,
) -> Any:
    logger.info("Loading %s (split=%s) ...", dataset_name, split or "<auto>")
    dataset = _load_split_auto(dataset_name, split)

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    def _add_prompt(example: Dict[str, Any]) -> Dict[str, Any]:
        example["prompt"] = example.get("problem", "")
        return example

    dataset = dataset.map(_add_prompt)
    logger.info("Loaded %d samples", len(dataset))
    return dataset


def _format_math500(
    example: Dict[str, Any],
    tokenizer: Any,
    split: str = "instruct",
    instruction_prefix: str = _MATH500_DEFAULTS["instruction_prefix"],
    response_prefix: str = _MATH500_DEFAULTS["response_prefix"],
    prefill: bool = True,
    direct_completion: bool = False,
) -> Dict[str, Any]:
    _ = split

    question = example.get("prompt") or example.get("problem") or ""
    task_prompt = (
        f"{question.strip()}\n\n"
        "Please reason step by step, and put your final answer within \\boxed{}."
    )

    if tokenizer.chat_template is None or direct_completion:
        _ = instruction_prefix
        example["formatted_prompt"] = f"{task_prompt}\n"
        return example

    formatted_instruction = task_prompt

    if prefill:
        _ = response_prefix
        response = _MATH_MAGIC_SPLITTER
        formatted_prompt = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": formatted_instruction},
                {"role": "assistant", "content": response},
            ],
            tokenize=False,
        ).split(_MATH_MAGIC_SPLITTER)[0]
    else:
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": formatted_instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )

    example["formatted_prompt"] = formatted_prompt
    return example


# ======================================================================
# Register built-in datasets
# ======================================================================


def _ensure_datasets_registered() -> None:
    """Register built-in datasets (idempotent)."""
    if _REGISTRY:
        return
    _register_dataset(
        DatasetSpec(
            dataset_id="math500",
            defaults=_MATH500_DEFAULTS,
            load_fn=_load_math500,
            format_fn=_format_math500,
        )
    )


# ======================================================================
# Dataset helpers
# ======================================================================


def _load_and_format_dataset(
    dataset_id: str,
    tokenizer: Any,
    *,
    max_samples: Optional[int],
    dataset_name: Optional[str],
    dataset_split: Optional[str],
    prompt_split: str,
    instruction_prefix: Optional[str],
    response_prefix: Optional[str],
    prefill: bool,
):
    """Load a registered dataset and apply prompt formatting."""
    _ensure_datasets_registered()
    spec = _get_dataset_spec(dataset_id)

    load_kwargs: Dict[str, Any] = {"max_samples": max_samples}
    if dataset_name is not None:
        load_kwargs["dataset_name"] = dataset_name
    if dataset_split is not None:
        load_kwargs["split"] = dataset_split

    dataset = spec.load_fn(**load_kwargs)
    if len(dataset) == 0:
        return dataset, spec

    fmt = partial(
        spec.format_fn,
        tokenizer=tokenizer,
        split=prompt_split,
        instruction_prefix=instruction_prefix or spec.defaults["instruction_prefix"],
        response_prefix=response_prefix or spec.defaults["response_prefix"],
        prefill=prefill,
    )
    dataset = dataset.map(fmt, batch_size=256)
    logger.info("Formatted %d prompts for dataset '%s'", len(dataset), dataset_id)
    return dataset, spec


# ======================================================================
# Main generation function
# ======================================================================


def run_vllm_generate(
    *,
    model_name: str,
    dataset_id: str,
    output_path: str,
    max_samples: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
    batch_size: int = 32,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.90,
    dataset_name: Optional[str] = None,
    dataset_split: Optional[str] = None,
    prompt_split: str = "instruct",
    instruction_prefix: Optional[str] = None,
    response_prefix: Optional[str] = None,
    prefill: bool = True,
) -> str:
    """Run vLLM greedy generation on a dataset and save results to JSONL.

    Args:
        model_name: HuggingFace model identifier.
        dataset_id: Registered dataset ID (e.g. ``"math500"``).
        output_path: Destination ``.jsonl`` file path.
        max_samples: Cap on the number of samples to process.
        max_new_tokens: Maximum tokens to generate. If omitted, vLLM's
            ``SamplingParams`` default is used.
        batch_size: Number of prompts per vLLM generation call.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        gpu_memory_utilization: Fraction of GPU memory for vLLM KV cache.
        dataset_name: Override HuggingFace dataset name.
        dataset_split: Override HuggingFace split/version.
        prompt_split: Prompt style (``"instruct"`` or ``"complete"``).
        instruction_prefix: Override dataset default instruction prefix.
        response_prefix: Override dataset default response prefix.
        prefill: Whether to add response prefix for prefill-style prompts.

    Returns:
        Absolute path to the saved JSONL file.
    """
    from vllm import LLM, SamplingParams  # deferred import — vLLM is optional

    _ensure_datasets_registered()

    t_start = time.perf_counter()

    logger.info("=" * 70)
    logger.info("  vLLM Generation — Stage 1")
    logger.info("=" * 70)
    logger.info(
        "  model=%s  dataset=%s  max_new_tokens=%s  batch_size=%d",
        model_name,
        dataset_id,
        "vllm_default" if max_new_tokens is None else str(max_new_tokens),
        batch_size,
    )
    logger.info("  output=%s", output_path)
    logger.info("=" * 70)

    # ------------------------------------------------------------------
    # Initialise vLLM engine
    # ------------------------------------------------------------------
    logger.info("[1/3] Initialising vLLM engine ...")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype="bfloat16",
        trust_remote_code=True,
        enable_prefix_caching=False,
        max_model_len=65536,
    )
    tokenizer = llm.get_tokenizer()

    # ------------------------------------------------------------------
    # Load and format dataset
    # ------------------------------------------------------------------
    logger.info("[2/3] Loading and formatting dataset '%s' ...", dataset_id)
    dataset, _ = _load_and_format_dataset(
        dataset_id,
        tokenizer,
        max_samples=max_samples,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        prompt_split=prompt_split,
        instruction_prefix=instruction_prefix,
        response_prefix=response_prefix,
        prefill=prefill,
    )
    if len(dataset) == 0:
        logger.warning("No samples found — aborting.")
        return ""

    prompts: List[str] = dataset["formatted_prompt"]
    n = len(prompts)

    # Greedy decoding: temperature=0, top_p=1
    sampling_kwargs: Dict[str, Any] = dict(
        temperature=0.0,
        top_p=1.0,
        # Retain special tokens so that prompt_token_ids length stays consistent
        # with what transformers would produce (needed for correct prefill slicing).
        skip_special_tokens=False,
    )
    if max_new_tokens is not None:
        sampling_kwargs["max_tokens"] = max_new_tokens
    sampling_params = SamplingParams(**sampling_kwargs)

    # ------------------------------------------------------------------
    # Generate and stream results to JSONL
    # ------------------------------------------------------------------
    logger.info("[3/3] Generating %d samples ...", n)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    t_gen_start = time.perf_counter()
    total_gen_tokens = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            batch_prompts = prompts[batch_start:batch_end]

            t0 = time.perf_counter()
            outputs = llm.generate(batch_prompts, sampling_params)
            dt = time.perf_counter() - t0

            gen_lens: List[int] = []
            for local_idx, output in enumerate(outputs):
                global_idx = batch_start + local_idx
                prompt_token_ids: List[int] = list(output.prompt_token_ids)
                generated_token_ids: List[int] = list(output.outputs[0].token_ids)
                generated_text: str = output.outputs[0].text
                gen_lens.append(len(generated_token_ids))
                total_gen_tokens += len(generated_token_ids)

                record = {
                    "sample_index": global_idx,
                    "prompt": batch_prompts[local_idx],
                    "prompt_token_ids": prompt_token_ids,
                    "generated_text": generated_text,
                    "generated_token_ids": generated_token_ids,
                    "is_correct": True,  # placeholder; update externally if needed
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            f.flush()
            logger.info(
                "  [%d-%d/%d] %.2fs  gen_len: min=%d  max=%d  avg=%.1f",
                batch_start + 1,
                batch_end,
                n,
                dt,
                min(gen_lens),
                max(gen_lens),
                sum(gen_lens) / len(gen_lens),
            )

    t_total = time.perf_counter() - t_start
    t_gen = time.perf_counter() - t_gen_start

    logger.info("-" * 70)
    logger.info(
        "Done in %.1fs  (vLLM gen %.1fs)  throughput: %.2f samples/s  "
        "avg gen_tokens/sample: %.1f",
        t_total,
        t_gen,
        n / t_total,
        total_gen_tokens / n,
    )
    logger.info("Saved %d records → %s", n, output_path)
    logger.info("-" * 70)

    return output_path


# ======================================================================
# CLI entry point
# ======================================================================


def main() -> None:
    _ensure_datasets_registered()

    parser = argparse.ArgumentParser(
        description="Stage 1: vLLM greedy generation — save per-sample token IDs to JSONL"
    )

    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="HuggingFace model identifier",
    )
    parser.add_argument(
        "--dataset",
        default="math500",
        choices=["math500"],
        help="Registered dataset adapter",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path.  Auto-generated from --output-dir if not set.",
    )
    parser.add_argument(
        "--output-dir",
        default="result/llm/vllm_generate",
        help="Root directory for auto-named output files",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Maximum tokens to generate (default: dataset spec default)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-split", default=None)
    parser.add_argument(
        "--prompt-split",
        default="instruct",
        choices=["instruct", "complete"],
    )
    parser.add_argument("--instruction-prefix", default=None)
    parser.add_argument("--response-prefix", default=None)
    parser.add_argument("--no-prefill", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(args.output_dir, exist_ok=True)
        model_short = args.model.replace("/", "--")
        args.output = os.path.join(
            args.output_dir,
            f"{args.dataset}_{model_short}_{timestamp}.jsonl",
        )

    run_vllm_generate(
        model_name=args.model,
        dataset_id=args.dataset,
        output_path=args.output,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        prompt_split=args.prompt_split,
        instruction_prefix=args.instruction_prefix,
        response_prefix=args.response_prefix,
        prefill=not args.no_prefill,
    )


if __name__ == "__main__":
    main()
