"""Unified online RACE pipeline for LLM datasets.

Supports multiple datasets via a lightweight adapter registry in
``race.llm.dataset.registry``. Dataset selection is controlled by CLI
argument ``--dataset``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from datetime import datetime
from functools import partial
from typing import Any, Dict, Optional

import torch

from race.core.online_accumulator import OnlineRACEAccumulator, OnlineRACEConfig
from race.llm.dataset.math500 import build_boxed_token_mask
from race.llm.dataset.registry import (
    ensure_builtin_datasets_registered,
    get_dataset_spec,
)
from race.llm.evalscope_results import (
    load_evalscope_generations as _shared_load_evalscope_generations,
)
from race.llm.recording.activation import LLMActivationRecorder

logger = logging.getLogger(__name__)

torch.set_grad_enabled(False)

DEFAULT_SAMPLE_SEED = 42
_EVALSCOPE_EVIDENCE_START_COLUMN = "evalscope_evidence_start_char"


def _setup_recorder(
    model_name: str,
    output_dir: str,
    *,
    first_token_only: bool = False,
    model_device_map: Any = "cuda",
    input_device: Optional[str] = None,
    activation_device: str = "cuda",
    compile_model: bool = True,
) -> LLMActivationRecorder:
    recorder = LLMActivationRecorder(
        model_name=model_name,
        output_dir=output_dir,
        first_token_only=first_token_only,
        activation_device=activation_device,
    )
    recorder.load_model(
        device_map=model_device_map,
        input_device=input_device,
        compile_model=compile_model,
    )
    recorder.register_hooks()
    return recorder


def _prepare_dataset(
    recorder: LLMActivationRecorder,
    *,
    dataset_id: str,
    dataset_name: Optional[str],
    dataset_split: Optional[str],
    max_samples: Optional[int],
    random_sample: bool,
    sample_seed: int,
    prompt_split: str,
    instruction_prefix: str,
    response_prefix: str,
    prefill: bool,
):
    ensure_builtin_datasets_registered()
    spec = get_dataset_spec(dataset_id)

    loader_handles_sampling = getattr(spec, "loader_handles_sampling", False)
    load_kwargs: Dict[str, Any] = {
        "max_samples": (
            max_samples
            if loader_handles_sampling
            else (None if random_sample else max_samples)
        )
    }
    if loader_handles_sampling:
        load_kwargs.update(
            random_sample=random_sample,
            sample_seed=sample_seed,
        )
    if dataset_name is not None:
        load_kwargs["dataset_name"] = dataset_name
    if dataset_split is not None:
        load_kwargs["split"] = dataset_split

    dataset = spec.load_fn(**load_kwargs)
    if len(dataset) == 0:
        return dataset

    if random_sample and not loader_handles_sampling:
        if max_samples is None:
            logger.warning("random_sample=True ignored because max_samples is None")
        else:
            total_available = len(dataset)
            n_select = min(max_samples, len(dataset))
            if n_select < total_available:
                rng = random.Random(sample_seed)
                selected_indices = rng.sample(range(total_available), n_select)
                dataset = dataset.select(selected_indices)
                if "sample_index" not in getattr(dataset, "column_names", []):
                    dataset = dataset.add_column("sample_index", selected_indices)
                logger.info(
                    "Randomly selected %d/%d samples with seed=%d",
                    n_select,
                    total_available,
                    sample_seed,
                )

    fmt = partial(
        spec.format_fn,
        tokenizer=recorder.tokenizer,
        split=prompt_split,
        instruction_prefix=instruction_prefix,
        response_prefix=response_prefix,
        prefill=prefill,
    )
    dataset = dataset.map(fmt, batch_size=256)
    logger.info("Formatted %d prompts", len(dataset))
    return dataset


def _decode_token_prefix(tokenizer: Any, token_ids: list[int]) -> str:
    try:
        return tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(token_ids, skip_special_tokens=False)


def _truncate_evalscope_generation(
    generation: str,
    tokenizer: Any,
    *,
    max_evidence_tokens: Optional[int],
    max_evidence_chars: Optional[int],
) -> tuple[str, bool, bool]:
    """Limit EvalScope assistant text before it is appended as evidence."""
    char_truncated = False
    token_truncated = False

    if max_evidence_chars is not None and len(generation) > max_evidence_chars:
        generation = generation[:max_evidence_chars]
        char_truncated = True

    if max_evidence_tokens is None:
        return generation, char_truncated, token_truncated
    if max_evidence_tokens == 0:
        return "", char_truncated or bool(generation), bool(generation)

    try:
        encoded = tokenizer(
            generation,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        input_ids = encoded["input_ids"]
        if len(input_ids) <= max_evidence_tokens:
            return generation, char_truncated, token_truncated

        offsets = encoded.get("offset_mapping")
        token_truncated = True
        if offsets is not None and len(offsets) >= max_evidence_tokens:
            end_char = offsets[max_evidence_tokens - 1][1]
            if end_char > 0:
                return generation[:end_char], char_truncated, token_truncated
        return (
            _decode_token_prefix(tokenizer, input_ids[:max_evidence_tokens]),
            char_truncated,
            token_truncated,
        )
    except Exception:
        logger.warning(
            "Token-based EvalScope evidence truncation failed; using text as-is",
            exc_info=True,
        )
        return generation, char_truncated, token_truncated


def _load_evalscope_generations(
    results_dir: str,
    *,
    dataset_id: str,
    dataset,
) -> Dict[int, str]:
    return _shared_load_evalscope_generations(
        results_dir,
        dataset_id=dataset_id,
        dataset=dataset,
    )


def _prepare_evalscope_dataset(
    recorder: LLMActivationRecorder,
    *,
    dataset_id: str,
    dataset_name: Optional[str],
    dataset_split: Optional[str],
    max_samples: Optional[int],
    random_sample: bool,
    sample_seed: int,
    prompt_split: str,
    instruction_prefix: str,
    response_prefix: str,
    results_dir: str,
    max_evidence_tokens: Optional[int] = None,
    max_evidence_chars: Optional[int] = None,
    evidence_scope: str = "all",
):
    ensure_builtin_datasets_registered()
    spec = get_dataset_spec(dataset_id)
    if max_evidence_tokens is not None and max_evidence_tokens < 0:
        raise ValueError("max_evidence_tokens must be >= 0")
    if max_evidence_chars is not None and max_evidence_chars < 0:
        raise ValueError("max_evidence_chars must be >= 0")
    if evidence_scope not in {"all", "assistant"}:
        raise ValueError("evidence_scope must be 'all' or 'assistant'")

    load_kwargs: Dict[str, Any] = {"max_samples": None}
    if dataset_name is not None:
        load_kwargs["dataset_name"] = dataset_name
    if dataset_split is not None:
        load_kwargs["split"] = dataset_split

    dataset = spec.load_fn(**load_kwargs)
    if len(dataset) == 0:
        return dataset

    fmt = partial(
        spec.format_fn,
        tokenizer=recorder.tokenizer,
        split=prompt_split,
        instruction_prefix=instruction_prefix,
        response_prefix=response_prefix,
        prefill=False,
    )
    dataset = dataset.map(fmt, batch_size=256)

    index_to_generation = _load_evalscope_generations(
        results_dir,
        dataset_id=spec.dataset_id,
        dataset=dataset,
    )
    matched_indices = sorted(index_to_generation)
    if max_samples is not None:
        n_select = min(max_samples, len(matched_indices))
        if random_sample and n_select < len(matched_indices):
            rng = random.Random(sample_seed)
            matched_indices = rng.sample(matched_indices, n_select)
            logger.info(
                "Randomly selected %d/%d EvalScope-matched samples with seed=%d",
                n_select,
                len(index_to_generation),
                sample_seed,
            )
        else:
            matched_indices = matched_indices[:n_select]
    elif random_sample:
        logger.warning("random_sample=True ignored because max_samples is None")

    if not matched_indices:
        raise RuntimeError(
            f"No EvalScope samples from {results_dir!r} matched dataset "
            f"{spec.dataset_id!r}"
        )

    prompts = dataset["formatted_prompt"]
    full_inputs = []
    evidence_start_chars = []
    n_char_truncated = 0
    n_token_truncated = 0
    for i in matched_indices:
        generation, char_truncated, token_truncated = _truncate_evalscope_generation(
            index_to_generation[i],
            recorder.tokenizer,
            max_evidence_tokens=max_evidence_tokens,
            max_evidence_chars=max_evidence_chars,
        )
        prompt = prompts[i]
        full_inputs.append(prompt + generation)
        evidence_start_chars.append(len(prompt))
        n_char_truncated += int(char_truncated)
        n_token_truncated += int(token_truncated)

    dataset = dataset.select(matched_indices)
    remove_columns = ["formatted_prompt"]
    if _EVALSCOPE_EVIDENCE_START_COLUMN in dataset.column_names:
        remove_columns.append(_EVALSCOPE_EVIDENCE_START_COLUMN)
    dataset = dataset.remove_columns(remove_columns)
    dataset = dataset.add_column("formatted_prompt", full_inputs)
    dataset = dataset.add_column("sample_index", matched_indices)
    if evidence_scope == "assistant":
        dataset = dataset.add_column(
            _EVALSCOPE_EVIDENCE_START_COLUMN,
            evidence_start_chars,
        )

    if max_evidence_chars is not None or max_evidence_tokens is not None:
        logger.info(
            "EvalScope evidence truncation: max_chars=%s max_tokens=%s "
            "char_truncated=%d token_truncated=%d",
            "none" if max_evidence_chars is None else str(max_evidence_chars),
            "none" if max_evidence_tokens is None else str(max_evidence_tokens),
            n_char_truncated,
            n_token_truncated,
        )

    logger.info(
        "Prepared %d EvalScope teacher-forcing samples for dataset=%s "
        "(evidence_scope=%s)",
        len(dataset),
        spec.dataset_id,
        evidence_scope,
    )
    return dataset


def _build_teacher_forcing_token_masks(
    tokenizer: Any,
    prompts: list[str],
    evidence_start_chars: list[int],
) -> list[list[bool]]:
    """Build padded token masks for character-delimited evidence spans.

    Fast tokenizers expose character offsets, which lets us select exactly the
    tokens whose text falls after the prompt/assistant boundary.  The fallback
    uses token counts for tokenizers without offset mappings.
    """
    if len(prompts) != len(evidence_start_chars):
        raise ValueError("prompts and evidence_start_chars length mismatch")

    try:
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
        )
        offsets = encoded["offset_mapping"].tolist()
        attention_masks = encoded["attention_mask"].tolist()
        return [
            [
                bool(attention) and int(end) > int(start_char)
                for (_, end), attention in zip(sample_offsets, sample_attention)
            ]
            for sample_offsets, sample_attention, start_char in zip(
                offsets,
                attention_masks,
                evidence_start_chars,
            )
        ]
    except (KeyError, NotImplementedError, TypeError, ValueError):
        logger.debug(
            "Tokenizer offset mappings unavailable; falling back to token counts",
            exc_info=True,
        )

    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    attention_masks = encoded["attention_mask"].tolist()
    padding_side = getattr(tokenizer, "padding_side", "right")
    masks: list[list[bool]] = []
    for prompt, start_char, attention in zip(
        prompts,
        evidence_start_chars,
        attention_masks,
    ):
        sequence_length = int(sum(attention))
        prefix_length = len(
            tokenizer.encode(prompt[: int(start_char)], add_special_tokens=True)
        )
        prefix_length = min(prefix_length, sequence_length)
        padding_length = len(attention) - sequence_length
        evidence_length = sequence_length - prefix_length
        if padding_side == "left":
            mask = [False] * (padding_length + prefix_length) + [True] * evidence_length
        else:
            mask = (
                [False] * prefix_length
                + [True] * evidence_length
                + [False] * padding_length
            )
        masks.append(mask)
    return masks


def _process_batches(
    recorder: LLMActivationRecorder,
    dataset,
    accumulator: OnlineRACEAccumulator,
    *,
    domain: str,
    dataset_id: str,
    batch_size: int,
    gen_kwargs: Dict[str, Any],
    sort_by_length: bool = True,
    teacher_forcing: bool = False,
    math500_boxed_token_mask: bool = False,
) -> Dict[str, Any]:
    total = len(dataset)
    t_fwd_total = 0.0
    t_evi_total = 0.0
    n_processed = 0
    t_start = time.perf_counter()

    all_prompts = dataset["formatted_prompt"]
    sample_index_column = (
        dataset["sample_index"]
        if "sample_index" in getattr(dataset, "column_names", [])
        else None
    )
    evidence_start_column = (
        dataset[_EVALSCOPE_EVIDENCE_START_COLUMN]
        if _EVALSCOPE_EVIDENCE_START_COLUMN in getattr(dataset, "column_names", [])
        else None
    )

    if sort_by_length and batch_size > 1:
        assert recorder.tokenizer is not None, "tokenizer must be loaded"
        logger.info("Pre-tokenising %d prompts for length-based sorting ...", total)
        tokenizer = recorder.tokenizer
        prompt_lengths = [
            len(tokenizer.encode(p, add_special_tokens=False)) for p in all_prompts
        ]
        processing_order = sorted(range(total), key=lambda i: prompt_lengths[i])
        logger.info(
            "  prompt token lengths: min=%d, max=%d, avg=%.0f — sorted for batching",
            min(prompt_lengths),
            max(prompt_lengths),
            sum(prompt_lengths) / len(prompt_lengths),
        )
    else:
        processing_order = list(range(total))

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

            batch_order = processing_order[batch_start:batch_end]
            prompts = [all_prompts[i] for i in batch_order]
            indices = (
                [int(sample_index_column[i]) for i in batch_order]
                if sample_index_column is not None
                else batch_order
            )

            t0 = time.perf_counter()
            if teacher_forcing:
                result = recorder.record_sample_teacher_forcing(
                    prompts=prompts,
                    sample_indices=indices,
                    is_correct=[True] * len(prompts),
                )
            else:
                result = recorder.record_sample(
                    prompts=prompts,
                    sample_indices=indices,
                    is_correct=[True] * len(prompts),
                    **gen_kwargs,
                )
            dt_fwd = time.perf_counter() - t0
            t_fwd_total += dt_fwd
            logger.info("  inference %.2fs", dt_fwd)

            gen_lens = result.num_generated_tokens
            logger.info(
                "  generation lengths: %s  (min=%d, max=%d, avg=%.1f)",
                gen_lens,
                min(gen_lens),
                max(gen_lens),
                sum(gen_lens) / len(gen_lens),
            )

            token_masks = None
            if teacher_forcing and evidence_start_column is not None:
                assert recorder.tokenizer is not None, "tokenizer must be loaded"
                evidence_start_chars = [
                    int(evidence_start_column[i]) for i in batch_order
                ]
                token_masks = _build_teacher_forcing_token_masks(
                    recorder.tokenizer,
                    prompts,
                    evidence_start_chars,
                )
                kept_counts = [sum(mask) for mask in token_masks]
                logger.info(
                    "  teacher-forcing evidence mask: kept=%s "
                    "(min=%d, max=%d, avg=%.1f)",
                    kept_counts,
                    min(kept_counts),
                    max(kept_counts),
                    sum(kept_counts) / len(kept_counts),
                )
            if math500_boxed_token_mask and dataset_id == "math500":
                assert recorder.tokenizer is not None, "tokenizer must be loaded"
                token_masks = []
                token_counts = []
                hits = 0
                generated_ids = result.generated_token_ids or [
                    [] for _ in range(result.batch_size)
                ]
                for token_ids in generated_ids:
                    mask = build_boxed_token_mask(token_ids, recorder.tokenizer)
                    token_masks.append(mask)
                    keep_n = sum(mask)
                    token_counts.append(keep_n)
                    if keep_n > 0:
                        hits += 1

                empty = len(token_counts) - hits
                avg_kept = (
                    sum(token_counts) / len(token_counts) if token_counts else 0.0
                )
                logger.info(
                    "  math500 boxed-token mask: hit=%d/%d, empty=%d, avg_kept=%.2f",
                    hits,
                    len(token_counts),
                    empty,
                    avg_kept,
                )

            t0 = time.perf_counter()
            n = recorder.stream_evidence(
                result,
                accumulator,
                domain,
                token_masks=token_masks,
            )
            dt_evi = time.perf_counter() - t0
            t_evi_total += dt_evi
            n_processed += n
            logger.info("  evidence  %.2fs (%d samples)", dt_evi, n)

            del result

            logger.info(
                "  progress %d/%d (%.1f%%)",
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


def _save_results(
    run_dir: str,
    output_path: str,
    accumulator: OnlineRACEAccumulator,
    *,
    model_name: str,
    dataset_id: str,
    domain: str,
    race_cfg: OnlineRACEConfig,
    run_args: Dict[str, Any],
    stats: Dict[str, Any],
    t_total: float,
) -> str:
    t0 = time.perf_counter()

    logger.info("Saving domain '%s' to H5 ...", domain)
    accumulator.save_domain_to_h5(output_path, domain)
    accumulator.finalize_h5_file(output_path)

    meta = {
        "model_name": model_name,
        "dataset": dataset_id,
        "domain": domain,
        "timestamp": os.path.basename(run_dir),
        "config": {
            "mu_0": race_cfg.mu_0,
            "lambda_0": race_cfg.lambda_0,
            "alpha_0": race_cfg.alpha_0,
            "beta_0": race_cfg.beta_0,
            "gamma": race_cfg.gamma,
            **run_args,
        },
        "results": {
            "total_samples": stats["total"],
            "processed": stats["processed"],
            "unique_domains": len(accumulator.state.domains),
        },
        "timing": {
            "total_s": round(t_total, 2),
            "inference_s": round(stats["elapsed"], 2),
            "forward_s": round(stats["forward_s"], 2),
            "evidence_s": round(stats["evidence_s"], 2),
            "other_s": round(stats["other_s"], 2),
            "throughput": round(stats["throughput"], 2),
        },
        "environment": {
            "device": race_cfg.device,
            "model_device_map": run_args.get("model_device_map"),
            "input_device": run_args.get("input_device"),
            "activation_device": run_args.get("activation_device"),
            "torch_version": torch.__version__,
            "cuda_devices": (
                [
                    torch.cuda.get_device_name(i)
                    for i in range(torch.cuda.device_count())
                ]
                if torch.cuda.is_available()
                else None
            ),
        },
        "output_h5": os.path.basename(output_path),
    }

    json_path = os.path.join(run_dir, "run_config.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    logger.info("  H5   -> %s", output_path)
    logger.info("  JSON -> %s", json_path)
    logger.info("  (%.1fs)", time.perf_counter() - t0)
    return output_path


def run_online_race_analysis(
    *,
    model_name: str,
    output_dir: str,
    dataset: str,
    dataset_name: Optional[str] = None,
    dataset_split: Optional[str] = None,
    max_samples: Optional[int] = None,
    random_sample: bool = False,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
    config: Optional[OnlineRACEConfig] = None,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    do_sample: Optional[bool] = None,
    top_p: Optional[float] = None,
    prompt_split: str = "instruct",
    instruction_prefix: Optional[str] = None,
    response_prefix: Optional[str] = None,
    prefill: bool = True,
    batch_size: int = 1,
    first_token_only: bool = False,
    sort_by_length: bool = True,
    teacher_forcing: Optional[bool] = None,
    math500_boxed_token_mask: bool = False,
    evalscope_results_dir: Optional[str] = None,
    evalscope_max_evidence_tokens: Optional[int] = None,
    evalscope_max_evidence_chars: Optional[int] = None,
    evalscope_evidence_scope: str = "all",
    model_device_map: Any = "cuda",
    input_device: Optional[str] = None,
    activation_device: str = "cuda",
    compile_model: bool = True,
) -> str:
    ensure_builtin_datasets_registered()
    spec = get_dataset_spec(dataset)
    defaults = spec.defaults
    generation_source = "model_generation_config"

    if evalscope_results_dir is not None and first_token_only:
        logger.warning(
            "first_token_only=True ignored because evalscope_results_dir uses "
            "teacher-forcing over prompt + assistant output"
        )
        first_token_only = False
    requested_max_new_tokens = max_new_tokens
    if first_token_only:
        max_new_tokens = 2
    instruction_prefix = (
        defaults["instruction_prefix"]
        if instruction_prefix is None
        else instruction_prefix
    )
    response_prefix = (
        defaults["response_prefix"] if response_prefix is None else response_prefix
    )

    domain = spec.dataset_id  # per requirement: single domain == dataset name

    # Resolve teacher_forcing: CLI override > spec default
    if teacher_forcing is None:
        teacher_forcing = spec.teacher_forcing
    if evalscope_results_dir is not None:
        teacher_forcing = True
    if teacher_forcing and first_token_only:
        raise ValueError(
            "teacher_forcing=True and first_token_only=True are mutually exclusive: "
            "teacher-forcing runs a prefill-only forward pass over the full input and "
            "never generates tokens, so first_token_only has no effect. "
            "Use teacher_forcing=True to analyse the full prompt, or "
            "first_token_only=True (with teacher_forcing=False) to restrict evidence "
            "to the first generated token."
        )
    if math500_boxed_token_mask and spec.dataset_id != "math500":
        logger.warning(
            "math500_boxed_token_mask=True ignored for dataset=%s", spec.dataset_id
        )
        math500_boxed_token_mask = False
    if math500_boxed_token_mask and teacher_forcing:
        logger.warning(
            "math500_boxed_token_mask=True requires generated token ids; "
            "disabling for teacher-forcing mode"
        )
        math500_boxed_token_mask = False

    t0_total = time.perf_counter()
    race_cfg = config or OnlineRACEConfig()

    logger.info("=" * 70)
    logger.info("  LLM Online RACE Analysis")
    logger.info("=" * 70)
    logger.info(
        "  dataset=%s  domain=%s  model=%s  batch=%d  "
        "max_tokens=%s  first_token_only=%s  model_device_map=%s",
        spec.dataset_id,
        domain,
        model_name,
        batch_size,
        "model" if max_new_tokens is None else str(max_new_tokens),
        first_token_only,
        model_device_map,
    )
    if evalscope_results_dir is not None:
        evidence_mode = "evalscope-teacher-forcing"
    elif teacher_forcing:
        evidence_mode = "teacher-forcing"
    elif first_token_only:
        evidence_mode = "first-token-only"
    else:
        evidence_mode = "all-generated"
    if first_token_only:
        logger.info(
            "  first_token_only=True: max_new_tokens will be forced to 2 "
            "(requested %s ignored)",
            (
                "model"
                if requested_max_new_tokens is None
                else str(requested_max_new_tokens)
            ),
        )
    logger.info(
        "  generation: source=%s  temp=%s  do_sample=%s  top_p=%s  evidence=%s",
        generation_source,
        "model" if temperature is None else f"{temperature:.2f}",
        "model" if do_sample is None else str(do_sample),
        "model" if top_p is None else f"{top_p:.2f}",
        evidence_mode,
    )
    logger.info("  math500_boxed_token_mask=%s", math500_boxed_token_mask)
    if evalscope_results_dir is not None:
        logger.info("  evalscope_results_dir=%s", evalscope_results_dir)
        logger.info(
            "  evalscope evidence caps: max_chars=%s  max_tokens=%s",
            (
                "none"
                if evalscope_max_evidence_chars is None
                else str(evalscope_max_evidence_chars)
            ),
            (
                "none"
                if evalscope_max_evidence_tokens is None
                else str(evalscope_max_evidence_tokens)
            ),
        )
        logger.info("  evalscope evidence scope=%s", evalscope_evidence_scope)
    logger.info(
        "  RACE: mu_0=%.1f  lambda_0=%.1f  alpha_0=%.1f  beta_0=%.1f  gamma=%.3f",
        race_cfg.mu_0,
        race_cfg.lambda_0,
        race_cfg.alpha_0,
        race_cfg.beta_0,
        race_cfg.gamma,
    )
    if max_samples is not None:
        logger.info("  max_samples=%d", max_samples)
    logger.info("  random_sample=%s  sample_seed=%d", random_sample, sample_seed)
    logger.info("  output_dir=%s", output_dir)
    logger.info("=" * 70)

    logger.info("[1/5] Setting up recorder ...")
    recorder = _setup_recorder(
        model_name,
        output_dir,
        first_token_only=first_token_only,
        model_device_map=model_device_map,
        input_device=input_device,
        activation_device=activation_device,
        compile_model=compile_model,
    )

    logger.info("[2/5] Loading dataset ...")
    if evalscope_results_dir is None:
        dataset_obj = _prepare_dataset(
            recorder,
            dataset_id=spec.dataset_id,
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            max_samples=max_samples,
            random_sample=random_sample,
            sample_seed=sample_seed,
            prompt_split=prompt_split,
            instruction_prefix=instruction_prefix,
            response_prefix=response_prefix,
            prefill=prefill,
        )
    else:
        dataset_obj = _prepare_evalscope_dataset(
            recorder,
            dataset_id=spec.dataset_id,
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            max_samples=max_samples,
            random_sample=random_sample,
            sample_seed=sample_seed,
            prompt_split=prompt_split,
            instruction_prefix=instruction_prefix,
            response_prefix=response_prefix,
            results_dir=evalscope_results_dir,
            max_evidence_tokens=evalscope_max_evidence_tokens,
            max_evidence_chars=evalscope_max_evidence_chars,
            evidence_scope=evalscope_evidence_scope,
        )
    if len(dataset_obj) == 0:
        logger.warning("No samples to process")
        return ""

    logger.info("[3/5] Extracting weights & initialising accumulator ...")
    layer_weights = recorder.extract_layer_weights()
    accumulator = OnlineRACEAccumulator(
        model_name=model_name,
        layer_weights=layer_weights,
        config=race_cfg,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    h5_name = f"{spec.dataset_id}_online_race.h5"
    output_path = accumulator.init_h5_file(os.path.join(run_dir, h5_name))
    logger.info("[4/5] H5 initialised: %s", output_path)

    logger.info(
        "[5/5] Processing %d samples (batch_size=%d) ...", len(dataset_obj), batch_size
    )
    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": do_sample,
        "top_p": top_p,
    }
    stats = _process_batches(
        recorder,
        dataset_obj,
        accumulator,
        domain=domain,
        dataset_id=spec.dataset_id,
        batch_size=batch_size,
        gen_kwargs=gen_kwargs,
        sort_by_length=sort_by_length,
        teacher_forcing=teacher_forcing,
        math500_boxed_token_mask=math500_boxed_token_mask,
    )

    recorder.remove_hooks()

    t_total = time.perf_counter() - t0_total
    run_args = {
        "dataset": spec.dataset_id,
        "dataset_name": dataset_name,
        "dataset_split": dataset_split,
        "max_samples": max_samples,
        "random_sample": random_sample,
        "sample_seed": sample_seed,
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "do_sample": do_sample,
        "top_p": top_p,
        "prompt_split": prompt_split,
        "prefill": prefill,
        "first_token_only": first_token_only,
        "sort_by_length": sort_by_length,
        "teacher_forcing": teacher_forcing,
        "math500_boxed_token_mask": math500_boxed_token_mask,
        "evalscope_results_dir": evalscope_results_dir,
        "evalscope_max_evidence_tokens": evalscope_max_evidence_tokens,
        "evalscope_max_evidence_chars": evalscope_max_evidence_chars,
        "evalscope_evidence_scope": evalscope_evidence_scope,
        "model_device_map": model_device_map,
        "input_device": str(recorder.device),
        "activation_device": activation_device,
        "compile_model": compile_model,
    }
    _save_results(
        run_dir,
        output_path,
        accumulator,
        model_name=model_name,
        dataset_id=spec.dataset_id,
        domain=domain,
        race_cfg=race_cfg,
        run_args=run_args,
        stats=stats,
        t_total=t_total,
    )

    logger.info("-" * 70)
    logger.info(
        "Done in %.1fs  (fwd %.1fs, evi %.1fs, other %.1fs)  %.1f samples/s",
        t_total,
        stats["forward_s"],
        stats["evidence_s"],
        stats["other_s"],
        stats["throughput"],
    )
    logger.info("Output: %s", run_dir)
    logger.info("-" * 70)

    return output_path


def main() -> None:
    ensure_builtin_datasets_registered()

    parser = argparse.ArgumentParser(description="Run unified online RACE for LLMs")

    parser.add_argument(
        "--dataset",
        default="math500",
        choices=[
            "math500",
            "mbpp_plus",
            "gpqa_diamond",
            "mmlu_redux",
            "competition_math",
            "apps",
            "code_alpaca",
            "wikitext2",
            "fineweb",
            "py_comprehension_statements",
        ],
        help="Dataset adapter to use",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Override Hugging Face dataset name (passed to loader if supported)",
    )
    parser.add_argument(
        "--dataset-split",
        default=None,
        help="Override Hugging Face split/version (passed to loader if supported)",
    )

    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        help="Hugging Face model identifier",
    )
    parser.add_argument(
        "--model-device-map",
        default="cuda",
        help=(
            "Hugging Face device_map for model loading. Use 'auto' to shard "
            "large models across visible GPUs."
        ),
    )
    parser.add_argument(
        "--input-device",
        default=None,
        help=(
            "Device for tokenized inputs. Defaults to the model input embedding "
            "device, which is usually cuda:0 for sharded models."
        ),
    )
    parser.add_argument(
        "--activation-device",
        default="cuda",
        help="Device for captured activations before accumulation (e.g. cuda or cpu).",
    )
    parser.add_argument(
        "--accumulator-device",
        default="cuda",
        help=(
            "Device used for RACE evidence accumulation. Use cpu to reduce GPU "
            "memory when loading very large sharded models."
        ),
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile for the loaded HF model.",
    )
    parser.add_argument(
        "--output-dir",
        default="result/llm/online_race",
        help="Root output directory for RACE results",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--random-sample",
        action="store_true",
        help=(
            "Randomly select --max-samples examples with --sample-seed instead "
            "of taking the dataset prefix."
        ),
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=DEFAULT_SAMPLE_SEED,
        help="Seed used by --random-sample (default: 42).",
    )

    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument(
        "--do-sample",
        action="store_true",
        default=None,
        help="Override the model generation config to enable sampling.",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Override the model generation config with do_sample=False and temperature=0.0.",
    )
    parser.add_argument("--top-p", type=float, default=None)

    parser.add_argument(
        "--prompt-split",
        default="instruct",
        choices=["instruct", "complete"],
        help="Prompt style (some datasets may ignore)",
    )
    parser.add_argument("--instruction-prefix", default=None)
    parser.add_argument("--response-prefix", default=None)
    parser.add_argument("--no-prefill", action="store_true")

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--first-token-only", action="store_true")
    parser.add_argument(
        "--math500-boxed-token-mask",
        action="store_true",
        help="For math500, restrict RACE evidence to generated tokens inside "
        "the final \\boxed{...} answer.",
    )
    parser.add_argument("--no-sort-by-length", action="store_true")
    parser.add_argument(
        "--teacher-forcing",
        action="store_true",
        default=None,
        help="Use teacher-forcing (prompt-only) mode instead of autoregressive "
        "generation. Activations are captured over the full input sequence. "
        "Enabled by default for wikitext2 and fineweb; use this flag to enable "
        "for any dataset.",
    )
    parser.add_argument(
        "--evalscope-results-dir",
        default=None,
        help="Use EvalScope JSONL predictions as the evidence source. The "
        "pipeline reconstructs each dataset prompt, appends the matched "
        "assistant output, and runs teacher-forcing over the concatenated text.",
    )
    parser.add_argument(
        "--evalscope-max-evidence-tokens",
        type=int,
        default=None,
        help="Limit each appended EvalScope assistant output to the first N "
        "tokens before teacher-forcing. Does not truncate the prompt.",
    )
    parser.add_argument(
        "--evalscope-max-evidence-chars",
        type=int,
        default=None,
        help="Limit each appended EvalScope assistant output to the first N "
        "characters before teacher-forcing. Applied before token truncation.",
    )
    parser.add_argument(
        "--evalscope-evidence-scope",
        choices=["all", "assistant"],
        default="all",
        help=(
            "Choose which replay tokens contribute RACE evidence. 'assistant' "
            "keeps only the appended EvalScope generation; 'all' includes "
            "both prompt and assistant tokens."
        ),
    )

    parser.add_argument("--gamma", type=float, default=0.05)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.greedy:
        args.do_sample = False
        args.temperature = 0.0
        logger.info("Greedy decoding: do_sample=False, temperature=0.0")

    os.makedirs(args.output_dir, exist_ok=True)

    race_config = OnlineRACEConfig(
        gamma=args.gamma, device=args.accumulator_device, model_family="llm"
    )

    run_online_race_analysis(
        model_name=args.model,
        output_dir=args.output_dir,
        dataset=args.dataset,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        max_samples=args.max_samples,
        random_sample=args.random_sample,
        sample_seed=args.sample_seed,
        config=race_config,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample,
        top_p=args.top_p,
        prompt_split=args.prompt_split,
        instruction_prefix=args.instruction_prefix,
        response_prefix=args.response_prefix,
        prefill=not args.no_prefill,
        batch_size=args.batch_size,
        first_token_only=args.first_token_only,
        sort_by_length=not args.no_sort_by_length,
        teacher_forcing=args.teacher_forcing,
        math500_boxed_token_mask=args.math500_boxed_token_mask,
        evalscope_results_dir=args.evalscope_results_dir,
        evalscope_max_evidence_tokens=args.evalscope_max_evidence_tokens,
        evalscope_max_evidence_chars=args.evalscope_max_evidence_chars,
        evalscope_evidence_scope=args.evalscope_evidence_scope,
        model_device_map=args.model_device_map,
        input_device=args.input_device,
        activation_device=args.activation_device,
        compile_model=not args.no_compile,
    )


if __name__ == "__main__":
    main()
