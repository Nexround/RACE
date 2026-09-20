"""Unified evaluation corpus loading for PPL / KL divergence experiments.

Two-stage flow:
1. **Generation stage**: Load dataset prompts, format with chat template,
   then either (a) load pre-existing inference results from a ``--results-dir``
   folder, or (b) run sampling (temperature=0.7, top_p=0.8, top_k=20) with
   the *original* (unmodified) model and cache results to a JSONL file on disk.
2. **Tokenization stage**: Concatenate each prompt with the model's own
   generated output and tokenize into ``input_ids`` for teacher-forcing
   evaluation.

Raw-text corpora (e.g. ``wikitext2``) skip the generation stage entirely:
the passage text is fed directly into the model without any prompt wrapper.

Results-dir JSONL format (from ``results/Qwen3-4B-Instruct-2507O/``)::

    {
      "index": 6,                      # original dataset index
      "messages": [{"role": "assistant", "content": "<generation>"}],
      "metadata": { ... }              # corpus-specific ground-truth
    }

File-name patterns per corpus:
- math_500:     ``math_500_*.jsonl``   (multiple level files, merged)
- mbpp_plus:    ``mbpp_plus_*.jsonl``
- gpqa_diamond: ``gpqa_diamond_*.jsonl``
- mmlu_redux:   ``mmlu_redux_*.jsonl``
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import torch
from torch import nn

from race.llm.evalscope_results import load_evalscope_generations

logger = logging.getLogger(__name__)

SUPPORTED_CORPORA = ("math_500", "mbpp_plus", "gpqa_diamond", "mmlu_redux", "wikitext2")

# Corpora that consist of raw prose text and require no model generation.
# For these, the passage text is used directly as the evaluation sequence.
_RAW_TEXT_CORPORA = frozenset({"wikitext2"})

_MAX_NEW_TOKENS: Dict[str, int] = {
    "math_500": 8192,
    "mbpp_plus": 2048,
    "gpqa_diamond": 4096,
    "mmlu_redux": 2048,
}


# ======================================================================
# Dataset loading & prompt formatting
# ======================================================================


def _load_raw_dataset(corpus_name: str, max_samples: Optional[int]):
    """Load the raw HuggingFace dataset for the given corpus."""
    if corpus_name == "math_500":
        from race.llm.dataset.math500 import load_math500

        return load_math500(max_samples=max_samples)
    elif corpus_name == "mbpp_plus":
        from race.llm.dataset.mbpp_plus import load_mbpp_plus

        return load_mbpp_plus(max_samples=max_samples)
    elif corpus_name == "gpqa_diamond":
        from race.llm.dataset.gpqa_diamond import load_gpqa_diamond

        return load_gpqa_diamond(max_samples=max_samples)
    elif corpus_name == "mmlu_redux":
        from race.llm.dataset.mmlu_redux import load_mmlu_redux

        return load_mmlu_redux(max_samples=max_samples)
    elif corpus_name == "wikitext2":
        from race.llm.dataset.wikitext2 import load_wikitext2

        return load_wikitext2(max_samples=max_samples)
    else:
        raise ValueError(
            f"Unknown corpus {corpus_name!r}. Supported: {SUPPORTED_CORPORA}"
        )


def _get_format_fn(corpus_name: str):
    """Return the ``format_prompt_for_model`` function for a corpus."""
    if corpus_name == "math_500":
        from race.llm.dataset.math500 import format_prompt_for_model

        return format_prompt_for_model
    elif corpus_name == "mbpp_plus":
        from race.llm.dataset.mbpp_plus import format_prompt_for_model

        return format_prompt_for_model
    elif corpus_name == "gpqa_diamond":
        from race.llm.dataset.gpqa_diamond import format_prompt_for_model

        return format_prompt_for_model
    elif corpus_name == "mmlu_redux":
        from race.llm.dataset.mmlu_redux import format_prompt_for_model

        return format_prompt_for_model
    elif corpus_name == "wikitext2":
        from race.llm.dataset.wikitext2 import format_prompt_for_model

        return format_prompt_for_model
    else:
        raise ValueError(f"Unknown corpus {corpus_name!r}")


def _format_prompts(
    dataset,
    corpus_name: str,
    tokenizer: Any,
) -> List[str]:
    """Format dataset examples into model-ready prompts (no answers).

    Uses each dataset's ``format_prompt_for_model`` with ``prefill=False``
    so the prompt ends with the generation-prompt marker but contains no
    pre-filled assistant content.
    """
    format_fn = _get_format_fn(corpus_name)
    prompts: List[str] = []
    for example in dataset:
        formatted = format_fn(
            dict(example),
            tokenizer,
            prefill=False,
        )
        prompts.append(formatted["formatted_prompt"])
    return prompts


# ======================================================================
# Greedy generation
# ======================================================================


@torch.inference_mode()
def _generate_batch(
    model: nn.Module,
    tokenizer: Any,
    prompts: List[str],
    max_new_tokens: int,
    batch_size: int,
) -> List[str]:
    """Run sampling on a list of prompts and return generated texts.

    Uses temperature=0.7, top_p=0.8, top_k=20.
    The model is assumed to be the original (unmodified) model with no
    intervention hooks applied.
    """
    device = next(model.parameters()).device
    all_generated: List[str] = []

    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        prompt_len = inputs["input_ids"].shape[1]

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        for i in range(outputs.shape[0]):
            gen_ids = outputs[i, prompt_len:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            all_generated.append(text)

        if (start // batch_size + 1) % 5 == 0 or start + batch_size >= len(prompts):
            logger.info(
                "  Generated %d/%d samples",
                min(start + batch_size, len(prompts)),
                len(prompts),
            )

    return all_generated


# ======================================================================
# Load from pre-existing results dir (e.g. results/Qwen3-4B-Instruct-2507O)
# ======================================================================

# File-name glob patterns for each corpus inside a results dir.
# math_500 is split across multiple "Level N" files; all are merged.
_RESULTS_GLOB: Dict[str, str] = {
    "math_500": "math_500_*.jsonl",
    "mbpp_plus": "mbpp_plus_*.jsonl",
    "gpqa_diamond": "gpqa_diamond_*.jsonl",
    "mmlu_redux": "mmlu_redux_*.jsonl",
}


def _load_from_results_dir(
    results_dir: str,
    corpus_name: str,
    dataset: Any,
) -> Optional[Dict[int, str]]:
    """Scan *results_dir* for JSONL files matching *corpus_name* and return
    a mapping of ``{merged_dataset_index -> generation_text}``.

    Expected JSONL record schema::

        {
          "index": <int>,          # dataset index, or split-local index
          "messages": [
            {"role": "assistant", "content": "<generated text>"}
          ],
          ...
        }

    Returns ``None`` if no matching files are found.
    """
    if not results_dir or not os.path.isdir(results_dir):
        return None

    if corpus_name not in _RESULTS_GLOB:
        return None

    try:
        index_to_gen = load_evalscope_generations(
            results_dir,
            dataset_id=corpus_name,
            dataset=dataset,
        )
    except FileNotFoundError:
        logger.debug(
            "No results files matching %r in %s",
            _RESULTS_GLOB.get(corpus_name),
            results_dir,
        )
        return None
    except (OSError, ValueError) as e:
        logger.warning("Skipping unreadable results dir %s: %s", results_dir, e)
        return None

    if not index_to_gen:
        return None

    logger.info(
        "Loaded %d records from results dir for corpus %r",
        len(index_to_gen),
        corpus_name,
    )
    return index_to_gen


# ======================================================================
# Generation cache (JSONL)
# ======================================================================


def _cache_path(cache_dir: str, model_name: str, corpus_name: str) -> str:
    """Build a deterministic cache file path."""
    safe_model = model_name.rstrip("/").replace("/", "--")
    return os.path.join(cache_dir, f"{safe_model}__{corpus_name}.jsonl")


def _save_generation_cache(
    path: str,
    prompts: List[str],
    generations: List[str],
    model_name: str,
    corpus_name: str,
) -> None:
    """Write generation results to a JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, (p, g) in enumerate(zip(prompts, generations)):
            record = {
                "index": i,
                "prompt": p,
                "generation": g,
                "model": model_name,
                "corpus": corpus_name,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Generation cache saved: %s (%d samples)", path, len(prompts))


def _load_generation_cache(
    path: str,
    expected_count: int,
) -> Optional[List[Dict[str, Any]]]:
    """Load generation results from a JSONL cache file.

    Returns ``None`` if the cache is missing, corrupted, or has a
    sample-count mismatch.
    """
    if not os.path.isfile(path):
        return None

    try:
        records: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Cache file corrupted, will regenerate: %s (%s)", path, e)
        return None

    if len(records) != expected_count:
        logger.info(
            "Cache sample count mismatch (%d vs expected %d), will regenerate: %s",
            len(records),
            expected_count,
            path,
        )
        return None

    logger.info("Loaded generation cache: %s (%d samples)", path, len(records))
    return records


# ======================================================================
# Public API
# ======================================================================


def load_eval_corpus(
    corpus_name: str,
    tokenizer: Any,
    model: Optional[nn.Module] = None,
    model_name: str = "",
    max_samples: int = 1000,
    max_length: int = 4096,
    min_length: int = 32,
    generation_cache_dir: str = "result/llm/generation_cache",
    generation_batch_size: int = 4,
    results_dir: Optional[str] = None,
    prompt_only: bool = False,
) -> List[torch.Tensor]:
    """Load an evaluation corpus with model-generated completions.

    Generation source (in priority order):

    1. **results_dir** — scan for pre-existing JSONL inference results in the
       given folder (e.g. ``results/Qwen3-4B-Instruct-2507O``).  Records are
       matched to dataset samples by the ``index`` field.
    2. **generation_cache_dir** — a previously saved ``--generation-cache-dir``
       JSONL produced by an earlier run of this script.
    3. **Live inference** — run sampling (temperature=0.7, top_p=0.8, top_k=20)
       with *model* and save to *generation_cache_dir*.

    In all cases the prompt is reconstructed from the original dataset using
    each corpus's ``format_prompt_for_model`` (chat template, no prefill), then
    concatenated with the generation for teacher-forcing tokenisation.

    **Raw-text corpora** (e.g. ``"wikitext2"``) skip all three generation steps
    above: passages are tokenised directly without any prompt wrapper or model
    call.

    Args:
        corpus_name: One of ``"math_500"``, ``"mbpp_plus"``, ``"gpqa_diamond"``,
                     ``"mmlu_redux"``, ``"wikitext2"``.
        tokenizer: HuggingFace tokenizer.
        model: The *original* (unmodified) language model for live generation.
               May be ``None`` when *results_dir* or cache is available, and is
               never used for raw-text corpora.
        model_name: Model identifier string (used for cache file key).
        max_samples: Maximum number of raw samples from the dataset.
        max_length: Maximum token length per concatenated sequence.
        min_length: Minimum token length; shorter samples are dropped.
        generation_cache_dir: Directory for storing/reading generation JSONL caches.
        generation_batch_size: Batch size for live generation.
        results_dir: Path to a folder of pre-existing inference JSONL results.
                     When set, skips live inference and cache lookup.

    Returns:
        List of 1-D ``input_ids`` tensors, one per valid sample.
    """
    if corpus_name not in SUPPORTED_CORPORA:
        raise ValueError(
            f"Unknown corpus {corpus_name!r}. Supported: {SUPPORTED_CORPORA}"
        )

    # ------------------------------------------------------------------
    # Fast path: raw-text corpora (e.g. wikitext2) — no generation needed
    # ------------------------------------------------------------------
    if corpus_name in _RAW_TEXT_CORPORA:
        return _load_raw_text_corpus(
            corpus_name=corpus_name,
            tokenizer=tokenizer,
            max_samples=max_samples,
            max_length=max_length,
            min_length=min_length,
        )

    # When reading from results_dir, load all rows first so split-local
    # EvalScope indices (e.g. MATH-500 levels) can be mapped back to merged
    # dataset row positions before max_samples is applied as an output cap.
    use_full_dataset = results_dir and os.path.isdir(results_dir)

    raw_max = None if use_full_dataset else max_samples
    dataset = _load_raw_dataset(corpus_name, raw_max)
    n_total = len(dataset)

    # Build index → (example, formatted_prompt) for all loaded rows
    examples = [dict(dataset[i]) for i in range(n_total)]
    prompts = _format_prompts(dataset, corpus_name, tokenizer)

    # 2. Resolve generations using priority order
    generations: Optional[List[str]] = None

    # In prompt_only mode, skip all generation steps
    if prompt_only:
        if use_full_dataset:
            prompts = prompts[:max_samples] if max_samples else prompts
        generations = [""] * len(prompts)

    # Priority 1: pre-existing results dir
    if not prompt_only and results_dir:
        index_to_gen = _load_from_results_dir(results_dir, corpus_name, dataset)
        if index_to_gen is not None:
            # Build aligned (prompt, generation) pairs ordered by merged dataset
            # row index, then cap to max_samples.
            matched_pairs: List[tuple] = []
            for pos, (example, prompt) in enumerate(zip(examples, prompts)):
                gen = index_to_gen.get(pos)
                if gen is not None:
                    matched_pairs.append((prompt, gen))

            if len(matched_pairs) == 0:
                logger.warning(
                    "No samples matched in results_dir for %s — "
                    "falling back to cache/generation.",
                    corpus_name,
                )
            else:
                if max_samples is not None:
                    matched_pairs = matched_pairs[:max_samples]
                prompts = [p for p, _ in matched_pairs]
                generations = [g for _, g in matched_pairs]
                logger.info(
                    "Using results_dir for %s: %d samples matched (max_samples=%s)",
                    corpus_name,
                    len(generations),
                    max_samples,
                )

    # Priority 2: generation cache
    if not prompt_only and generations is None:
        # Re-slice prompts to max_samples if we loaded full dataset above
        if use_full_dataset:
            prompts = prompts[:max_samples] if max_samples else prompts
            examples = examples[:max_samples] if max_samples else examples
        n_samples = len(prompts)
        cache_file = _cache_path(generation_cache_dir, model_name, corpus_name)
        cached = _load_generation_cache(cache_file, n_samples)
        if cached is not None:
            generations = [r["generation"] for r in cached]

    # Priority 3: live inference
    if not prompt_only and generations is None:
        if model is None:
            raise RuntimeError(
                f"No results_dir, no generation cache found for {corpus_name!r}, "
                "and model=None — cannot generate. "
                "Pass --results-dir or --generation-cache-dir, or provide a model."
            )
        logger.info(
            "Generating completions for %s (%d prompts, temp=0.7/top_p=0.8/top_k=20) ...",
            corpus_name,
            len(prompts),
        )
        max_new_tokens = _MAX_NEW_TOKENS.get(corpus_name, 2048)
        generations = _generate_batch(
            model,
            tokenizer,
            prompts,
            max_new_tokens=max_new_tokens,
            batch_size=generation_batch_size,
        )
        _save_generation_cache(
            cache_file,
            prompts,
            generations,
            model_name,
            corpus_name,
        )

    # 3. Concatenate prompt + generation and tokenize
    input_ids_list: List[torch.Tensor] = []
    skipped = 0

    if prompt_only:
        pairs = [(p, "") for p in prompts]
    else:
        pairs = list(zip(prompts, generations))

    for prompt, generation in pairs:
        if not prompt_only and not generation.strip():
            skipped += 1
            continue
        full_text = prompt if prompt_only else prompt + generation
        if not full_text.strip():
            skipped += 1
            continue

        encoding = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        ids = encoding["input_ids"].squeeze(0)

        if ids.shape[0] < min_length:
            skipped += 1
            continue

        input_ids_list.append(ids)

    logger.info(
        "Corpus %s: %d valid samples (%d skipped, max_length=%d)",
        corpus_name,
        len(input_ids_list),
        skipped,
        max_length,
    )
    return input_ids_list


def _load_raw_text_corpus(
    corpus_name: str,
    tokenizer: Any,
    max_samples: int,
    max_length: int,
    min_length: int,
) -> List[torch.Tensor]:
    """Tokenise a raw-text corpus directly (no prompt/generation split).

    Used for corpora like ``wikitext2`` where passages are evaluated as-is
    without any model generation step.
    """
    dataset = _load_raw_dataset(corpus_name, max_samples)
    input_ids_list: List[torch.Tensor] = []
    skipped = 0

    for example in dataset:
        text = (example.get("text") or "").strip()
        if not text:
            skipped += 1
            continue

        encoding = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        ids = encoding["input_ids"].squeeze(0)

        if ids.shape[0] < min_length:
            skipped += 1
            continue

        input_ids_list.append(ids)

    logger.info(
        "Corpus %s: %d valid samples (%d skipped, max_length=%d)",
        corpus_name,
        len(input_ids_list),
        skipped,
        max_length,
    )
    return input_ids_list
