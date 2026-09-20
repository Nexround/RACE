"""Utilities for loading EvalScope JSONL generations.

EvalScope writes per-benchmark prediction files with a local ``index`` field.
For single-file benchmarks this index is the dataset row index.  For split
benchmarks such as MATH-500 levels and MMLU-Redux subjects, the index is local
to the split file and must be mapped back to the merged dataset row.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


EVALSCOPE_RESULTS_GLOBS: Dict[str, str] = {
    "math500": "math_500_*.jsonl",
    "math_500": "math_500_*.jsonl",
    "mbpp_plus": "mbpp_plus_*.jsonl",
    "gpqa_diamond": "gpqa_diamond_*.jsonl",
    "mmlu_redux": "mmlu_redux_*.jsonl",
}


def canonical_dataset_id(dataset_id: str) -> str:
    """Return the shared EvalScope helper's canonical dataset id."""
    if dataset_id == "math_500":
        return "math500"
    return dataset_id


def extract_evalscope_generation(record: Dict[str, Any]) -> str:
    """Extract assistant text from known EvalScope prediction schemas."""
    messages = record.get("messages") or []
    for msg in messages:
        if msg.get("role") == "assistant":
            return msg.get("content") or ""

    try:
        return record["model_output"]["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""


def _column_values(dataset: Any, column: str) -> list[Any]:
    column_names = getattr(dataset, "column_names", None)
    if column_names is not None and column not in column_names:
        raise ValueError(f"Dataset is missing required column {column!r}")

    try:
        return list(dataset[column])
    except (KeyError, TypeError):
        values: list[Any] = []
        for row in dataset:
            if column not in row:
                raise ValueError(f"Dataset is missing required column {column!r}")
            values.append(row[column])
        return values


def _positions_by_column(dataset: Any, column: str) -> Dict[str, list[int]]:
    positions: Dict[str, list[int]] = {}
    for idx, value in enumerate(_column_values(dataset, column)):
        positions.setdefault(str(value), []).append(idx)
    return positions


def _math500_level_key(value: Any) -> str:
    text = str(value)
    match = re.search(r"(\d+)", text)
    return match.group(1) if match else text


def _math500_positions(dataset: Any) -> Dict[str, list[int]]:
    positions: Dict[str, list[int]] = {}
    for idx, value in enumerate(_column_values(dataset, "level")):
        positions.setdefault(_math500_level_key(value), []).append(idx)
    return positions


def _local_positions_for_file(
    dataset_id: str,
    basename: str,
    *,
    level_positions: Optional[Dict[str, list[int]]],
    subject_positions: Optional[Dict[str, list[int]]],
) -> Optional[list[int]]:
    if dataset_id == "math500":
        match = re.match(r"math_500_Level\s+(\d+)\.jsonl$", basename)
        if match is None:
            logger.warning("Skipping unexpected math500 EvalScope file: %s", basename)
            return []
        assert level_positions is not None
        return level_positions.get(match.group(1), [])

    if dataset_id == "mmlu_redux":
        match = re.match(r"mmlu_redux_(.+)\.jsonl$", basename)
        if match is None:
            logger.warning(
                "Skipping unexpected mmlu_redux EvalScope file: %s", basename
            )
            return []
        assert subject_positions is not None
        return subject_positions.get(match.group(1), [])

    return None


def load_evalscope_generations(
    results_dir: str,
    *,
    dataset_id: str,
    dataset: Any,
) -> Dict[int, str]:
    """Return ``{merged_dataset_index: assistant_generation}`` from EvalScope.

    The returned indices always refer to positions in ``dataset``.  For MATH-500
    files named ``math_500_Level N.jsonl``, each record's ``index`` is treated as
    the local row within level ``N``.  For MMLU-Redux files, ``index`` is local
    to the subject encoded in the file name.  Other supported corpora use
    ``index`` directly as a merged dataset row index.
    """
    dataset_id = canonical_dataset_id(dataset_id)
    pattern = EVALSCOPE_RESULTS_GLOBS.get(dataset_id)
    if pattern is None:
        raise ValueError(
            f"EvalScope evidence is not configured for dataset {dataset_id!r}. "
            f"Supported: {', '.join(sorted(EVALSCOPE_RESULTS_GLOBS))}"
        )

    matched_files = sorted(glob.glob(os.path.join(results_dir, pattern)))
    if not matched_files:
        raise FileNotFoundError(
            f"No EvalScope JSONL files matching {pattern!r} in {results_dir!r}"
        )

    level_positions = _math500_positions(dataset) if dataset_id == "math500" else None
    subject_positions = (
        _positions_by_column(dataset, "subject") if dataset_id == "mmlu_redux" else None
    )

    index_to_generation: Dict[int, str] = {}
    n_records = 0
    n_skipped = 0

    for fpath in matched_files:
        basename = os.path.basename(fpath)
        local_positions = _local_positions_for_file(
            dataset_id,
            basename,
            level_positions=level_positions,
            subject_positions=subject_positions,
        )
        if local_positions == [] and dataset_id in {"math500", "mmlu_redux"}:
            continue

        with open(fpath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                n_records += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("%s:%d invalid JSON: %s", fpath, line_num, exc)
                    n_skipped += 1
                    continue

                raw_idx = record.get("index")
                generation = extract_evalscope_generation(record)
                if raw_idx is None or not generation.strip():
                    n_skipped += 1
                    continue

                raw_idx = int(raw_idx)
                if local_positions is not None:
                    if raw_idx < 0 or raw_idx >= len(local_positions):
                        n_skipped += 1
                        continue
                    dataset_idx = local_positions[raw_idx]
                else:
                    if raw_idx < 0 or raw_idx >= len(dataset):
                        n_skipped += 1
                        continue
                    dataset_idx = raw_idx

                if dataset_idx in index_to_generation:
                    logger.warning(
                        "Duplicate EvalScope generation for dataset index %d; "
                        "keeping the later record from %s",
                        dataset_idx,
                        basename,
                    )
                index_to_generation[dataset_idx] = generation

    logger.info(
        "Loaded EvalScope evidence: %d matched samples from %d records "
        "(%d files, %d skipped)",
        len(index_to_generation),
        n_records,
        len(matched_files),
        n_skipped,
    )
    return index_to_generation
