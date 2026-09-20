#!/usr/bin/env bash
set -euo pipefail

# Qwen3-4B RACE knowledge-QA intervention suite:
#   scoring set: MMLU-Redux (57 subjects / 5,700 examples)
#   held-out validation: GPQA Diamond + ARC
#   method: RACE (positive NIG lower-confidence bound / lcb_pos)
#   RSF reference: WikiText-2; exclude its top 1%, then select target top 1%
#   modules: ATTN and MLP, suppress residual top-1% neurons per layer
#   benchmark evaluation: scripts/eval_full_list.py, three repeats
#
# Run all stages:
#   bash scripts/run_mmlu_redux_knowledge_qa_full.sh
# Run or resume selected stages:
#   STAGES="models eval" bash scripts/run_mmlu_redux_knowledge_qa_full.sh
# Resume evaluation from a cache after interruption:
#   EVAL_USE_CACHE=outputs/mmlu_redux_knowledge_qa_rsf_wikitext2_top1pct \
#     STAGES=eval bash scripts/run_mmlu_redux_knowledge_qa_full.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
PREDICTIONS_DIR="${PREDICTIONS_DIR:-}"
GPU_CSV="${GPU_CSV:-0}"
RACE_GPU="${RACE_GPU:-0}"
STAGES="${STAGES:-all}"

RACE_ROOT="${RACE_ROOT:-result/llm/online_race/mmlu_redux_knowledge_qa}"
GENERAL_RACE_H5="${GENERAL_RACE_H5:-}"
GENERAL_CONCEPT="${GENERAL_CONCEPT:-wikitext2}"

MODEL_ROOT="${MODEL_ROOT:-results/llm/perturbed_models/mmlu_redux_knowledge_qa}"
RACE_MODEL_ROOT="$MODEL_ROOT/race_rsf_wikitext2_top1pct"
EVAL_ROOT="${EVAL_ROOT:-outputs/mmlu_redux_knowledge_qa_rsf_wikitext2_top1pct}"
EVAL_USE_CACHE="${EVAL_USE_CACHE:-$EVAL_ROOT}"

has_stage() {
    [[ " $STAGES " == *" all "* || " $STAGES " == *" $1 "* ]]
}

latest_race_h5() {
    # ``run_config.json`` is written only after the pipeline finishes.  This
    # excludes timestamp directories left behind by interrupted runs.
    while IFS= read -r h5_path; do
        if [[ -f "$(dirname "$h5_path")/run_config.json" ]]; then
            printf '%s\n' "$h5_path"
        fi
    done < <(
        find "$RACE_ROOT" -type f -name 'mmlu_redux_online_race.h5' 2>/dev/null \
            | sort
    ) | tail -n 1
}

if has_stage race; then
    if [[ -z "$PREDICTIONS_DIR" ]]; then
        echo "PREDICTIONS_DIR is required for the replay-based race stage." >&2
        exit 1
    fi
    if [[ -n "$(latest_race_h5)" ]]; then
        echo "RACE H5 already exists; skipping scoring."
    else
        CUDA_VISIBLE_DEVICES="$RACE_GPU" uv run --no-sync python \
            -m race.llm.pipelines.online_race \
            --dataset mmlu_redux \
            --model "$MODEL" \
            --output-dir "$RACE_ROOT" \
            --batch-size 2 \
            --no-compile \
            --evalscope-results-dir "$PREDICTIONS_DIR" \
            --evalscope-evidence-scope assistant \
            --evalscope-max-evidence-tokens 2048
    fi
fi

save_model_if_missing() {
    local h5_path=$1
    local metric=$2
    local module=$3
    local output_root=$4
    local expected="$output_root/$(basename "$MODEL")--suppress_mmlu_redux_${metric}_top1.0_${module}_rsf"
    if [[ -f "$expected/config.json" ]]; then
        echo "Checkpoint exists, skipping: $expected"
        return
    fi

    local -a args=(
        --model "$MODEL"
        --race-h5 "$h5_path"
        --concept mmlu_redux
        --operation suppress
        --top-k-percent 1.0
        --metric "$metric"
        --modules "$module"
        --general-race-h5 "$GENERAL_RACE_H5"
        --general-concept "$GENERAL_CONCEPT"
        --output-dir "$output_root"
    )
    uv run --no-sync python -m race_eval.llm.pipelines.save_perturbed_model "${args[@]}"
}

if has_stage models; then
    RACE_H5="$(latest_race_h5)"
    if [[ -z "$RACE_H5" ]]; then
        echo "Missing RACE scoring H5; finish the race stage first." >&2
        exit 1
    fi
    if [[ -z "$GENERAL_RACE_H5" || ! -f "$GENERAL_RACE_H5" ]]; then
        echo "Missing WikiText-2 RSF H5: $GENERAL_RACE_H5" >&2
        exit 1
    fi

    for module in attn mlp; do
        save_model_if_missing "$RACE_H5" lcb_pos "$module" "$RACE_MODEL_ROOT"
    done
fi

run_eval() {
    EVAL_USE_CACHE="$EVAL_USE_CACHE" uv run --no-sync python \
        scripts/eval_full_list.py \
        --models-roots "$1" \
        --datasets mmlu_redux gpqa_diamond arc \
        --work-dir "$EVAL_ROOT" \
        --use-cache "$EVAL_USE_CACHE" \
        --cuda-visible-devices "$GPU_CSV" \
        --repeats 3 \
        --seed 42 \
        "${@:2}"
}

if has_stage eval; then
    run_eval "$MODEL"
    run_eval "$RACE_MODEL_ROOT"
fi

echo "Qwen3-4B MMLU-Redux RACE suite completed."
