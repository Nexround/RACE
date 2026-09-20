#!/bin/bash
# Minimal LLM-RACE example

set -e  # Exit on error

echo "=========================================="
echo "LLM-RACE Example Usage"
echo "=========================================="

# Configuration
MODEL="Qwen/Qwen2.5-Coder-1.5B-Instruct"
OUTPUT_DIR="result/llm/math500"
MAX_SAMPLES=10  # Small test run

uv run python -m race.llm.pipelines.online_race \
    --dataset math500 \
    --model "$MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --max-samples $MAX_SAMPLES

echo ""
echo "=========================================="
echo "✓ RACE analysis complete!"
echo "  Check output in: $OUTPUT_DIR"
echo "=========================================="
