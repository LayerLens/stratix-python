#!/usr/bin/env bash
# Run an evaluation and wait for results
set -euo pipefail

MODEL="${1:-openai/gpt-4o}"
BENCHMARK="${2:-arc-agi-2}"

echo "==> Running evaluation: $MODEL on $BENCHMARK"
layerlens evaluate run \
  --model "$MODEL" \
  --benchmark "$BENCHMARK" \
  --wait

echo "==> Recent evaluations:"
layerlens evaluate list --sort-by submitted_at --order desc --page-size 5
