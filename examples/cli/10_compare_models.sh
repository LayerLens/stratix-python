#!/usr/bin/env bash
# Compare multiple models on the same benchmark
#
# Usage: ./10_compare_models.sh [BENCHMARK] [MODEL1] [MODEL2] ...
set -euo pipefail

BENCHMARK="${1:-arc-agi-2}"
shift 2>/dev/null || true

if [ $# -eq 0 ]; then
  MODELS=("openai/gpt-4o" "openai/gpt-4o-mini")
else
  MODELS=("$@")
fi

echo "==> Comparing ${#MODELS[@]} models on $BENCHMARK"

for model in "${MODELS[@]}"; do
  echo "  Running: $model"
  layerlens evaluate run --model "$model" --benchmark "$BENCHMARK" --wait &
done

# Wait for all background evaluations
wait

echo ""
echo "==> Results (sorted by accuracy):"
layerlens evaluate list --sort-by accuracy --order desc --page-size 10
