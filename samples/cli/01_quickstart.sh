#!/usr/bin/env bash
# Quick start: configure, list traces, inspect, evaluate
#
# Usage: ./01_quickstart.sh [MODEL_ID]
# MODEL_ID defaults to the first available judge model.
set -euo pipefail

MODEL_ID="${1:-67e1fe69e014f9fa6e50d7be}"

# 1. List available traces
echo "==> Listing traces..."
stratix trace list --page-size 5

# 2. Get the first trace ID
TRACE_ID=$(stratix --format json trace list --page-size 1 \
  | python3 -c "import sys,json; print(json.load(sys.stdin)[0]['id'])")
echo "==> First trace: $TRACE_ID"

# 3. Inspect it
stratix trace get "$TRACE_ID"

# 4. Create a judge (capture ID from the "Judge created: <id>" line)
echo "==> Creating judge..."
JUDGE_ID=$(stratix judge create \
  --name "Quick Start Judge $(date +%s)" \
  --goal "Rate whether the response is accurate and helpful" \
  --model-id "$MODEL_ID" \
  | grep "^Judge created:" | awk '{print $NF}')
echo "==> Created judge: $JUDGE_ID"

# 5. Test the judge against the trace
echo "==> Testing judge against trace..."
stratix judge test --judge-id "$JUDGE_ID" --trace-id "$TRACE_ID"
