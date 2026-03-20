#!/usr/bin/env bash
# Judge workflow: create, test, review
#
# Usage: ./03_judge_workflow.sh <TRACE_ID> [MODEL_ID]
set -euo pipefail

TRACE_ID="${1:?Usage: $0 <TRACE_ID> [MODEL_ID]}"
MODEL_ID="${2:-67e1fe69e014f9fa6e50d7be}"

# Create a judge (capture ID from the "Judge created: <id>" line)
echo "==> Creating judge..."
JUDGE_ID=$(stratix judge create \
  --name "Response Quality $(date +%s)" \
  --goal "Rate the response for accuracy, completeness, and clarity on a 1-5 scale" \
  --model-id "$MODEL_ID" \
  | grep "^Judge created:" | awk '{print $NF}')
echo "==> Judge ID: $JUDGE_ID"

# Test against a trace
echo "==> Testing judge..."
stratix judge test --judge-id "$JUDGE_ID" --trace-id "$TRACE_ID"

# Review judge details
echo "==> Judge details:"
stratix judge get "$JUDGE_ID"
