#!/usr/bin/env bash
# Scorer lifecycle: create, list, inspect, delete
#
# Usage: ./07_scorer_lifecycle.sh <MODEL_ID>
set -euo pipefail

MODEL_ID="${1:?Usage: $0 <MODEL_ID>}"
SCORER_NAME="CLI Demo $(date +%s)"

# Create (dry-run)
echo "==> Dry-run create:"
stratix scorer create \
  --name "$SCORER_NAME" \
  --description "Evaluates generated code for correctness and readability" \
  --model-id "$MODEL_ID" \
  --prompt "Rate the following code on a 1-10 scale for correctness, readability, and adherence to best practices." \
  --dry-run

# Create for real
echo ""
echo "==> Creating scorer..."
stratix scorer create \
  --name "$SCORER_NAME" \
  --description "Evaluates generated code for correctness and readability" \
  --model-id "$MODEL_ID" \
  --prompt "Rate the following code on a 1-10 scale for correctness, readability, and adherence to best practices."

# Find the scorer by name in the list
echo ""
echo "==> Finding scorer in list..."
SCORER_ID=$(stratix --format json scorer list \
  | python3 -c "import sys,json
for s in json.load(sys.stdin):
    if s['name'] == '$SCORER_NAME':
        print(s['id']); break")
echo "==> Scorer ID: $SCORER_ID"

# Inspect
stratix scorer get "$SCORER_ID"

# Delete
echo ""
echo "==> Cleaning up..."
stratix scorer delete "$SCORER_ID" -y
echo "==> Done."
