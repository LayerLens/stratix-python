#!/usr/bin/env bash
# LayerLens CLI — Evaluation Lifecycle
# Demonstrates: create evaluation run, poll status, retrieve results.
#
# Usage:
#   export LAYERLENS_API_KEY="ll-..."
#   chmod +x evaluations.sh
#   ./evaluations.sh --judge-id jdg_abc123 --dataset-id ds_xyz789
#
# Requires: curl, python3

set -euo pipefail

LAYERLENS_API_KEY="${LAYERLENS_API_KEY:?Set LAYERLENS_API_KEY}"
BASE_URL="${LAYERLENS_API_URL:-https://api.layerlens.ai}"
AUTH="Authorization: Bearer $LAYERLENS_API_KEY"
CT="Content-Type: application/json"

JUDGE_ID="${1:?Usage: evaluations.sh <judge_id> <dataset_id>}"
DATASET_ID="${2:?Usage: evaluations.sh <judge_id> <dataset_id>}"
POLL_INTERVAL="${POLL_INTERVAL:-10}"
TIMEOUT="${EVAL_TIMEOUT:-600}"

# ── 1. Trigger an evaluation run ─────────────────────────────────────────────
echo "=== Creating evaluation run ==="
echo "    Judge:   $JUDGE_ID"
echo "    Dataset: $DATASET_ID"

EVAL_RESPONSE=$(curl -sf -X POST \
    -H "$AUTH" -H "$CT" \
    -d "{\"judge_id\": \"$JUDGE_ID\", \"dataset_id\": \"$DATASET_ID\"}" \
    "$BASE_URL/v1/evaluations")

EVAL_ID=$(echo "$EVAL_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
echo "    Eval ID: $EVAL_ID"

# ── 2. Poll until completion ─────────────────────────────────────────────────
echo ""
echo "=== Polling evaluation status (timeout=${TIMEOUT}s) ==="
ELAPSED=0
STATUS="pending"

while [[ "$STATUS" != "completed" && "$STATUS" != "failed" ]]; do
    if (( ELAPSED >= TIMEOUT )); then
        echo "ERROR: Evaluation timed out after ${TIMEOUT}s." >&2
        exit 1
    fi

    sleep "$POLL_INTERVAL"
    ELAPSED=$((ELAPSED + POLL_INTERVAL))

    STATUS=$(curl -sf -H "$AUTH" "$BASE_URL/v1/evaluations/$EVAL_ID" \
        | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
    echo "    [${ELAPSED}s] status=$STATUS"
done

if [[ "$STATUS" == "failed" ]]; then
    echo "ERROR: Evaluation failed." >&2
    curl -sf -H "$AUTH" "$BASE_URL/v1/evaluations/$EVAL_ID" | python3 -m json.tool >&2
    exit 1
fi

# ── 3. Retrieve results ──────────────────────────────────────────────────────
echo ""
echo "=== Evaluation Results ==="
curl -sf -H "$AUTH" "$BASE_URL/v1/evaluations/$EVAL_ID/results" | python3 -m json.tool

# ── 4. Summary metrics ───────────────────────────────────────────────────────
echo ""
echo "=== Summary ==="
curl -sf -H "$AUTH" "$BASE_URL/v1/evaluations/$EVAL_ID/results" \
    | python3 -c "
import sys, json
data = json.load(sys.stdin)
scores = [r['score'] for r in data.get('results', []) if 'score' in r]
if scores:
    print(f'  Total samples : {len(scores)}')
    print(f'  Average score : {sum(scores)/len(scores):.4f}')
    print(f'  Min / Max     : {min(scores):.4f} / {max(scores):.4f}')
    passing = sum(1 for s in scores if s >= 0.7)
    print(f'  Pass rate     : {passing}/{len(scores)} ({passing/len(scores):.1%})')
else:
    print('  No scored results found.')
"

echo ""
echo "Done. Evaluation $EVAL_ID completed."
