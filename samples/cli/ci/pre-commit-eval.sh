#!/usr/bin/env bash
# LayerLens CI — Pre-commit Evaluation Gate
# Runs an evaluation suite and blocks the commit if the pass rate is below threshold.
#
# Installation:
#   cp pre-commit-eval.sh .git/hooks/pre-commit
#   chmod +x .git/hooks/pre-commit
#
# Configuration via environment:
#   LAYERLENS_API_KEY       - Required API key
#   LAYERLENS_JUDGE_ID      - Judge to evaluate with (default: auto-detect from .layerlens.yml)
#   LAYERLENS_DATASET_ID    - Dataset to evaluate against
#   LAYERLENS_PASS_RATE     - Minimum pass rate 0.0-1.0 (default: 0.8)
#   LAYERLENS_API_URL       - API base URL (default: https://api.layerlens.ai)
#
# Requires: curl, python3

set -euo pipefail

LAYERLENS_API_KEY="${LAYERLENS_API_KEY:?Set LAYERLENS_API_KEY to use the pre-commit eval hook}"
BASE_URL="${LAYERLENS_API_URL:-https://api.layerlens.ai}"
AUTH="Authorization: Bearer $LAYERLENS_API_KEY"
CT="Content-Type: application/json"
PASS_THRESHOLD="${LAYERLENS_PASS_RATE:-0.8}"
TIMEOUT=300
POLL=5

# Try to load config from .layerlens.yml if vars not set
if [[ -z "${LAYERLENS_JUDGE_ID:-}" ]] && [[ -f ".layerlens.yml" ]]; then
    LAYERLENS_JUDGE_ID=$(python3 -c "
import yaml, sys
try:
    cfg = yaml.safe_load(open('.layerlens.yml'))
    print(cfg.get('judge_id', ''))
except: pass
" 2>/dev/null || true)
fi

JUDGE_ID="${LAYERLENS_JUDGE_ID:?Set LAYERLENS_JUDGE_ID or configure .layerlens.yml}"
DATASET_ID="${LAYERLENS_DATASET_ID:?Set LAYERLENS_DATASET_ID}"

echo "[layerlens] Running pre-commit evaluation..."
echo "[layerlens]   Judge: $JUDGE_ID  Dataset: $DATASET_ID  Threshold: $PASS_THRESHOLD"

# ── Submit evaluation ─────────────────────────────────────────────────────────
EVAL_RESP=$(curl -sf -X POST -H "$AUTH" -H "$CT" \
    -d "{\"judge_id\": \"$JUDGE_ID\", \"dataset_id\": \"$DATASET_ID\"}" \
    "$BASE_URL/v1/evaluations") || {
    echo "[layerlens] WARNING: Could not submit evaluation. Allowing commit." >&2
    exit 0
}

EVAL_ID=$(echo "$EVAL_RESP" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
echo "[layerlens]   Eval ID: $EVAL_ID"

# ── Poll for completion ───────────────────────────────────────────────────────
ELAPSED=0
while true; do
    if (( ELAPSED >= TIMEOUT )); then
        echo "[layerlens] WARNING: Evaluation timed out. Allowing commit." >&2
        exit 0
    fi
    sleep "$POLL"
    ELAPSED=$((ELAPSED + POLL))

    STATUS=$(curl -sf -H "$AUTH" "$BASE_URL/v1/evaluations/$EVAL_ID" \
        | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])" 2>/dev/null || echo "unknown")

    [[ "$STATUS" == "completed" || "$STATUS" == "failed" ]] && break
done

if [[ "$STATUS" == "failed" ]]; then
    echo "[layerlens] ERROR: Evaluation failed. Blocking commit." >&2
    exit 1
fi

# ── Check pass rate ───────────────────────────────────────────────────────────
RESULT=$(curl -sf -H "$AUTH" "$BASE_URL/v1/evaluations/$EVAL_ID/results")
GATE=$(echo "$RESULT" | python3 -c "
import sys, json
data = json.load(sys.stdin)
scores = [r['score'] for r in data.get('results', []) if 'score' in r]
if not scores:
    print('SKIP 0 0.0')
    sys.exit(0)
passing = sum(1 for s in scores if s >= 0.7)
rate = passing / len(scores)
threshold = float('$PASS_THRESHOLD')
verdict = 'PASS' if rate >= threshold else 'FAIL'
print(f'{verdict} {len(scores)} {rate:.4f}')
")

VERDICT=$(echo "$GATE" | cut -d' ' -f1)
TOTAL=$(echo "$GATE" | cut -d' ' -f2)
RATE=$(echo "$GATE" | cut -d' ' -f3)

echo "[layerlens]   Samples: $TOTAL  Pass rate: $RATE  Threshold: $PASS_THRESHOLD"

if [[ "$VERDICT" == "FAIL" ]]; then
    echo "[layerlens] BLOCKED: Pass rate $RATE is below threshold $PASS_THRESHOLD." >&2
    echo "[layerlens] Review results: ${BASE_URL}/evaluations/${EVAL_ID}" >&2
    exit 1
fi

echo "[layerlens] PASSED. Commit allowed."
exit 0
