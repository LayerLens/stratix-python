#!/usr/bin/env bash
# LayerLens CLI — Trace Replay
# Demonstrates: trigger replay of a trace, compare original vs. replayed output.
#
# Usage:
#   export LAYERLENS_API_KEY="ll-..."
#   chmod +x replay.sh
#   ./replay.sh <trace_id> [--model gpt-4o]
#
# Requires: curl, python3

set -euo pipefail

LAYERLENS_API_KEY="${LAYERLENS_API_KEY:?Set LAYERLENS_API_KEY}"
BASE_URL="${LAYERLENS_API_URL:-https://api.layerlens.ai}"
AUTH="Authorization: Bearer $LAYERLENS_API_KEY"
CT="Content-Type: application/json"

TRACE_ID="${1:?Usage: replay.sh <trace_id> [--model <model>]}"
shift

# Parse optional --model flag
MODEL=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# ── 1. Fetch the original trace ──────────────────────────────────────────────
echo "=== Original Trace ==="
ORIGINAL=$(curl -sf -H "$AUTH" "$BASE_URL/v1/traces/$TRACE_ID") || {
    echo "ERROR: Failed to fetch trace $TRACE_ID." >&2; exit 1
}
echo "$ORIGINAL" | python3 -c "
import sys, json
t = json.load(sys.stdin)
print(f\"  Agent  : {t.get('agent_id', 'N/A')}\")
print(f\"  Model  : {t.get('model', 'N/A')}\")
print(f\"  Status : {t.get('status', 'N/A')}\")
print(f\"  Events : {len(t.get('events', []))}\")
"

# ── 2. Trigger replay ────────────────────────────────────────────────────────
echo ""
echo "=== Triggering Replay ==="
REPLAY_BODY="{\"trace_id\": \"$TRACE_ID\"}"
if [[ -n "$MODEL" ]]; then
    REPLAY_BODY="{\"trace_id\": \"$TRACE_ID\", \"model_override\": \"$MODEL\"}"
    echo "    Model override: $MODEL"
fi

REPLAY=$(curl -sf -X POST -H "$AUTH" -H "$CT" -d "$REPLAY_BODY" \
    "$BASE_URL/v1/replays") || {
    echo "ERROR: Failed to trigger replay." >&2; exit 1
}
REPLAY_ID=$(echo "$REPLAY" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
echo "    Replay ID: $REPLAY_ID"

# ── 3. Poll replay status ────────────────────────────────────────────────────
echo ""
echo "=== Waiting for replay to complete ==="
TIMEOUT=120
ELAPSED=0
POLL=5

while true; do
    if (( ELAPSED >= TIMEOUT )); then
        echo "ERROR: Replay timed out after ${TIMEOUT}s." >&2; exit 1
    fi
    sleep "$POLL"
    ELAPSED=$((ELAPSED + POLL))

    STATUS=$(curl -sf -H "$AUTH" "$BASE_URL/v1/replays/$REPLAY_ID" \
        | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
    echo "    [${ELAPSED}s] status=$STATUS"

    [[ "$STATUS" == "completed" || "$STATUS" == "failed" ]] && break
done

[[ "$STATUS" == "failed" ]] && { echo "ERROR: Replay failed." >&2; exit 1; }

# ── 4. Diff original vs. replay ──────────────────────────────────────────────
echo ""
echo "=== Diff: Original vs. Replay ==="
curl -sf -H "$AUTH" "$BASE_URL/v1/replays/$REPLAY_ID/diff" | python3 -c "
import sys, json
diff = json.load(sys.stdin)
for field, changes in diff.get('differences', {}).items():
    print(f'  {field}:')
    print(f'    original : {changes.get(\"original\", \"N/A\")}')
    print(f'    replayed : {changes.get(\"replayed\", \"N/A\")}')
if not diff.get('differences'):
    print('  No differences detected.')
"

echo ""
echo "Done. Replay $REPLAY_ID completed."
