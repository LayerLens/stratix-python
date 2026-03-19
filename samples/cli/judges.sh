#!/usr/bin/env bash
# LayerLens CLI — Judge Management
# Demonstrates: list judges, create a new judge, test a judge on sample input.
#
# Usage:
#   export LAYERLENS_API_KEY="ll-..."
#   chmod +x judges.sh
#   ./judges.sh [create|list|test]
#
# Requires: curl, python3

set -euo pipefail

LAYERLENS_API_KEY="${LAYERLENS_API_KEY:?Set LAYERLENS_API_KEY}"
BASE_URL="${LAYERLENS_API_URL:-https://api.layerlens.ai}"
AUTH="Authorization: Bearer $LAYERLENS_API_KEY"
CT="Content-Type: application/json"

COMMAND="${1:-list}"

# ── Helper ────────────────────────────────────────────────────────────────────
api_get()  { curl -sf -H "$AUTH" "$BASE_URL$1" | python3 -m json.tool; }
api_post() { curl -sf -X POST -H "$AUTH" -H "$CT" -d "$2" "$BASE_URL$1" | python3 -m json.tool; }

# ── list: Show all available judges ──────────────────────────────────────────
cmd_list() {
    echo "=== Available Judges ==="
    api_get "/v1/judges"
}

# ── create: Create a new custom judge ────────────────────────────────────────
cmd_create() {
    local name="${2:-cli-sample-judge}"
    local description="${3:-Judge created from CLI sample script}"

    echo "=== Creating Judge: $name ==="
    PAYLOAD=$(python3 -c "
import json
print(json.dumps({
    'name': '$name',
    'description': '$description',
    'criteria': [
        {'name': 'relevance',  'weight': 0.4, 'description': 'Response addresses the query directly'},
        {'name': 'accuracy',   'weight': 0.35,'description': 'Factual correctness of the response'},
        {'name': 'clarity',    'weight': 0.25,'description': 'Response is clear and well-structured'}
    ],
    'model': 'gpt-4o',
    'pass_threshold': 0.7
}))
")
    api_post "/v1/judges" "$PAYLOAD"
}

# ── test: Run a judge against a sample input/output pair ─────────────────────
cmd_test() {
    local judge_id="${2:?Usage: judges.sh test <judge_id>}"

    echo "=== Testing Judge: $judge_id ==="
    PAYLOAD=$(python3 -c "
import json
print(json.dumps({
    'judge_id': '$judge_id',
    'input':  'What is the capital of France?',
    'output': 'The capital of France is Paris, located in the north-central part of the country.',
    'expected': 'Paris'
}))
")
    api_post "/v1/judges/$judge_id/test" "$PAYLOAD"
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
case "$COMMAND" in
    list)   cmd_list ;;
    create) cmd_create "$@" ;;
    test)   cmd_test "$@" ;;
    *)
        echo "Usage: judges.sh [list|create|test]" >&2
        echo "  list              List all judges" >&2
        echo "  create [name]     Create a new judge" >&2
        echo "  test <judge_id>   Test a judge on sample data" >&2
        exit 1
        ;;
esac

echo ""
echo "Done."
