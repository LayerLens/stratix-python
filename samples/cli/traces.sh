#!/usr/bin/env bash
# LayerLens CLI — Trace Management
# Demonstrates: list, filter, search, tag, and retrieve traces via the REST API.
#
# Usage:
#   export LAYERLENS_API_KEY="ll-..."
#   chmod +x traces.sh
#   ./traces.sh
#
# Requires: curl, python3 (for json pretty-printing)

set -euo pipefail

LAYERLENS_API_KEY="${LAYERLENS_API_KEY:?Set LAYERLENS_API_KEY}"
BASE_URL="${LAYERLENS_API_URL:-https://api.layerlens.ai}"
AUTH_HEADER="Authorization: Bearer $LAYERLENS_API_KEY"

# Helper: pretty-print JSON response and exit on HTTP errors
api_get() {
    local path="$1"
    local response
    response=$(curl -sf -H "$AUTH_HEADER" "$BASE_URL$path") || {
        echo "ERROR: GET $path failed (HTTP error or network issue)." >&2
        return 1
    }
    echo "$response" | python3 -m json.tool
}

# ── 1. List recent traces ────────────────────────────────────────────────────
echo "=== List recent traces (limit=5) ==="
api_get "/v1/traces?limit=5"

# ── 2. Filter by agent ID ────────────────────────────────────────────────────
echo ""
echo "=== Filter by agent (support-agent, limit=3) ==="
api_get "/v1/traces?agent_id=support-agent&limit=3"

# ── 3. Filter by date range ──────────────────────────────────────────────────
YESTERDAY=$(date -u -d "yesterday" +%Y-%m-%dT00:00:00Z 2>/dev/null \
    || date -u -v-1d +%Y-%m-%dT00:00:00Z)  # GNU / BSD fallback
TODAY=$(date -u +%Y-%m-%dT23:59:59Z)

echo ""
echo "=== Traces from the last 24 hours ==="
api_get "/v1/traces?start=$YESTERDAY&end=$TODAY&limit=5"

# ── 4. Search traces by keyword ──────────────────────────────────────────────
SEARCH_TERM="${1:-error}"
echo ""
echo "=== Search traces containing '$SEARCH_TERM' ==="
api_get "/v1/traces?search=$SEARCH_TERM&limit=5"

# ── 5. Retrieve a single trace by ID ─────────────────────────────────────────
echo ""
echo "=== Retrieve first trace from recent list ==="
TRACE_ID=$(curl -sf -H "$AUTH_HEADER" "$BASE_URL/v1/traces?limit=1" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null) || {
    echo "WARN: Could not extract trace ID; skipping single-trace fetch." >&2
    exit 0
}
api_get "/v1/traces/$TRACE_ID"

# ── 6. Tag a trace ───────────────────────────────────────────────────────────
echo ""
echo "=== Tag trace $TRACE_ID with 'reviewed' ==="
curl -sf -X PATCH \
    -H "$AUTH_HEADER" \
    -H "Content-Type: application/json" \
    -d '{"tags": ["reviewed", "sample-cli"]}' \
    "$BASE_URL/v1/traces/$TRACE_ID" | python3 -m json.tool

echo ""
echo "Done. All trace operations completed successfully."
