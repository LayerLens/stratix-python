#!/usr/bin/env bash
# LayerLens CLI — Data Export
# Demonstrates: export traces and evaluation results as CSV, JSON, or Parquet.
#
# Usage:
#   export LAYERLENS_API_KEY="ll-..."
#   chmod +x exports.sh
#   ./exports.sh [csv|json|parquet] [--type traces|evaluations] [--output FILE]
#
# Requires: curl, python3

set -euo pipefail

LAYERLENS_API_KEY="${LAYERLENS_API_KEY:?Set LAYERLENS_API_KEY}"
BASE_URL="${LAYERLENS_API_URL:-https://api.layerlens.ai}"
AUTH="Authorization: Bearer $LAYERLENS_API_KEY"

FORMAT="${1:-csv}"
shift || true

# Defaults
EXPORT_TYPE="traces"
OUTPUT_FILE=""

# Parse flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --type)   EXPORT_TYPE="$2"; shift 2 ;;
        --output) OUTPUT_FILE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# Validate format
case "$FORMAT" in
    csv|json|parquet) ;;
    *) echo "ERROR: Format must be csv, json, or parquet. Got: $FORMAT" >&2; exit 1 ;;
esac

# Validate export type
case "$EXPORT_TYPE" in
    traces|evaluations) ;;
    *) echo "ERROR: Type must be traces or evaluations. Got: $EXPORT_TYPE" >&2; exit 1 ;;
esac

# Default output filename
if [[ -z "$OUTPUT_FILE" ]]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_FILE="layerlens_${EXPORT_TYPE}_${TIMESTAMP}.${FORMAT}"
fi

# ── 1. Request export ────────────────────────────────────────────────────────
echo "=== Requesting Export ==="
echo "    Type   : $EXPORT_TYPE"
echo "    Format : $FORMAT"
echo "    Output : $OUTPUT_FILE"

ACCEPT_HEADER="text/csv"
case "$FORMAT" in
    json)    ACCEPT_HEADER="application/json" ;;
    parquet) ACCEPT_HEADER="application/octet-stream" ;;
esac

HTTP_CODE=$(curl -s -o "$OUTPUT_FILE" -w "%{http_code}" \
    -H "$AUTH" \
    -H "Accept: $ACCEPT_HEADER" \
    "$BASE_URL/v1/exports/${EXPORT_TYPE}?format=${FORMAT}")

if [[ "$HTTP_CODE" -lt 200 || "$HTTP_CODE" -ge 300 ]]; then
    echo "ERROR: Export request failed with HTTP $HTTP_CODE." >&2
    cat "$OUTPUT_FILE" >&2
    rm -f "$OUTPUT_FILE"
    exit 1
fi

# ── 2. Report results ────────────────────────────────────────────────────────
FILE_SIZE=$(wc -c < "$OUTPUT_FILE" | tr -d ' ')
echo ""
echo "=== Export Complete ==="
echo "    File : $OUTPUT_FILE"
echo "    Size : ${FILE_SIZE} bytes"

# Show preview for text formats
if [[ "$FORMAT" == "csv" ]]; then
    echo ""
    echo "=== Preview (first 5 lines) ==="
    head -n 5 "$OUTPUT_FILE"
elif [[ "$FORMAT" == "json" ]]; then
    echo ""
    echo "=== Preview (first 3 records) ==="
    python3 -c "
import json, sys
with open('$OUTPUT_FILE') as f:
    data = json.load(f)
items = data if isinstance(data, list) else data.get('data', [])
print(json.dumps(items[:3], indent=2))
"
fi

echo ""
echo "Done. Exported $EXPORT_TYPE to $OUTPUT_FILE."
