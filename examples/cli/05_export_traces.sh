#!/usr/bin/env bash
# Export all traces to individual JSON files
#
# Usage: ./05_export_traces.sh [OUTPUT_DIR]
set -euo pipefail

OUTPUT_DIR="${1:-./exported_traces}"
mkdir -p "$OUTPUT_DIR"

echo "==> Exporting traces to $OUTPUT_DIR/"

stratix --format json trace list | python3 -c "
import sys, json
for t in json.load(sys.stdin):
    print(t['id'])
" | while read -r id; do
  echo "  Exporting $id..."
  stratix trace export "$id" -o "$OUTPUT_DIR/${id}.json"
done

echo "==> Done. Files in $OUTPUT_DIR/"
ls -la "$OUTPUT_DIR/"
