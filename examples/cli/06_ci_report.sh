#!/usr/bin/env bash
# Generate a CI evaluation report (for GitHub Actions)
set -euo pipefail

OUTPUT="${1:-summary.md}"

echo "==> Generating CI report..."
stratix ci report --limit 10 -o "$OUTPUT"

echo "==> Report written to $OUTPUT"
cat "$OUTPUT"
