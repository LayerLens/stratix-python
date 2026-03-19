#!/usr/bin/env bash
# Bulk evaluation from a JSONL file
set -euo pipefail

JOBS_FILE="${1:-/dev/stdin}"

# Create a sample jobs file if none provided
if [ "$JOBS_FILE" = "/dev/stdin" ]; then
  JOBS_FILE=$(mktemp /tmp/layerlens-jobs-XXXXX.jsonl)
  cat > "$JOBS_FILE" <<'EOF'
{"model": "openai/gpt-4o", "benchmark": "arc-agi-2"}
{"model": "openai/gpt-4o-mini", "benchmark": "arc-agi-2"}
EOF
  echo "==> Created sample jobs file: $JOBS_FILE"
fi

# Dry-run first
echo "==> Dry-run:"
layerlens bulk eval --file "$JOBS_FILE" --dry-run

echo ""
read -p "Proceed? [y/N] " confirm
if [[ "$confirm" =~ ^[Yy]$ ]]; then
  layerlens bulk eval --file "$JOBS_FILE" --wait
fi
