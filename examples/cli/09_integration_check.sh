#!/usr/bin/env bash
# Check integration health
set -euo pipefail

echo "==> Integrations:"
layerlens integration list

# Test each integration (skip if none found)
echo ""
echo "==> Testing all integrations..."
OUTPUT=$(layerlens --format json integration list 2>&1)
if echo "$OUTPUT" | python3 -c "import sys,json; json.load(sys.stdin)" 2>/dev/null; then
  echo "$OUTPUT" | python3 -c "
import sys, json
for i in json.load(sys.stdin):
    print(i['id'], i.get('name', ''))
" | while read -r id name; do
    echo "  Testing $name ($id)..."
    layerlens integration test "$id" || echo "  FAILED: $name"
  done
else
  echo "  No integrations to test."
fi
