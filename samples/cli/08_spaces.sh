#!/usr/bin/env bash
# Evaluation spaces: create, list, inspect, delete
set -euo pipefail

# Create a private space (capture ID from output)
echo "==> Creating evaluation space..."
SPACE_ID=$(stratix space create \
  --name "CLI Demo Space" \
  --description "Temporary space for CLI examples" \
  --visibility private \
  | grep -oP '[a-f0-9]{24}' | head -1)
echo "==> Created space: $SPACE_ID"

# List spaces
echo ""
echo "==> All spaces:"
stratix space list

# Get details
echo ""
echo "==> Space details:"
stratix space get "$SPACE_ID"

# Clean up
echo ""
echo "==> Deleting space..."
stratix space delete "$SPACE_ID" -y
echo "==> Done."
