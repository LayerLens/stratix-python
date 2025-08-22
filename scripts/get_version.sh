#!/bin/bash
# Extract and print the version number from _version.py

ROOT_DIR=$(git rev-parse --show-toplevel)

VERSION_FILE="$ROOT_DIR/src/_version.py"

VERSION=$(grep -E '^VERSION\s*=' "$VERSION_FILE" | grep -o '".*"' | tr -d '"')

echo "$VERSION"