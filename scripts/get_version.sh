#!/bin/bash
# Extract and print the version number from _version.py

set -e

ROOT_DIR=$(git rev-parse --show-toplevel)
VERSION_FILE="$ROOT_DIR/src/layerlens/_version.py"

if [ ! -f "$VERSION_FILE" ]; then
    echo "Error: Version file not found at $VERSION_FILE" >&2
    exit 1
fi

VERSION=$(grep -E '^__version__\s*=' "$VERSION_FILE" | grep -o '".*"' | tr -d '"')

if [ -z "$VERSION" ]; then
    echo "Error: Could not extract version from $VERSION_FILE" >&2
    exit 1
fi

echo "$VERSION"
