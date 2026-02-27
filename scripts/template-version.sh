#!/usr/bin/env bash

set -e

VERSION_FILE="src/layerlens/_version.py"

GIT_COMMIT=$(git rev-parse HEAD)

sed_inplace() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "$@"
    else
        sed -i "$@"
    fi
}

# Update git commit hash
sed_inplace "s/__GIT_COMMIT__/$GIT_COMMIT/g" "$VERSION_FILE"

# Get current version
CURRENT_VERSION=$(grep '__version__ = ' "$VERSION_FILE" | cut -d'"' -f2)

# If we're uploading to testpypi, add a run number to the version so we can
# test multiple times.
if [[ "$PYPI_REPO" == "testpypi" ]] && [[ -n "$GITHUB_RUN_NUMBER" ]]; then
    NEW_VERSION="${CURRENT_VERSION}rc${GITHUB_RUN_NUMBER}"
    sed_inplace "s/__version__ = \".*\"/__version__ = \"$NEW_VERSION\"/" "$VERSION_FILE"
fi
