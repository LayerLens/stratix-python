#!/bin/bash
# Validate release requirements
# - Checks if the tag matches semantic versioning convention (v{major}.{minor}.{patch} with optional suffix)
# - Checks if the tag matches the version in the package
# - Ensures we're releasing from the release branch

set -e

# Get the tag from the first command line argument
if [ $# -eq 0 ]; then
  echo "ERROR: Release tag argument not provided"
  echo "Usage: $0 <release-tag>"
  exit 1
fi

ROOT_DIR=$(git rev-parse --show-toplevel)

# Fetch the latest tags to ensure we're up to date
git fetch --tags --prune --force

TAG=$1

# Check if tag follows semantic versioning pattern (v{major}.{minor}.{patch} with optional suffix)
if [[ ! "$TAG" =~ ^v[0-9]+\.[0-9]+\.[0-9]+.*$ ]]; then
  echo "ERROR: Tag must follow semantic versioning pattern (v{major}.{minor}.{patch} with optional suffix)"
  echo "Examples: v1.0.0, v2.1.3, v1.0.0-beta, v1.0.0-rc.1"
  exit 1
fi

# Extract version without the 'v' prefix
VERSION=${TAG#v}

PACKAGE_VERSION=$(bash "$ROOT_DIR/scripts/get_version.sh")

# Check if the tag version matches the package version
if [ "$VERSION" != "$PACKAGE_VERSION" ]; then
  echo "ERROR: Tag version ($VERSION) does not match package version ($PACKAGE_VERSION)"
  exit 1
fi

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$CURRENT_BRANCH" != "release" ]; then
  # If we're in detached HEAD state (which is likely in GitHub Actions with a tag),
  # we need to check if the tag is on the release branch
  if ! git rev-parse "$TAG" &>/dev/null; then
    echo "ERROR: Tag $TAG does not exist in the repository"
    exit 1
  fi

  TAG_COMMIT=$(git rev-parse "$TAG")

  # Ensure we have release branch history
  git fetch origin release --depth=1000

  # Check if tag is on release branch
  if ! git merge-base --is-ancestor "$TAG_COMMIT" origin/release; then
    echo "ERROR: Tag $TAG is not on the release branch"
    exit 1
  fi
fi

# All checks passed
exit 0