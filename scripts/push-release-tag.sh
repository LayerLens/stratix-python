#!/bin/bash
set -euo pipefail

ROOT_DIR=$(git rev-parse --show-toplevel)

# Parse command line arguments
DRY_RUN=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--dry-run]"
      exit 1
      ;;
  esac
done

git fetch --tags --prune

REPO_URL="https://github.com/LayerLens/atlas-python"
TAG_PREFIX="sdk-v"
COMMIT=$(git rev-parse --short HEAD)
VERSION=$(bash "$ROOT_DIR/scripts/get_version.sh")
TAG="${TAG_PREFIX}${VERSION}"

if git rev-parse "$TAG" >/dev/null 2>&1; then
  echo "Error: Tag $TAG already exists"
  exit 1
fi

# Find the most recent version tag
LAST_RELEASE=$(git tag -l "${TAG_PREFIX}*" --sort=-v:refname | head -n 1)

echo "================================================"
echo "  Atlas Python SDK Release"
echo "================================================"
echo "version:      ${TAG}"
echo "commit:       ${COMMIT}"
echo "code:         ${REPO_URL}/commit/${COMMIT}"
echo "changeset:    ${REPO_URL}/compare/${LAST_RELEASE}...${COMMIT}"

if [ "$DRY_RUN" = true ]; then
  exit 0
fi

echo ""
echo ""
echo "Are you ready to release version ${VERSION}? Type 'YES' to continue:"
read -r CONFIRMATION

if [ "$CONFIRMATION" != "YES" ]; then
  echo "Release cancelled."
  exit 1
fi

# Create and push the tag
echo ""
echo "Creating and pushing tag ${TAG}"
echo ""

git tag "$TAG" "$COMMIT"
git push origin "$TAG"

echo ""
echo "Tag ${TAG} has been created and pushed to origin. Check GitHub Actions for build progress:"
echo "https://github.com/LayerLens/atlas-python/actions/workflows/publish-sdk.yaml"
echo ""