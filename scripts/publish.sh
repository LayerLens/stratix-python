#!/bin/bash
# Publish the package to PyPI or TestPyPI depending
# on the PYPI_REPO (pypi | testpypi) environment variable

if [ -z "$PYPI_REPO" ]; then
    echo "Error: PYPI_REPO environment variable must be set"
    exit 1
fi

if [ "$PYPI_REPO" != "pypi" ] && [ "$PYPI_REPO" != "testpypi" ]; then
    echo "Error: PYPI_REPO must be either 'pypi' or 'testpypi'"
    exit 1
fi

VERSION=$(bash scripts/get_version.sh)

if [ -z "$VERSION" ]; then
    echo "Error: Could not determine version"
    exit 1
fi

echo "Publishing version $VERSION to $PYPI_REPO"

twine upload --repository "$PYPI_REPO" dist/*"$VERSION"*