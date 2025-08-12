install-build-deps:
	# Install build dependencies using rye
	rye sync --all-features

build: install-build-deps clean _template-version
	python -m build
	# Restore the original version file after the build
	git checkout src/_version.py

test-wheel: build
	# Test the built wheel
	rye run pytest --wheel

clean:
	rm -rf build dist

_publish:
	./scripts/publish.sh

_template-version:
	@bash scripts/template-version.sh

_check-git-clean:
	@if [ -n "$$(git status --porcelain)" ]; then \
		echo "Error: Git working directory is not clean. Won't run publish."; \
		exit 1; \
	fi

_verify-build-publish: _check-git-clean build test-wheel _publish

publish-to-testpypi: export PYPI_REPO := testpypi
publish-to-testpypi: _verify-build-publish

publish-to-pypi: export PYPI_REPO := pypi
publish-to-pypi: _verify-build-publish

push-release-tag:
	@bash scripts/push-release-tag.sh $(DRY_RUN)

help:
	@echo "Available targets:"
	@echo "  build               - Build Python package"
	@echo "  clean               - Remove build artifacts"
	@echo "  help                - Show this help message"
	@echo "  install-build-deps  - Install build dependencies for CI"
	@echo "  test-wheel          - Run tests against built wheel"
	@echo "  publish-to-pypi     - Publish to PyPI"
	@echo "  publish-to-testpypi - Publish to TestPyPI"
	@echo "  push-release-tag    - Create and push a release tag"