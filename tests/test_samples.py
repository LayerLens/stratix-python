"""Tests for SDK sample files.

Validates that all sample files are valid Python, structurally correct,
and follow conventions (main function, docstring, correct imports).
"""

import os
import ast
import sys
from unittest.mock import Mock

import pytest

SAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "samples")

# Directories containing library/support modules (not standalone samples)
_LIBRARY_DIRS = {"judges", "lib", "components", "hooks"}


def _collect_samples():
    """Collect all sample .py files, excluding helpers and __init__."""
    samples = []
    for root, dirs, files in os.walk(SAMPLES_DIR):
        for f in files:
            if f.endswith(".py") and not f.startswith("_"):
                rel = os.path.relpath(os.path.join(root, f), SAMPLES_DIR)
                samples.append(rel)
    return sorted(samples)


def _is_library_module(sample_path: str) -> bool:
    """Check if a sample path is a library/support module (not a standalone sample)."""
    parts = sample_path.replace("\\", "/").split("/")
    return any(part in _LIBRARY_DIRS for part in parts)


SAMPLE_FILES = _collect_samples()
STANDALONE_SAMPLES = [s for s in SAMPLE_FILES if not _is_library_module(s)]
LIBRARY_MODULES = [s for s in SAMPLE_FILES if _is_library_module(s)]


class TestSampleStructure:
    """Validate structure and conventions for every SDK sample."""

    @pytest.mark.parametrize("sample_path", SAMPLE_FILES)
    def test_sample_parses(self, sample_path):
        """Each sample should be valid Python."""
        full_path = os.path.join(SAMPLES_DIR, sample_path)
        with open(full_path) as f:
            source = f.read()
        tree = ast.parse(source, filename=sample_path)
        assert tree is not None

    @pytest.mark.parametrize("sample_path", STANDALONE_SAMPLES)
    def test_sample_has_main(self, sample_path):
        """Each standalone sample should define a main() function."""
        full_path = os.path.join(SAMPLES_DIR, sample_path)
        with open(full_path) as f:
            source = f.read()
        tree = ast.parse(source)
        func_names = [node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        assert "main" in func_names, f"{sample_path} should define a main() function"

    @pytest.mark.parametrize("sample_path", SAMPLE_FILES)
    def test_no_invalid_imports(self, sample_path):
        """No sample should import from non-existent SDK modules."""
        invalid_modules = [
            "layerlens.adapters",
            "layerlens.trace",
            "layerlens.judges",
            "layerlens.memory",
            "layerlens.otel",
            "stratix.sdk.python",
        ]
        full_path = os.path.join(SAMPLES_DIR, sample_path)
        with open(full_path) as f:
            source = f.read()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for invalid in invalid_modules:
                    assert not node.module.startswith(invalid), (
                        f"{sample_path} imports from {node.module} which doesn't exist in the SDK"
                    )

    @pytest.mark.parametrize("sample_path", STANDALONE_SAMPLES)
    def test_imports_layerlens(self, sample_path):
        """Each standalone sample should import from layerlens (directly or via _runner)."""
        full_path = os.path.join(SAMPLES_DIR, sample_path)
        with open(full_path) as f:
            source = f.read()
        tree = ast.parse(source)
        has_layerlens = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if "layerlens" in node.module:
                    has_layerlens = True
                # openclaw demos import layerlens transitively via _runner
                if node.module == "_runner" or node.module.endswith("._runner"):
                    has_layerlens = True
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if "layerlens" in alias.name:
                        has_layerlens = True
        assert has_layerlens, f"{sample_path} should import from layerlens"

    @pytest.mark.parametrize("sample_path", SAMPLE_FILES)
    def test_has_docstring(self, sample_path):
        """Each sample should have a module-level docstring."""
        full_path = os.path.join(SAMPLES_DIR, sample_path)
        with open(full_path) as f:
            source = f.read()
        tree = ast.parse(source)
        docstring = ast.get_docstring(tree)
        assert docstring, f"{sample_path} should have a module docstring"


class TestHelpers:
    """Tests for the shared _helpers module."""

    def test_upload_trace_dict(self):
        """Test the shared upload_trace_dict helper."""
        sys.path.insert(0, SAMPLES_DIR)
        try:
            from _helpers import upload_trace_dict
        finally:
            sys.path.pop(0)

        mock_client = Mock()
        mock_response = Mock()
        mock_response.trace_ids = ["trace-abc"]
        mock_client.traces.upload.return_value = mock_response

        result = upload_trace_dict(
            mock_client,
            input_text="test input",
            output_text="test output",
            metadata={"key": "value"},
        )

        assert result == mock_response
        mock_client.traces.upload.assert_called_once()
        call_args = mock_client.traces.upload.call_args
        uploaded_path = call_args[0][0]
        assert not os.path.exists(uploaded_path), "Temp file should be cleaned up"

    def test_upload_trace_dict_without_metadata(self):
        """Test upload_trace_dict without optional metadata."""
        sys.path.insert(0, SAMPLES_DIR)
        try:
            from _helpers import upload_trace_dict
        finally:
            sys.path.pop(0)

        mock_client = Mock()
        mock_response = Mock()
        mock_response.trace_ids = ["trace-def"]
        mock_client.traces.upload.return_value = mock_response

        result = upload_trace_dict(
            mock_client,
            input_text="hello",
            output_text="world",
        )

        assert result == mock_response
        mock_client.traces.upload.assert_called_once()
