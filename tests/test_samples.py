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

# Directories containing library/support modules (not standalone samples).
# ``vendor`` holds instrumented vendor forks (genuine forks of an upstream
# framework's own example app, e.g. samples/vendor/langgraph/...): the graph
# module (``app.py``) has no ``main()`` / ``layerlens`` import — the runnable
# instrumentation lives in ``run_instrumented.py`` — so the standalone-sample
# contract does not apply. Parse/docstring/no-invalid-import checks still run.
_LIBRARY_DIRS = {"judges", "lib", "components", "hooks", "vendor"}

# Directories to skip entirely during sample discovery. The CopilotKit
# sample ships a Next.js app under ``app/frontend``; once a developer runs
# ``npm install`` there, dependencies like ``katex`` drop their own .py
# helper scripts into ``node_modules`` -- those are not LayerLens samples
# and must not be treated as such.
_SKIP_DIRS = {
    "node_modules",
    ".next",
    # ``samples/data/generators/`` holds the record-real-once fixture generators
    # (dev tooling that reuses the ``_generate_fixtures.py`` capture seam), not
    # customer samples — exclude it from sample discovery like the ``_``-prefixed
    # generator itself.
    "generators",
    "__pycache__",
    ".venv",
    "venv",
    "dist",
    "build",
    ".pytest_cache",
    "test-results",
    "playwright-report",
}


def _collect_samples():
    """Collect all sample .py files, excluding helpers and __init__."""
    samples = []
    for root, dirs, files in os.walk(SAMPLES_DIR):
        # Mutate ``dirs`` in place so ``os.walk`` does not descend into
        # build artefacts, virtualenvs, or vendored packages.
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
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


class TestFrameworkSampleConventions:
    """Runtime-correctness conventions for the adapter samples (LAY-3567 C2)."""

    @pytest.mark.parametrize("sample_path", STANDALONE_SAMPLES)
    def test_framework_adapter_constructors_pass_client(self, sample_path):
        """Framework adapters/handlers require a positional ``client``; a bare
        ``FooAdapter()`` in a sample is a guaranteed TypeError at runtime."""
        full_path = os.path.join(SAMPLES_DIR, sample_path)
        with open(full_path) as f:
            source = f.read()
        tree = ast.parse(source)

        adapter_names = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("layerlens.instrument.adapters.frameworks")
            ):
                adapter_names.update(alias.asname or alias.name for alias in node.names)

        violations = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in adapter_names
                and not node.args
                and not node.keywords
            ):
                violations.append(f"{node.func.id}() at line {node.lineno}")

        assert not violations, f"{sample_path} constructs a framework adapter without a client: {violations}"

    @pytest.mark.parametrize("sample_path", [s for s in SAMPLE_FILES if "autogen" in os.path.basename(s)])
    def test_autogen_samples_use_agentchat_api(self, sample_path):
        """The ``autogen`` extra installs autogen-agentchat (modules
        ``autogen_agentchat``/``autogen_core``/``autogen_ext``); there is no
        top-level ``autogen`` module, so the old pyautogen API cannot work."""
        full_path = os.path.join(SAMPLES_DIR, sample_path)
        with open(full_path) as f:
            source = f.read()
        tree = ast.parse(source)

        old_api_imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (
                node.module == "autogen" or (node.module or "").startswith("autogen.")
            ):
                old_api_imports.append(f"from {node.module} import ... at line {node.lineno}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "autogen" or alias.name.startswith("autogen."):
                        old_api_imports.append(f"import {alias.name} at line {node.lineno}")

        assert not old_api_imports, f"{sample_path} imports the old pyautogen API: {old_api_imports}"


_ADAPTER_IMPORT_PREFIX = "layerlens.instrument.adapters"


def _adapter_api_violations(source: str) -> list:
    """Return ``var.attr`` references in *source* that don't exist on the
    adapter class the variable was constructed from (LAY-3584 / T10).

    Resolves adapter types from ``from layerlens.instrument.adapters... import
    X`` + ``var = X(...)`` assignments, then checks every attribute access on
    those variables against the real class surface.
    """
    import importlib

    tree = ast.parse(source)

    imported = {}  # local name -> (module path, class name)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(_ADAPTER_IMPORT_PREFIX):
            for alias in node.names:
                imported[alias.asname or alias.name] = (node.module, alias.name)
    if not imported:
        return []

    var_types = {}  # variable name -> (module path, class name)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            fn = node.value.func
            if isinstance(fn, ast.Name) and fn.id in imported:
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        var_types[tgt.id] = imported[fn.id]
    if not var_types:
        return []

    cls_cache = {}
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in var_types:
            key = var_types[node.value.id]
            if key not in cls_cache:
                try:
                    cls_cache[key] = getattr(importlib.import_module(key[0]), key[1], None)
                except Exception:
                    # Adapter module not importable in this env — cannot check.
                    cls_cache[key] = None
            cls = cls_cache[key]
            if cls is not None and not hasattr(cls, node.attr):
                violations.append(
                    f"{node.value.id}.{node.attr} at line {node.lineno} — {key[1]} has no attribute {node.attr!r}"
                )
    return violations


class TestSampleAdapterAPIs:
    """Samples must only call APIs that exist on the adapter (LAY-3584 / T10).

    The N8 bug shipped two agentforce samples calling MVP-only methods that
    never existed in this SDK; conventions checks alone could not see that.
    """

    @pytest.mark.parametrize("sample_path", SAMPLE_FILES)
    def test_samples_reference_only_existing_adapter_apis(self, sample_path):
        full_path = os.path.join(SAMPLES_DIR, sample_path)
        with open(full_path) as f:
            source = f.read()
        violations = _adapter_api_violations(source)
        assert not violations, f"{sample_path} references nonexistent adapter APIs: {violations}"

    def test_guard_catches_synthetic_violation(self):
        source = (
            "from layerlens.instrument.adapters.frameworks.agentforce import AgentforceAdapter\n"
            "adapter = AgentforceAdapter(None)\n"
            "adapter.definitely_not_a_real_method()\n"
        )
        violations = _adapter_api_violations(source)
        assert violations and "definitely_not_a_real_method" in violations[0]

    def test_guard_accepts_real_apis(self):
        source = (
            "from layerlens.instrument.adapters.frameworks.agentforce import AgentforceAdapter\n"
            "adapter = AgentforceAdapter(None)\n"
            "adapter.connect({})\n"
            "sessions = adapter.import_sessions(start_date='2026-01-01')\n"
            "adapter.disconnect()\n"
        )
        assert _adapter_api_violations(source) == []


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

    def test_recorded_trace_path_resolves_under_data_traces(self):
        """recorded_trace_path points at a fixture under samples/data/traces/."""
        sys.path.insert(0, SAMPLES_DIR)
        try:
            from _helpers import recorded_trace_path
        finally:
            sys.path.pop(0)

        path = recorded_trace_path("industry", "financial_fraud.jsonl")
        assert os.path.isabs(path)
        norm = path.replace("\\", "/")
        assert norm.endswith("samples/data/traces/industry/financial_fraud.jsonl")

    def test_upload_recorded_trace_returns_trace_ids_in_order(self):
        """upload_recorded_trace reads every fixture line and uploads them as a
        single JSON array (so the backend creates one trace per record), then
        returns the created IDs in file order."""
        import json
        import tempfile

        sys.path.insert(0, SAMPLES_DIR)
        try:
            from _helpers import upload_recorded_trace
        finally:
            sys.path.pop(0)

        fd, fixture = tempfile.mkstemp(suffix=".jsonl")
        try:
            with os.fdopen(fd, "w") as f:
                for i in range(3):
                    f.write(json.dumps({"trace_id": f"t{i}", "events": []}) + "\n")

            mock_client = Mock()
            mock_response = Mock()
            mock_response.trace_ids = ["trc-1", "trc-2", "trc-3"]
            mock_client.traces.upload.return_value = mock_response

            result = upload_recorded_trace(mock_client, fixture)
        finally:
            os.unlink(fixture)

        assert result == ["trc-1", "trc-2", "trc-3"]
        mock_client.traces.upload.assert_called_once()
        # It uploads a temp JSON array (one trace per fixture line), not the
        # raw .jsonl, and cleans the temp file up.
        uploaded_path = mock_client.traces.upload.call_args[0][0]
        assert uploaded_path.endswith(".json")
        assert not os.path.exists(uploaded_path)

    def test_upload_recorded_trace_returns_empty_on_rejection(self):
        """A rejected upload (no trace_ids, no raise) yields an empty list, not a crash."""
        import json
        import tempfile

        sys.path.insert(0, SAMPLES_DIR)
        try:
            from _helpers import upload_recorded_trace
        finally:
            sys.path.pop(0)

        fd, fixture = tempfile.mkstemp(suffix=".jsonl")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(json.dumps({"trace_id": "t0", "events": []}) + "\n")

            mock_client = Mock()
            rejected = Mock()
            rejected.trace_ids = None
            mock_client.traces.upload.return_value = rejected

            assert upload_recorded_trace(mock_client, fixture) == []
        finally:
            os.unlink(fixture)

    def test_upload_recorded_trace_empty_fixture_returns_empty(self):
        """An empty fixture uploads nothing and returns an empty list."""
        import tempfile

        sys.path.insert(0, SAMPLES_DIR)
        try:
            from _helpers import upload_recorded_trace
        finally:
            sys.path.pop(0)

        fd, fixture = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            mock_client = Mock()
            assert upload_recorded_trace(mock_client, fixture) == []
            mock_client.traces.upload.assert_not_called()
        finally:
            os.unlink(fixture)

    def test_create_judge_namespace_disambiguates_name(self):
        """A namespace is appended to the judge name so identically-named judges
        across samples never collide."""
        sys.path.insert(0, SAMPLES_DIR)
        try:
            from _helpers import create_judge
        finally:
            sys.path.pop(0)

        mock_client = Mock()
        mock_client.judges.create.return_value = Mock(id="judge-1")

        create_judge(
            mock_client,
            name="Relevance Judge",
            evaluation_goal="Evaluate whether the response is relevant.",
            model_id="model-x",
            namespace="retail_recommender",
        )

        kwargs = mock_client.judges.create.call_args.kwargs
        assert kwargs["name"] == "Relevance Judge (retail_recommender)"
        # Without a namespace the name is untouched (backward compatible).
        create_judge(
            mock_client,
            name="Relevance Judge",
            evaluation_goal="Evaluate whether the response is relevant.",
            model_id="model-x",
        )
        assert mock_client.judges.create.call_args.kwargs["name"] == "Relevance Judge"
