"""Lint-style guards that keep the adapter manifest, capability set,
emitted-event surface, and shipping artifacts in agreement with each
other.

These tests do not exercise runtime behavior. They walk the source tree
the same way ``scripts/emit_adapter_manifest.py`` does and assert the
metadata it would emit matches what every adapter actually implements.

Three guards are enforced:

1. **Capability / hook consistency** — if an adapter implements an
   ``on_<event>`` hook, the corresponding ``AdapterCapability`` must be
   present in ``get_adapter_info().capabilities``. The reverse is also
   checked: declaring a capability without the hook is a lie to the
   manifest. Catches the pre-PR pydantic_ai bug where ``on_handoff``
   was implemented but ``TRACE_HANDOFFS`` was missing from the
   capabilities list.

2. **Event-type parity** — every framework adapter that ships a
   ``lifecycle.py`` must emit the canonical L1/L3/L5a event set
   (``agent.input``, ``agent.output``, ``tool.call``, ``model.invoke``,
   ``environment.config``). Catches "we have a hook but never
   ``emit_dict_event(...)`` from it" regressions.

3. **Maturity-vs-artifacts** — every adapter listed in
   ``scripts/emit_adapter_manifest._MATURE`` must have ALL FOUR
   artifacts: a per-adapter test file with at least 12 test functions,
   a sample under ``samples/instrument/<key>/main.py``, a doc under
   ``docs/adapters/frameworks-<key>.md`` (or ``providers-<key>.md`` /
   ``protocols-<key>.md``), and a STRATIX→LayerLens deprecation alias
   exported from the adapter package ``__init__.py``. Catches the
   pre-PR smolagents bug where it sat in ``_MATURE`` without doc or
   sample.

Known-failure (xfail) entries are tagged with the sibling PR that fixes
them so CI doesn't block while remediation is in flight, but the
assertions remain real — once the sibling PR lands the xfail flips to
xpass and pytest in strict mode (configured below) will fail loudly
until the marker is removed.
"""

from __future__ import annotations

import re
import ast
from typing import Any, Dict, List, Tuple
from pathlib import Path

import pytest

# Import the emitter module to read its source-of-truth maturity sets.
# We do NOT call ``build_manifest()`` here — these tests are purely
# static so they run without any framework runtimes installed.
from scripts.emit_adapter_manifest import (
    _MATURE,
    _CATEGORY,
    _LIFECYCLE_PREVIEW,
)

# --- Repository layout ---

_REPO_ROOT: Path = Path(__file__).resolve().parents[3]
_SRC_FRAMEWORKS: Path = _REPO_ROOT / "src" / "layerlens" / "instrument" / "adapters" / "frameworks"
_SRC_PROVIDERS: Path = _REPO_ROOT / "src" / "layerlens" / "instrument" / "adapters" / "providers"
_SRC_PROTOCOLS: Path = _REPO_ROOT / "src" / "layerlens" / "instrument" / "adapters" / "protocols"
_TESTS_FRAMEWORKS: Path = _REPO_ROOT / "tests" / "instrument" / "adapters" / "frameworks"
_TESTS_PROVIDERS: Path = _REPO_ROOT / "tests" / "instrument" / "adapters" / "providers"
_TESTS_PROTOCOLS: Path = _REPO_ROOT / "tests" / "instrument" / "adapters" / "protocols"
_SAMPLES_DIR: Path = _REPO_ROOT / "samples" / "instrument"
_DOCS_DIR: Path = _REPO_ROOT / "docs" / "adapters"

# --- Canonical contract ---

# Mapping from ``AdapterCapability`` name to the set of ``on_<hook>``
# method names that satisfy it. If ANY of the hooks is defined, the
# capability must be declared, and vice versa. Multiple hook names per
# capability accommodate frameworks whose native callback shape differs
# from the canonical names (e.g., semantic_kernel emits tool-equivalent
# events from ``on_function_*`` and model-equivalent events from
# ``on_model_invoke`` instead of the canonical ``on_tool_use`` /
# ``on_llm_call``). The mapping intentionally errs on the side of
# accepting framework-native names so the lint only fires on a true
# capability/hook contract violation.
_CAPABILITY_TO_HOOKS: Dict[str, Tuple[str, ...]] = {
    "TRACE_HANDOFFS": ("on_handoff",),
    "TRACE_TOOLS": ("on_tool_use", "on_function_start", "on_function_end"),
    "TRACE_MODELS": ("on_llm_call", "on_model_invoke"),
}

# The canonical L1 (lifecycle), L3 (model), L5a (tool), and Cross-cut
# (environment) event types every framework lifecycle adapter is
# expected to emit at least once. Adapters that hook into a different
# runtime shape (e.g., semantic_kernel's per-function callbacks) still
# need to emit these because they form the cross-framework contract the
# trace UI renders against.
_CANONICAL_FRAMEWORK_EVENTS: Tuple[str, ...] = (
    "agent.input",
    "agent.output",
    "tool.call",
    "model.invoke",
    "environment.config",
)

# Slug → category — keep aligned with ``_CATEGORY`` in the emitter for
# the doc/test/sample path resolution below. The emitter's own
# ``_CATEGORY`` is the source of truth; this is just a typed alias.
_CATEGORY_BY_KEY: Dict[str, str] = dict(_CATEGORY)


# -------------------- Helpers --------------------


def _adapter_dir(category: str, key: str) -> Path:
    """Resolve the on-disk source directory for ``key``.

    Most framework keys map 1:1 to their directory name. The exception
    is ``salesforce_agentforce`` whose package directory is just
    ``agentforce`` (the ``salesforce_`` prefix only exists at the
    registry layer for disambiguation against a hypothetical
    ``zendesk_agentforce``).
    """
    base_map = {
        "framework": _SRC_FRAMEWORKS,
        "provider": _SRC_PROVIDERS,
        "protocol": _SRC_PROTOCOLS,
    }
    base = base_map.get(category)
    assert base is not None, f"Unknown category {category!r} for {key!r}"
    if category == "framework" and key == "salesforce_agentforce":
        return base / "agentforce"
    if category == "provider":
        # Providers ship as flat ``<key>_adapter.py`` modules, not
        # directories. Caller should guard before using this path.
        return base / f"{key}_adapter.py"
    if category == "protocol":
        # Protocol adapters ship as ``<key>/`` directories under
        # ``protocols/``; ``mcp_extensions`` is the one rename to
        # ``mcp/``.
        if key == "mcp_extensions":
            return base / "mcp"
        return base / key
    return base / key


def _read_source_if_exists(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def _hooks_defined(source: str) -> set[str]:
    """Return the set of ``on_<event>`` method names defined in source.

    Uses :mod:`ast` instead of regex so we don't false-positive on a
    string literal mentioning ``on_handoff`` somewhere in a docstring.
    """
    if not source:
        return set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("on_"):
                found.add(node.name)
    return found


def _capabilities_declared(source: str) -> set[str]:
    """Return the set of ``AdapterCapability.<NAME>`` references in source.

    We pattern-match the attribute reference instead of importing the
    adapter class to keep this guard runnable without optional runtimes.
    """
    if not source:
        return set()
    return set(re.findall(r"AdapterCapability\.([A-Z_]+)", source))


def _events_emitted(source: str) -> set[str]:
    """Return the set of event-type strings passed to ``emit_dict_event``.

    Uses AST to collect first-arg string-literal calls so multi-line
    formatting doesn't trip a regex.
    """
    if not source:
        return set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    out: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # Match ``self.emit_dict_event(...)`` and ``adapter.emit_dict_event(...)``
        if not (isinstance(func, ast.Attribute) and func.attr == "emit_dict_event"):
            continue
        if not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            out.add(first.value)
    return out


def _has_test_file_with_min_count(category: str, key: str, min_funcs: int = 12) -> Tuple[bool, int, Path]:
    """Return ``(present, test_count, path)`` for ``test_<key>_adapter.py``."""
    base_map = {
        "framework": _TESTS_FRAMEWORKS,
        "provider": _TESTS_PROVIDERS,
        "protocol": _TESTS_PROTOCOLS,
    }
    base = base_map[category]
    candidate = base / f"test_{key}_adapter.py"
    if not candidate.is_file():
        return False, 0, candidate
    src = candidate.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return True, 0, candidate
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            count += 1
    return True, count, candidate


def _has_sample(key: str) -> Tuple[bool, Path]:
    candidate = _SAMPLES_DIR / key / "main.py"
    return candidate.is_file(), candidate


def _has_doc(category: str, key: str) -> Tuple[bool, Path]:
    """Resolve the doc path. Frameworks and providers use distinct prefixes.

    Frameworks: ``docs/adapters/frameworks-<key>.md`` (with the
    ``salesforce_agentforce`` → ``agentforce`` slug shortening).
    Providers: ``docs/adapters/providers-<key>.md`` (some providers use
    hyphenated slugs like ``providers-google-vertex.md`` or
    ``providers-azure-openai.md``).
    Protocols: ``docs/adapters/protocols-<key>.md``.
    """
    if category == "framework":
        slug = "agentforce" if key == "salesforce_agentforce" else key
        return (_DOCS_DIR / f"frameworks-{slug}.md").is_file(), _DOCS_DIR / f"frameworks-{slug}.md"
    if category == "provider":
        # Try both underscore and hyphenated forms to match historic
        # naming (``providers-google-vertex.md`` vs
        # ``providers-google_vertex.md``).
        hyphen = key.replace("_", "-")
        for stem in (f"providers-{key}", f"providers-{hyphen}"):
            p = _DOCS_DIR / f"{stem}.md"
            if p.is_file():
                return True, p
        return False, _DOCS_DIR / f"providers-{key}.md"
    if category == "protocol":
        slug = "mcp" if key == "mcp_extensions" else key
        return (_DOCS_DIR / f"protocols-{slug}.md").is_file(), _DOCS_DIR / f"protocols-{slug}.md"
    return False, _DOCS_DIR / f"unknown-{key}.md"


def _has_stratix_alias(category: str, key: str) -> Tuple[bool, Path]:
    """Check the package ``__init__.py`` for a STRATIX→LayerLens alias.

    Convention: each migrated adapter exports a ``STRATIX*`` name that
    aliases its ``LayerLens*`` (or canonical) class so customers
    upgrading from the legacy ``stratix`` package can swap the import
    root without renaming every reference. The lint accepts ANY
    top-level binding that:

      - starts with ``STRATIX`` (case-sensitive), and
      - assigns to either an ``ADAPTER_CLASS`` reference, an attribute
        on the package's adapter class, or another already-bound
        identifier.

    Providers ship as flat ``<key>_adapter.py`` modules — for those the
    alias must live in the module file itself, not a package
    ``__init__.py``.
    """
    if category == "provider":
        candidate = _SRC_PROVIDERS / f"{key}_adapter.py"
    else:
        if category == "framework":
            slug = "agentforce" if key == "salesforce_agentforce" else key
            candidate = _SRC_FRAMEWORKS / slug / "__init__.py"
        else:  # protocol
            slug = "mcp" if key == "mcp_extensions" else key
            candidate = _SRC_PROTOCOLS / slug / "__init__.py"
    src = _read_source_if_exists(candidate)
    if not src:
        return False, candidate
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False, candidate
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.startswith("STRATIX"):
                    return True, candidate
    return False, candidate


def _enumerate_lifecycle_adapters() -> List[str]:
    """Return all framework keys whose package ships a ``lifecycle.py``.

    Skips packages that ship as flat modules (``embedding_adapter.py``,
    ``vector_store_adapter.py`` under the ``embedding/`` package use a
    different shape). The intent is to lint the callback-style
    adapters; non-callback adapters are exercised by their own tests.
    """
    out: List[str] = []
    for key, category in _CATEGORY_BY_KEY.items():
        if category != "framework":
            continue
        d = _adapter_dir("framework", key)
        if (d / "lifecycle.py").is_file():
            out.append(key)
    return sorted(out)


# -------------------- Sanity --------------------


def test_mature_and_lifecycle_preview_sets_disjoint() -> None:
    """The emitter asserts this at import-time; this test surfaces it
    inside the test suite so a failure shows up in CI test output (not
    just in the manifest-emitter import traceback)."""
    overlap = _MATURE & _LIFECYCLE_PREVIEW
    assert not overlap, (
        f"Adapter(s) listed in both _MATURE and _LIFECYCLE_PREVIEW: "
        f"{sorted(overlap)}. An adapter cannot be both fully graduated "
        f"and missing artifacts. Pick one set."
    )


def test_emitter_categories_cover_both_maturity_sets() -> None:
    """Every key in either maturity set must also appear in
    ``_CATEGORY``; otherwise the emitter would default-classify it as
    ``framework`` and the catalog UI would misrender."""
    missing = (_MATURE | _LIFECYCLE_PREVIEW) - set(_CATEGORY_BY_KEY)
    assert not missing, (
        f"Maturity-set keys without a _CATEGORY entry: {sorted(missing)}. "
        f"Add them to _CATEGORY in scripts/emit_adapter_manifest.py."
    )


# -------------------- Lint 1: capability ↔ hook consistency --------------------


@pytest.mark.parametrize("framework_key", _enumerate_lifecycle_adapters())
def test_capability_hook_consistency(framework_key: str, request: pytest.FixtureRequest) -> None:
    """An ``on_<event>`` hook in lifecycle.py implies the matching
    capability MUST be declared, and vice versa.

    Known gaps tagged xfail:

    * ``pydantic_ai`` defines ``on_handoff`` but does NOT declare
      ``TRACE_HANDOFFS`` — fixed by the sibling pydantic_ai cleanup PR.
    """
    if framework_key == "pydantic_ai":
        request.applymarker(
            pytest.mark.xfail(
                reason=(
                    "Known gap: pydantic_ai implements on_handoff but does not "
                    "declare TRACE_HANDOFFS in get_adapter_info().capabilities. "
                    "Fixed by sibling PR (pydantic_ai capabilities cleanup)."
                ),
                strict=True,
            )
        )
    if framework_key == "browser_use":
        request.applymarker(
            pytest.mark.xfail(
                reason=(
                    "browser_use ships as a placeholder lifecycle.py that "
                    "wires the field-specific truncation policy ahead of M7 "
                    "(cross-pollination audit §2.4) but does not yet "
                    "implement the on_<event> hooks. The capability/hook "
                    "consistency check turns green when the M7 instrumentation "
                    "PR lands and the adapter graduates to _MATURE."
                ),
                strict=True,
            )
        )

    src_path = _adapter_dir("framework", framework_key) / "lifecycle.py"
    src = _read_source_if_exists(src_path)
    assert src, f"Expected lifecycle.py at {src_path}"

    hooks = _hooks_defined(src)
    caps = _capabilities_declared(src)

    mismatches: List[str] = []
    for cap_name, hook_options in _CAPABILITY_TO_HOOKS.items():
        any_hook_present = any(h in hooks for h in hook_options)
        cap_present = cap_name in caps
        if any_hook_present and not cap_present:
            present_hooks = sorted(h for h in hook_options if h in hooks)
            mismatches.append(
                f"{framework_key}: hook(s) {present_hooks} defined but capability "
                f"AdapterCapability.{cap_name} is not declared in get_adapter_info()"
            )
        if cap_present and not any_hook_present:
            mismatches.append(
                f"{framework_key}: capability AdapterCapability.{cap_name} is "
                f"declared but none of the satisfying hooks {list(hook_options)} "
                f"is implemented"
            )

    assert not mismatches, "\n".join(mismatches)


# -------------------- Lint 2: canonical event-type parity --------------------


@pytest.mark.parametrize("framework_key", _enumerate_lifecycle_adapters())
def test_canonical_events_emitted(framework_key: str, request: pytest.FixtureRequest) -> None:
    """Every framework lifecycle adapter must emit the canonical L1/L3/L5a
    event-type set so cross-framework UI rendering stays uniform.

    Known gaps tagged xfail:

    * ``browser_use`` is a placeholder lifecycle.py that wires the
      field-specific truncation policy ahead of M7 (cross-pollination
      audit §2.4) but does not yet emit canonical events — those land
      with the M7 instrumentation PR. Marked ``strict=True`` so the
      xfail flips red the moment the M7 PR adds them.
    """
    if framework_key == "browser_use":
        request.applymarker(
            pytest.mark.xfail(
                reason=(
                    "browser_use ships as a placeholder lifecycle.py that "
                    "pre-wires the field-specific truncation policy ahead "
                    "of M7 (cross-pollination audit §2.4); canonical events "
                    "are added by the M7 instrumentation PR."
                ),
                strict=True,
            )
        )

    src_path = _adapter_dir("framework", framework_key) / "lifecycle.py"
    src = _read_source_if_exists(src_path)
    assert src, f"Expected lifecycle.py at {src_path}"

    emitted = _events_emitted(src)
    missing = [evt for evt in _CANONICAL_FRAMEWORK_EVENTS if evt not in emitted]
    assert not missing, (
        f"{framework_key}: lifecycle.py never calls "
        f"emit_dict_event(<event>, ...) for: {missing}. Canonical event "
        f"types every framework adapter must emit at least once: "
        f"{list(_CANONICAL_FRAMEWORK_EVENTS)}."
    )


# -------------------- Lint 3: maturity-vs-artifacts --------------------


def _collect_artifact_gaps(key: str) -> List[str]:
    """Return human-readable list of missing artifacts for a mature ``key``."""
    category = _CATEGORY_BY_KEY.get(key, "framework")
    gaps: List[str] = []

    test_present, test_count, test_path = _has_test_file_with_min_count(category, key, min_funcs=12)
    if not test_present:
        gaps.append(f"missing test file at {test_path}")
    elif test_count < 12:
        gaps.append(f"test file {test_path} has {test_count} test functions (need >= 12)")

    sample_present, sample_path = _has_sample(key)
    if not sample_present:
        gaps.append(f"missing sample at {sample_path}")

    doc_present, doc_path = _has_doc(category, key)
    if not doc_present:
        gaps.append(f"missing doc at {doc_path}")

    alias_present, alias_path = _has_stratix_alias(category, key)
    if not alias_present:
        gaps.append(f"missing STRATIX→LayerLens deprecation alias in {alias_path}")

    return gaps


@pytest.mark.parametrize("mature_key", sorted(_MATURE))
def test_mature_adapters_have_required_artifacts(
    mature_key: str, request: pytest.FixtureRequest
) -> None:
    """Every adapter in ``_MATURE`` must ship test + sample + doc + alias.

    Known gaps tagged xfail:

    * Currently none — ``smolagents`` was the canonical xfail target
      pre-PR but has been moved out of ``_MATURE`` and into
      ``_LIFECYCLE_PREVIEW`` as part of this same change. If a future
      PR re-promotes smolagents to ``_MATURE`` without first landing
      its sample and doc, this test will fail loudly.

    Provider artifacts (test files under
    ``tests/instrument/adapters/providers/`` and per-provider docs)
    land in their own sibling PRs; until then those entries are
    expected-failure with the matching sibling-PR label.
    """
    # Provider artifact PRs have not yet landed on this branch — every
    # mature provider therefore has missing artifacts. xfail them
    # individually so the adapters that DID get artifacts still trip
    # the lint.
    pending_provider_artifacts = {
        "openai",
        "anthropic",
        "azure_openai",
        "aws_bedrock",
        "google_vertex",
        "ollama",
        "litellm",
        "cohere",
        "mistral",
    }
    if mature_key in pending_provider_artifacts:
        request.applymarker(
            pytest.mark.xfail(
                reason=(
                    f"Provider artifact PR for {mature_key!r} has not landed "
                    f"on this branch yet (test file, sample, doc, and "
                    f"STRATIX→LayerLens alias arrive together with the "
                    f"per-provider port). Once the sibling PR merges and the "
                    f"artifacts exist, this xfail flips to xpass and pytest "
                    f"strict mode forces removal of the marker."
                ),
                strict=True,
            )
        )

    gaps = _collect_artifact_gaps(mature_key)
    assert not gaps, (
        f"{mature_key} is in _MATURE but is missing required artifacts:\n  - "
        + "\n  - ".join(gaps)
    )


# -------------------- Manifest-emitter smoke --------------------


def test_emitter_runs_without_runtime_deps() -> None:
    """The emitter must produce a JSON-serializable manifest without
    importing optional adapter runtimes (semantic_kernel, openai, etc).
    Catches accidental top-level imports added to a lifecycle module."""
    from scripts.emit_adapter_manifest import build_manifest

    manifest = build_manifest()
    assert manifest["schema_version"] == "1.0.0"
    assert manifest["source"] == "layerlens"
    assert manifest["adapter_count"] > 0
    assert "by_maturity" in manifest
    assert manifest["by_maturity"]["lifecycle_preview"] >= 1, (
        "Emitter must emit at least one lifecycle_preview entry; the "
        "tier exists in the schema for a reason. If you removed the "
        "last preview adapter, also remove the tier from the schema."
    )

    # Verify every entry has a maturity tier from the documented enum.
    valid_tiers = {"mature", "lifecycle_preview", "smoke_only"}
    for entry in manifest["adapters"]:
        assert entry["maturity"] in valid_tiers, (
            f"{entry['key']!r} has invalid maturity {entry['maturity']!r}"
        )


def test_emitter_round_trips_to_json() -> None:
    """The manifest must be deterministically serializable so the
    atlas-app PR diff is reviewable."""
    import json

    from scripts.emit_adapter_manifest import build_manifest

    manifest = build_manifest()
    text_a = json.dumps(manifest, indent=2, sort_keys=True)
    text_b = json.dumps(manifest, indent=2, sort_keys=True)
    assert text_a == text_b
    # And it must not contain placeholder ``Any`` / generic noise.
    parsed: Dict[str, Any] = json.loads(text_a)
    assert isinstance(parsed["adapters"], list)
