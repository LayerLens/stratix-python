"""Adapter coverage-manifest gate (LAY-3574 / T0).

Enforces ``tests/adapter_manifest.toml``:

* every adapter the code registers must be declared in the manifest (new
  adapters cannot merge without declaring their tier matrix);
* the manifest must not declare adapters the code doesn't have (typo guard);
* every non-pending tier must point at an existing test file that defines
  tests (and contains the declared pattern, when one is given);
* pending tiers must carry a reason naming the PR/ticket that delivers them;
* ``matrix_row`` references must exist in ``tests/matrix/frameworks.toml``;
* ``live_id`` references must exist in the live registries.

Skip-at-runtime enforcement is split by environment: base CI runs this
structural gate plus ``tests/test_skip_hygiene.py`` (no fictional skip
targets); the per-framework matrix (``tests/matrix/run_matrix.py``) fails any
row whose modules skip while the framework IS installed.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict

import pytest

try:
    import tomllib  # py >= 3.11
except ImportError:  # py3.9/3.10 — tomli ships with pytest's dependencies
    import tomli as tomllib  # type: ignore[no-redef]

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TESTS_DIR)
MANIFEST_PATH = os.path.join(TESTS_DIR, "adapter_manifest.toml")
MATRIX_SPEC_PATH = os.path.join(TESTS_DIR, "matrix", "frameworks.toml")

_TIER_KEYS = (
    "unit",
    "streaming",
    "async",
    "redaction",
    "concurrency",
    "disconnect_restore",
    "doubles",
)

#: Adapters that exist in code but are wired explicitly (not in auto()).
_EXTRA_CODE_ADAPTERS = {"agentforce", "langfuse", "embedding", "vector_store"}
_PROTOCOL_ADAPTERS = {"mcp", "a2a", "agui", "ap2", "ucp", "a2ui"}


def _load_manifest() -> Dict[str, Any]:
    with open(MANIFEST_PATH, "rb") as f:
        return tomllib.load(f)["adapters"]


def _code_adapters() -> set:
    from layerlens.instrument.adapters._registry import (
        _PROVIDER_PACKAGES,
        _FRAMEWORK_ADAPTERS,
    )

    return set(_FRAMEWORK_ADAPTERS) | set(_PROVIDER_PACKAGES) | _EXTRA_CODE_ADAPTERS | _PROTOCOL_ADAPTERS


MANIFEST = _load_manifest()


class TestManifestCoverage:
    def test_every_code_adapter_is_declared(self) -> None:
        missing = sorted(_code_adapters() - set(MANIFEST))
        assert not missing, (
            f"adapters registered in code but missing from tests/adapter_manifest.toml: {missing}. "
            "New adapters must declare their test-tier matrix."
        )

    def test_no_unknown_manifest_entries(self) -> None:
        unknown = sorted(set(MANIFEST) - _code_adapters())
        assert not unknown, f"manifest declares adapters the code doesn't register (typo?): {unknown}"

    def test_every_adapter_declares_a_unit_tier(self) -> None:
        missing = sorted(name for name, entry in MANIFEST.items() if "unit" not in entry)
        assert not missing, f"adapters without a declared unit tier: {missing}"


def _tier_params():
    params = []
    for name, entry in sorted(MANIFEST.items()):
        for tier in _TIER_KEYS:
            if tier in entry:
                params.append(pytest.param(name, tier, entry[tier], id=f"{name}-{tier}"))
    return params


class TestTierDeclarations:
    @pytest.mark.parametrize("adapter,tier,decl", _tier_params())
    def test_tier_resolves(self, adapter: str, tier: str, decl: Dict[str, Any]) -> None:
        assert isinstance(decl, dict) and decl.get("path"), f"{adapter}.{tier} must declare a path"
        pending = decl.get("pending")
        if pending is not None:
            assert isinstance(pending, str) and pending.strip(), f"{adapter}.{tier} is pending but names no PR/ticket"
            return  # delivered by an open branch; enforced once it merges

        path = os.path.join(REPO_ROOT, decl["path"])
        assert os.path.isfile(path), f"{adapter}.{tier} points at a missing file: {decl['path']}"
        with open(path) as f:
            source = f.read()
        assert re.search(r"^\s*(async )?def test_", source, re.M), f"{adapter}.{tier}: {decl['path']} defines no tests"
        pattern = decl.get("pattern")
        if pattern:
            assert pattern in source, f"{adapter}.{tier}: pattern {pattern!r} not found in {decl['path']}"


class TestCrossReferences:
    def test_matrix_rows_exist(self) -> None:
        with open(MATRIX_SPEC_PATH, "rb") as f:
            rows = set(tomllib.load(f)["frameworks"])
        bad = sorted(
            (name, entry["matrix_row"])
            for name, entry in MANIFEST.items()
            if "matrix_row" in entry and entry["matrix_row"] not in rows
        )
        assert not bad, f"manifest matrix_row values missing from tests/matrix/frameworks.toml: {bad}"

    def test_live_ids_exist(self) -> None:
        try:
            from tests.e2e.live._registry import PROVIDERS as live_providers
            from tests.e2e.live._protocol_registry import PROTOCOLS as live_protocols
            from tests.e2e.live._framework_registry import FRAMEWORKS as live_frameworks
        except Exception as exc:  # live suite imports must never break the gate
            pytest.skip(f"live registries not importable here: {exc}")

        live_ids = {c.id for c in live_providers} | {c.id for c in live_frameworks} | {c.id for c in live_protocols}
        bad = sorted(
            (name, entry["live_id"])
            for name, entry in MANIFEST.items()
            if "live_id" in entry and entry["live_id"] not in live_ids
        )
        assert not bad, f"manifest live_id values missing from the live registries: {bad}"


class TestPendingReport:
    def test_report_pending_tiers(self) -> None:
        """Never fails — prints the outstanding pending tiers so reviewers see
        what enforcement is still waiting on which PR."""
        pending = [
            f"{name}.{tier} -> {entry[tier]['pending']}"
            for name, entry in sorted(MANIFEST.items())
            for tier in _TIER_KEYS
            if tier in entry and isinstance(entry[tier], dict) and entry[tier].get("pending")
        ]
        for line in pending:
            print(f"PENDING TIER: {line}")
        assert True
