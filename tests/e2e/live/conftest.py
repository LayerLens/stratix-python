"""Fixtures + gating for the live provider-adapter verification suite (L1 + L2).

This suite makes REAL provider SDK calls and uploads to a REAL (staging) LayerLens
backend. It is opt-in twice over:

1. Collection gate (``pytest_collection_modifyitems``): every test here is skipped
   unless ``LAYERLENS_LIVE=1`` or ``-m live`` is passed. CI runs bare ``pytest`` with
   no ``-m "not live"`` deselect, so this guarantees the live bodies never execute
   (and never spend money / hit a backend) in CI or a plain ``./scripts/test`` run.
2. Per-test credential skips: each test skips unless the staging LayerLens creds and
   that provider's key(s) are present.

See README.md (the runbook) for how to run and the L3 manual checklist.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import pytest

OPT_IN_ENV = "LAYERLENS_LIVE"


def _opted_in(config: pytest.Config) -> bool:
    if os.environ.get(OPT_IN_ENV) == "1":
        return True
    markexpr = getattr(config.option, "markexpr", "") or ""
    return "live" in markexpr and "not live" not in markexpr


def pytest_collection_modifyitems(config: pytest.Config, items: List[pytest.Item]) -> None:
    """Skip every item under this directory unless the suite is explicitly opted into."""
    if _opted_in(config):
        return
    here = os.path.dirname(__file__)
    skip = pytest.mark.skip(reason=f"live suite: set {OPT_IN_ENV}=1 or pass -m live to run")
    for item in items:
        if str(item.fspath).startswith(here):
            item.add_marker(skip)


def pytest_configure(config: pytest.Config) -> None:
    # Session-wide accumulator for the L3 report (one row per provider x variant).
    config._live_report_rows = []  # type: ignore[attr-defined]


@pytest.fixture
def stratix_live_client() -> Any:
    """A real Stratix client pointed at staging.

    Skips BEFORE constructing the client because ``Stratix.__init__`` makes a real
    org/project network call. Requires both the API key and an explicit staging base
    URL — we never run against the default (prod) base URL.
    """
    if not os.environ.get("LAYERLENS_STRATIX_API_KEY"):
        pytest.skip("LAYERLENS_STRATIX_API_KEY not set")
    if not os.environ.get("LAYERLENS_STRATIX_BASE_URL"):
        pytest.skip("LAYERLENS_STRATIX_BASE_URL not set (refusing to run against the default/prod base URL)")

    from layerlens import Stratix

    return Stratix()


@pytest.fixture
def record_result(request: pytest.FixtureRequest):
    """Return a callable that appends a harness result row to the session report."""

    def _record(row: Dict[str, Any]) -> None:
        request.config._live_report_rows.append(row)  # type: ignore[attr-defined]

    return _record


def pytest_terminal_summary(terminalreporter: Any, exitstatus: int, config: pytest.Config) -> None:
    rows = getattr(config, "_live_report_rows", [])
    if not rows:
        return
    from ._report import write_markdown_report, terminal_summary_lines

    path = write_markdown_report(rows)
    terminalreporter.section("LayerLens live adapter verification")
    for line in terminal_summary_lines(rows, path):
        terminalreporter.write_line(line)
