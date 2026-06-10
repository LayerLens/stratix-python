"""Live L1 + L2 + linkage verification, one test per (framework, variant).

Opt-in: see ``conftest.py`` (``LAYERLENS_LIVE=1`` or ``-m live``). Each test
additionally skips unless the framework's package(s) are installed and its
credentials are present. Platform-side linkage is asserted only when
``LAYERLENS_LIVE_INTEGRATION_ID`` is set (see ``_linkage``), keeping the suite
green on a machine without a registered audit integration.

    LAYERLENS_LIVE=1 \\
    LAYERLENS_STRATIX_BASE_URL=http://localhost:8080/api/v1 LAYERLENS_STRATIX_API_KEY=... \\
    LAYERLENS_LIVE_INTEGRATION_ID=<id> OPENAI_API_KEY=... \\
    ./scripts/test tests/e2e/live -k langchain
"""

from __future__ import annotations

import pytest

from ._framework_harness import run_framework_case, run_self_flushing_case
from ._framework_registry import FRAMEWORKS, missing_credentials

_VARIANTS = ("default", "redaction")
_AMBIENT = [c for c in FRAMEWORKS if not c.self_flushing]
_SELF_FLUSH = [c for c in FRAMEWORKS if c.self_flushing]
_CASES = [(c, v) for c in _AMBIENT for v in (_VARIANTS if c.supports_redaction else ("default",))]
_IDS = [f"{c.id}-{v}" for c, v in _CASES]


def _skip_unless_ready(case) -> None:
    pytest.importorskip(case.import_name, reason=f"{case.id}: '{case.import_name}' not installed")
    for extra in case.extra_imports:
        pytest.importorskip(extra, reason=f"{case.id}: '{extra}' not installed")
    reason = missing_credentials(case)
    if reason:
        pytest.skip(f"{case.id}: {reason}")


@pytest.mark.live
@pytest.mark.parametrize("case, variant", _CASES, ids=_IDS)
def test_framework_live(case, variant, stratix_live_client, record_result) -> None:
    _skip_unless_ready(case)
    row = run_framework_case(stratix_live_client, case, variant)
    record_result(row)


@pytest.mark.live
@pytest.mark.parametrize("case", _SELF_FLUSH, ids=[c.id for c in _SELF_FLUSH])
def test_framework_live_self_flushing(case, stratix_live_client, record_result) -> None:
    """Adapters that manage + upload their own trace (e.g. openai_agents)."""
    _skip_unless_ready(case)
    row = run_self_flushing_case(stratix_live_client, case)
    record_result(row)
