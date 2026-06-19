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

import os

import pytest

from . import _framework_scenarios as fs
from ._framework_harness import run_framework_case, run_self_flushing_case
from ._framework_registry import FRAMEWORKS, missing_credentials

_AMBIENT = [c for c in FRAMEWORKS if not c.self_flushing]
_SELF_FLUSH = [c for c in FRAMEWORKS if c.self_flushing]


def _variants(case) -> tuple:
    out = ["default"]
    if case.supports_redaction:
        out.append("redaction")
    if case.supports_error:
        out.append("error")
    return tuple(out)


_CASES = [(c, v) for c in _AMBIENT for v in _variants(c)]
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


@pytest.mark.live
def test_bedrock_agents_trace_completeness_live(stratix_live_client, record_result) -> None:
    """LAY-3606: the five trace-completeness members on the REAL wire.

    Triple-gated and opt-in: needs the live suite enabled, ``LAYERLENS_LIVE_BEDROCK_FEATURES=1``,
    and a Bedrock agent CONFIGURED with a guardrail + RETURN_CONTROL action group + code
    interpreter + user input (point ``BEDROCK_FEATURES_ALIAS_ID`` at that version/DRAFT and grant
    the agent role ``bedrock:ApplyGuardrail``). Skipped by default so standard live runs — which
    use a vanilla agent alias — are unaffected. ``reprompt`` is intentionally excluded (forcing it
    needs a custom parser Lambda; it stays doubles-covered).
    """
    if os.environ.get("LAYERLENS_LIVE_BEDROCK_FEATURES") != "1":
        pytest.skip("set LAYERLENS_LIVE_BEDROCK_FEATURES=1 with a feature-configured Bedrock agent")
    pytest.importorskip("boto3", reason="boto3 not installed")
    for env in ("BEDROCK_AGENT_ID", "BEDROCK_AGENT_ALIAS_ID"):
        if not os.environ.get(env):
            pytest.skip(f"bedrock_agents features: {env} not set")

    result = fs.run_bedrock_agents_features(stratix_live_client)
    seen = set(result["seen_types"])
    expected = {"policy.violation", "tool.call", "agent.code", "agent.step"}
    missing = expected - seen
    assert not missing, f"trace-completeness members missing on the wire: {sorted(missing)} (saw {sorted(seen)})"
    record_result(
        {
            "framework": "bedrock_agents",
            "variant": "trace-completeness",
            "verdict": "pass",
            "event_types": {t: 1 for t in result["seen_types"]},
            "trace_ids": result["trace_ids"],
        }
    )
