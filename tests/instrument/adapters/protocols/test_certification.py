"""Unit tests for the protocol-adapter GA certification suite.

The suite (:class:`ProtocolCertificationSuite`) runs ~14+ structural
checks against an adapter class, including: BaseProtocolAdapter
inheritance, required class attributes, required methods, lifecycle
correctness, missing-framework graceful handling, and return-type
validation for ``get_adapter_info()`` / ``probe_health()`` /
``serialize_for_replay()``.

These tests exercise the suite end-to-end against the three GA-certified
adapters and synthesised passing/failing adapter classes.
"""

from __future__ import annotations

from typing import Any

import pytest

from layerlens.instrument.adapters._base.adapter import (
    AdapterInfo,
    AdapterHealth,
    AdapterStatus,
    ReplayableTrace,
)
from layerlens.instrument.adapters.protocols.a2a import A2AAdapter
from layerlens.instrument.adapters.protocols.mcp import MCPExtensionsAdapter
from layerlens.instrument.adapters.protocols.agui import AGUIAdapter
from layerlens.instrument.adapters.protocols.base import BaseProtocolAdapter
from layerlens.instrument.adapters.protocols.certification import (
    CheckResult,
    CertificationResult,
    ProtocolCertificationSuite,
)


def test_check_result_dataclass_defaults() -> None:
    cr = CheckResult(name="x", passed=True, message="ok")
    assert cr.severity == "error"
    assert cr.name == "x"
    assert cr.passed is True


def test_certification_result_summary_format() -> None:
    res = CertificationResult(
        passed=True,
        adapter_name="X",
        protocol_version="1.0.0",
        checks=[
            {"name": "a", "passed": True, "message": "", "severity": "error"},
            {"name": "b", "passed": False, "message": "", "severity": "warning"},
        ],
    )
    summary = res.summary()
    assert "X" in summary
    assert "GA certification" in summary
    # 1/2 passed (b is failing)
    assert "1/2" in summary


def test_certify_a2a_passes() -> None:
    suite = ProtocolCertificationSuite()
    result = suite.certify(A2AAdapter)
    assert isinstance(result, CertificationResult)
    assert result.adapter_name == "A2AAdapter"
    assert result.protocol_version == "0.2.1"
    assert result.passed, [c for c in result.checks if not c["passed"]]


def test_certify_agui_passes() -> None:
    suite = ProtocolCertificationSuite()
    result = suite.certify(AGUIAdapter)
    assert result.passed, [c for c in result.checks if not c["passed"]]
    assert result.adapter_name == "AGUIAdapter"


def test_certify_mcp_passes() -> None:
    suite = ProtocolCertificationSuite()
    result = suite.certify(MCPExtensionsAdapter)
    assert result.passed, [c for c in result.checks if not c["passed"]]
    assert result.adapter_name == "MCPExtensionsAdapter"


def test_certify_all_returns_three_results() -> None:
    suite = ProtocolCertificationSuite()
    results = suite.certify_all()
    assert len(results) == 3
    names = {r.adapter_name for r in results}
    assert names == {"A2AAdapter", "AGUIAdapter", "MCPExtensionsAdapter"}
    assert all(r.passed for r in results), [
        (r.adapter_name, c) for r in results for c in r.checks if not c["passed"]
    ]


def test_certify_results_contain_required_check_names() -> None:
    suite = ProtocolCertificationSuite()
    result = suite.certify(A2AAdapter)
    check_names = {c["name"] for c in result.checks}
    # Spot-check that the structural anchors are present
    assert "inherits_BaseProtocolAdapter" in check_names
    assert "inherits_BaseAdapter" in check_names
    assert "class_attr_FRAMEWORK" in check_names
    assert "class_attr_PROTOCOL_VERSION" in check_names
    assert "implements_connect" in check_names
    assert "implements_disconnect" in check_names
    assert "implements_probe_health" in check_names
    assert "instantiation" in check_names
    assert "initial_state_disconnected" in check_names
    assert "connect_succeeds" in check_names
    assert "disconnect_succeeds" in check_names
    assert "get_adapter_info_returns_AdapterInfo" in check_names
    assert "probe_health_returns_valid_dict" in check_names
    assert "serialize_for_replay_returns_ReplayableTrace" in check_names


def test_certify_check_count_meets_minimum() -> None:
    """The suite has at least 14 checks per adapter (5 base + 1 protocol
    methods + 4 class attributes + 4 lifecycle + 1 error handling +
    3 return-type checks = 18 nominal; tolerate growth)."""
    suite = ProtocolCertificationSuite()
    result = suite.certify(A2AAdapter)
    assert len(result.checks) >= 14


def test_certify_failing_adapter_reports_failure() -> None:
    """A class missing required class attributes / not extending
    BaseProtocolAdapter must FAIL certification."""

    class NotAnAdapter:
        pass

    suite = ProtocolCertificationSuite()
    result = suite.certify(NotAnAdapter)
    assert result.passed is False
    failed = [c for c in result.checks if not c["passed"] and c["severity"] == "error"]
    assert len(failed) > 0
    failed_names = {c["name"] for c in failed}
    assert "inherits_BaseProtocolAdapter" in failed_names


def test_certify_subclass_with_empty_class_attributes_fails() -> None:
    """A subclass that fails to override FRAMEWORK / PROTOCOL_VERSION
    is rejected because those values must be non-empty strings."""

    class _Incomplete(BaseProtocolAdapter):
        # Inherits empty-string defaults from BaseProtocolAdapter — a real
        # adapter MUST override these.
        def connect(self) -> None:
            self._connected = True
            self._status = AdapterStatus.HEALTHY

        def disconnect(self) -> None:
            self._connected = False
            self._status = AdapterStatus.DISCONNECTED

        def health_check(self) -> AdapterHealth:
            return AdapterHealth(
                status=self._status,
                framework_name=self.FRAMEWORK,
                adapter_version=self.VERSION,
            )

        def get_adapter_info(self) -> AdapterInfo:
            return AdapterInfo(name="x", version="0.0.0", framework=self.FRAMEWORK)

        def serialize_for_replay(self) -> ReplayableTrace:
            return ReplayableTrace(
                adapter_name="x", framework=self.FRAMEWORK, trace_id="t"
            )

        def probe_health(self, endpoint: str | None = None) -> dict[str, Any]:
            return {
                "reachable": self._connected,
                "latency_ms": 0.0,
                "protocol_version": None,
            }

    suite = ProtocolCertificationSuite()
    result = suite.certify(_Incomplete)
    failed_names = {c["name"] for c in result.checks if not c["passed"]}
    # FRAMEWORK / PROTOCOL / PROTOCOL_VERSION / VERSION are all empty strings
    # by default so all 4 class_attr checks must fail.
    assert "class_attr_FRAMEWORK" in failed_names
    assert "class_attr_PROTOCOL" in failed_names
    assert "class_attr_PROTOCOL_VERSION" in failed_names
    assert result.passed is False


def test_certify_handles_constructor_failures() -> None:
    """An adapter class whose constructor raises should cause an
    instantiation-failure check rather than crashing the suite."""

    class _Crashy(BaseProtocolAdapter):
        FRAMEWORK = "crashy"
        PROTOCOL = "crashy"
        PROTOCOL_VERSION = "1.0.0"
        VERSION = "0.0.0"

        def __init__(self, **kwargs: Any) -> None:
            raise RuntimeError("boom in __init__")

        def connect(self) -> None:  # pragma: no cover - never reached
            pass

        def disconnect(self) -> None:  # pragma: no cover - never reached
            pass

        def health_check(self) -> AdapterHealth:  # pragma: no cover
            return AdapterHealth(
                status=AdapterStatus.HEALTHY,
                framework_name=self.FRAMEWORK,
                adapter_version=self.VERSION,
            )

        def get_adapter_info(self) -> AdapterInfo:  # pragma: no cover
            return AdapterInfo(name="x", version="0.0.0", framework=self.FRAMEWORK)

        def serialize_for_replay(self) -> ReplayableTrace:  # pragma: no cover
            return ReplayableTrace(adapter_name="x", framework=self.FRAMEWORK, trace_id="t")

        def probe_health(self, endpoint: str | None = None) -> dict[str, Any]:  # pragma: no cover
            return {"reachable": False, "latency_ms": 0.0, "protocol_version": None}

    suite = ProtocolCertificationSuite()
    # Must NOT raise — instantiation-failure is captured as a CheckResult
    result = suite.certify(_Crashy)
    assert result.passed is False
    inst_check = next(c for c in result.checks if c["name"] == "instantiation")
    assert inst_check["passed"] is False
    assert "boom in __init__" in inst_check["message"]


@pytest.mark.parametrize("cls", [A2AAdapter, AGUIAdapter, MCPExtensionsAdapter])
def test_each_ga_adapter_has_no_failing_checks(cls: type) -> None:
    suite = ProtocolCertificationSuite()
    result = suite.certify(cls)
    failures = [c for c in result.checks if not c["passed"] and c["severity"] == "error"]
    assert failures == [], f"{cls.__name__} failed: {failures}"
