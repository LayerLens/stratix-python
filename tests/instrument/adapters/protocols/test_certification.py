"""Tests for the protocol-adapter certification suite."""

from __future__ import annotations

from typing import Any

import pytest

from layerlens.instrument.adapters._base import AdapterInfo
from layerlens.instrument.adapters.protocols import (
    CheckResult,
    CertificationResult,
    ProtocolCertificationSuite,
)
from layerlens.instrument.adapters.protocols._base_protocol import BaseProtocolAdapter

# ---------------------------------------------------------------------------
# Shipped protocol adapters
# ---------------------------------------------------------------------------


class TestShippedAdapters:
    """The three shim adapters (a2ui, ap2, ucp) should certify cleanly."""

    @pytest.mark.parametrize(
        "module_path,class_name",
        [
            ("layerlens.instrument.adapters.protocols.a2ui", "A2UIProtocolAdapter"),
            ("layerlens.instrument.adapters.protocols.ap2", "AP2ProtocolAdapter"),
            ("layerlens.instrument.adapters.protocols.ucp", "UCPProtocolAdapter"),
        ],
    )
    def test_certifies(self, module_path, class_name):
        import importlib

        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        suite = ProtocolCertificationSuite()
        result = suite.certify(cls)
        assert isinstance(result, CertificationResult)
        # Print the result so a failing test surfaces every check
        if not result.passed:
            for check in result.checks:
                if not check.passed and check.severity == "error":
                    print(f"  FAIL [{check.severity}] {check.name}: {check.message}")
        assert result.passed, result.summary()


# ---------------------------------------------------------------------------
# Synthetic adapters — exercise each check independently
# ---------------------------------------------------------------------------


class _GoodAdapter(BaseProtocolAdapter):
    PROTOCOL = "test"
    PROTOCOL_VERSION = "1.2.3"

    def connect(self, target: Any = None, **kwargs: Any) -> Any:
        self._client = target
        return target


class TestIndividualChecks:
    def setup_method(self):
        self.suite = ProtocolCertificationSuite()

    def test_good_adapter_passes(self):
        result = self.suite.certify(_GoodAdapter)
        assert result.passed
        assert result.protocol == "test"
        assert result.protocol_version == "1.2.3"

    def test_missing_protocol_attr_fails(self):
        class _MissingProtocol(_GoodAdapter):
            PROTOCOL = ""

        result = self.suite.certify(_MissingProtocol)
        assert not result.passed
        assert any(c.name == "class_attr.PROTOCOL" and not c.passed for c in result.checks)

    def test_missing_protocol_version_fails(self):
        class _MissingVersion(_GoodAdapter):
            PROTOCOL_VERSION = ""

        result = self.suite.certify(_MissingVersion)
        assert not result.passed
        assert any(c.name == "class_attr.PROTOCOL_VERSION" and not c.passed for c in result.checks)

    def test_does_not_inherit_base_protocol_fails(self):
        class _StandaloneAdapter:
            PROTOCOL = "x"
            PROTOCOL_VERSION = "1.0.0"

            def connect(self, target=None, **kwargs):
                return target

            def disconnect(self):
                pass

            def adapter_info(self):
                return AdapterInfo(name="x", adapter_type="protocol")

        result = self.suite.certify(_StandaloneAdapter)
        assert not result.passed
        assert any(c.name == "inherits_base_protocol_adapter" and not c.passed for c in result.checks)
        assert any(c.name == "inherits_base_adapter" and not c.passed for c in result.checks)

    def test_adapter_info_wrong_type_fails(self):
        class _BadInfo(BaseProtocolAdapter):
            PROTOCOL = "test"
            PROTOCOL_VERSION = "1.0.0"

            def connect(self, target=None, **kwargs):
                return target

            def adapter_info(self):  # type: ignore[override]
                return {"name": "wrong"}  # dict instead of AdapterInfo

        result = self.suite.certify(_BadInfo)
        assert not result.passed
        # Find the failing adapter_info check
        info_check = [c for c in result.checks if c.name == "adapter_info.returns_adapter_info"][0]
        assert not info_check.passed
        assert "expected AdapterInfo" in info_check.message

    def test_adapter_info_wrong_adapter_type_fails(self):
        class _NotProtocolType(BaseProtocolAdapter):
            PROTOCOL = "test"
            PROTOCOL_VERSION = "1.0.0"

            def connect(self, target=None, **kwargs):
                return target

            def adapter_info(self):  # type: ignore[override]
                return AdapterInfo(name="x", adapter_type="framework")  # wrong

        result = self.suite.certify(_NotProtocolType)
        assert not result.passed
        info_check = [c for c in result.checks if c.name == "adapter_info.returns_adapter_info"][0]
        assert not info_check.passed
        assert "expected 'protocol'" in info_check.message

    def test_probe_health_wrong_type_warns(self):
        class _BadHealth(BaseProtocolAdapter):
            PROTOCOL = "test"
            PROTOCOL_VERSION = "1.0.0"

            def connect(self, target=None, **kwargs):
                return target

            def probe_health(self, endpoint=None):  # type: ignore[override]
                return {"reachable": True}  # dict instead of ProtocolHealth

        result = self.suite.certify(_BadHealth)
        health_check = [c for c in result.checks if c.name == "probe_health.returns_protocol_health"][0]
        assert not health_check.passed

    def test_negotiate_version_picks_exact_match(self):
        suite = ProtocolCertificationSuite()
        result = suite.certify(_GoodAdapter)
        # Find the negotiate_version check
        check = [c for c in result.checks if c.name == "negotiate_version.behavior"][0]
        assert check.passed

    def test_class_with_required_init_args_safely_skipped(self):
        class _RequiresArg(BaseProtocolAdapter):
            PROTOCOL = "test"
            PROTOCOL_VERSION = "1.0.0"

            def __init__(self, required_kwarg: str, **kwargs: Any) -> None:
                super().__init__(**kwargs)
                self._required = required_kwarg

            def connect(self, target=None, **kwargs):
                return target

        result = self.suite.certify(_RequiresArg)
        # Cannot instantiate -> adapter_info/probe_health/negotiate_version
        # all return warnings; should not crash.
        assert isinstance(result, CertificationResult)


# ---------------------------------------------------------------------------
# Bulk certification
# ---------------------------------------------------------------------------


class TestCertifyAll:
    def test_runs_against_a_list(self):
        suite = ProtocolCertificationSuite()
        results = suite.certify_all([_GoodAdapter, _GoodAdapter])
        assert len(results) == 2
        assert all(r.passed for r in results)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class TestResultTypes:
    def test_check_result_to_dict(self):
        c = CheckResult(name="t", passed=True, message="ok", severity="error")
        d = c.to_dict()
        assert d == {"name": "t", "passed": True, "message": "ok", "severity": "error"}

    def test_summary_format(self):
        suite = ProtocolCertificationSuite()
        result = suite.certify(_GoodAdapter)
        summary = result.summary()
        assert "PASSED" in summary
        assert "test" in summary
        assert "1.2.3" in summary

    def test_failing_summary_shows_failed_status(self):
        class _Bad(BaseProtocolAdapter):
            PROTOCOL = ""
            PROTOCOL_VERSION = ""

            def connect(self, target=None, **kwargs):
                return target

        suite = ProtocolCertificationSuite()
        result = suite.certify(_Bad)
        assert "FAILED" in result.summary()

    def test_certification_result_to_dict_serializes(self):
        import json

        suite = ProtocolCertificationSuite()
        result = suite.certify(_GoodAdapter)
        d = result.to_dict()
        # Round-trips through JSON
        json.dumps(d)
        assert d["passed"] is True
        assert d["protocol"] == "test"
        assert isinstance(d["checks"], list)
