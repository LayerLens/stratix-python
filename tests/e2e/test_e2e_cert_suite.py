"""End-to-end: ProtocolCertificationSuite over real shipped adapters.

Exercises the suite by running it against the actual a2ui / ap2 / ucp
adapter classes, plus a deliberately-broken protocol adapter to prove
the suite catches contract violations.
"""

from __future__ import annotations

import json
from typing import Any

from layerlens.instrument.adapters._base import AdapterInfo
from layerlens.instrument.adapters.protocols import (
    AP2ProtocolAdapter,
    UCPProtocolAdapter,
    A2UIProtocolAdapter,
    BaseProtocolAdapter,
    CertificationResult,
    ProtocolCertificationSuite,
)


class TestShippedAdaptersCertify:
    """Every adapter we ship in `protocols/__init__.py` should certify cleanly."""

    def setup_method(self):
        self.suite = ProtocolCertificationSuite()

    def test_a2ui_certifies(self):
        result = self.suite.certify(A2UIProtocolAdapter)
        assert result.passed, _format_failures(result)
        assert result.protocol == "a2ui"

    def test_ap2_certifies(self):
        result = self.suite.certify(AP2ProtocolAdapter)
        assert result.passed, _format_failures(result)

    def test_ucp_certifies(self):
        result = self.suite.certify(UCPProtocolAdapter)
        assert result.passed, _format_failures(result)

    def test_certify_all_returns_results(self):
        results = self.suite.certify_all([A2UIProtocolAdapter, AP2ProtocolAdapter, UCPProtocolAdapter])
        assert len(results) == 3
        assert all(r.passed for r in results)
        # Bulk results round-trip through JSON for telemetry/CI consumption.
        json.dumps([r.to_dict() for r in results])


class TestBrokenAdapterDetected:
    """A deliberately non-compliant adapter must fail certification."""

    def test_class_without_required_attrs_fails(self):
        class _Broken(BaseProtocolAdapter):
            # PROTOCOL + PROTOCOL_VERSION intentionally left empty
            def connect(self, target: Any = None, **kwargs: Any) -> Any:
                return target

        suite = ProtocolCertificationSuite()
        result = suite.certify(_Broken)
        assert not result.passed
        failed_names = {c.name for c in result.checks if not c.passed}
        assert "class_attr.PROTOCOL" in failed_names
        assert "class_attr.PROTOCOL_VERSION" in failed_names

    def test_class_with_wrong_adapter_type_fails(self):
        class _NotProtocol(BaseProtocolAdapter):
            PROTOCOL = "x"
            PROTOCOL_VERSION = "1.0"

            def connect(self, target: Any = None, **kwargs: Any) -> Any:
                return target

            def adapter_info(self) -> AdapterInfo:
                return AdapterInfo(name="x", adapter_type="framework")  # wrong type

        suite = ProtocolCertificationSuite()
        result = suite.certify(_NotProtocol)
        assert not result.passed
        info_check = next(c for c in result.checks if c.name == "adapter_info.returns_adapter_info")
        assert not info_check.passed


def _format_failures(result: CertificationResult) -> str:
    lines = [result.summary()]
    for c in result.checks:
        if not c.passed and c.severity == "error":
            lines.append(f"  - {c.name}: {c.message}")
    return "\n".join(lines)
