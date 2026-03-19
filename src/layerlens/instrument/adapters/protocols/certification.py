"""
STRATIX Protocol Adapter GA Certification Suite

Validates that protocol adapters comply with the BaseProtocolAdapter contract
required for General Availability (GA) release. Checks interface compliance,
required attributes, error handling patterns, and lifecycle correctness.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Type

from layerlens.instrument.adapters._base import BaseAdapter
from layerlens.instrument.adapters.protocols.base import BaseProtocolAdapter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    """Result of a single certification check."""

    name: str
    passed: bool
    message: str
    severity: str = "error"  # "error" | "warning"


@dataclass
class CertificationResult:
    """Aggregate result for a single adapter's certification."""

    passed: bool
    adapter_name: str
    protocol_version: str
    checks: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        total = len(self.checks)
        passed = sum(1 for c in self.checks if c["passed"])
        status = "PASSED" if self.passed else "FAILED"
        return f"{self.adapter_name} GA certification: {status} ({passed}/{total} checks)"


# ---------------------------------------------------------------------------
# Required interface definitions
# ---------------------------------------------------------------------------

# Methods that BaseAdapter declares abstract — every adapter must implement these
_BASE_ADAPTER_REQUIRED_METHODS = [
    "connect",
    "disconnect",
    "health_check",
    "get_adapter_info",
    "serialize_for_replay",
]

# Methods that BaseProtocolAdapter declares abstract on top of BaseAdapter
_PROTOCOL_REQUIRED_METHODS = [
    "probe_health",
]

# Class attributes that protocol adapters must set to non-empty values
_REQUIRED_CLASS_ATTRIBUTES = [
    ("FRAMEWORK", str),
    ("PROTOCOL", str),
    ("PROTOCOL_VERSION", str),
    ("VERSION", str),
]


# ---------------------------------------------------------------------------
# Certification suite
# ---------------------------------------------------------------------------


class ProtocolCertificationSuite:
    """
    Runs GA certification checks against protocol adapter classes.

    Usage::

        suite = ProtocolCertificationSuite()
        result = suite.certify(A2AAdapter)
        assert result.passed

        results = suite.certify_all()
        assert all(r.passed for r in results)
    """

    def certify(self, adapter_class: Type) -> CertificationResult:
        """
        Run all certification checks on a single adapter class.

        Args:
            adapter_class: The adapter class to certify (not an instance).

        Returns:
            CertificationResult with all check outcomes.
        """
        checks: list[CheckResult] = []

        checks.append(self._check_inherits_base_protocol(adapter_class))
        checks.append(self._check_inherits_base_adapter(adapter_class))
        checks.extend(self._check_required_class_attributes(adapter_class))
        checks.extend(self._check_required_methods(adapter_class))
        checks.extend(self._check_lifecycle_correctness(adapter_class))
        checks.extend(self._check_error_handling(adapter_class))
        checks.append(self._check_adapter_info_returns_type(adapter_class))
        checks.append(self._check_probe_health_returns_dict(adapter_class))
        checks.append(self._check_serialize_for_replay_returns_type(adapter_class))

        all_passed = all(
            c.passed for c in checks if c.severity == "error"
        )

        # Derive adapter_name and protocol_version from the class
        adapter_name = getattr(adapter_class, "__name__", str(adapter_class))
        protocol_version = getattr(adapter_class, "PROTOCOL_VERSION", "unknown")

        return CertificationResult(
            passed=all_passed,
            adapter_name=adapter_name,
            protocol_version=protocol_version,
            checks=[
                {
                    "name": c.name,
                    "passed": c.passed,
                    "message": c.message,
                    "severity": c.severity,
                }
                for c in checks
            ],
        )

    def certify_all(self) -> list[CertificationResult]:
        """
        Certify all three GA protocol adapters: A2A, AG-UI, MCP Extensions.

        Returns:
            List of CertificationResult, one per adapter.
        """
        from layerlens.instrument.adapters.protocols.a2a.adapter import A2AAdapter
        from layerlens.instrument.adapters.protocols.agui.adapter import AGUIAdapter
        from layerlens.instrument.adapters.protocols.mcp.adapter import MCPExtensionsAdapter

        results = []
        for cls in (A2AAdapter, AGUIAdapter, MCPExtensionsAdapter):
            result = self.certify(cls)
            logger.info(result.summary())
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_inherits_base_protocol(self, cls: Type) -> CheckResult:
        ok = issubclass(cls, BaseProtocolAdapter)
        return CheckResult(
            name="inherits_BaseProtocolAdapter",
            passed=ok,
            message=(
                f"{cls.__name__} extends BaseProtocolAdapter"
                if ok
                else f"{cls.__name__} does NOT extend BaseProtocolAdapter"
            ),
        )

    def _check_inherits_base_adapter(self, cls: Type) -> CheckResult:
        ok = issubclass(cls, BaseAdapter)
        return CheckResult(
            name="inherits_BaseAdapter",
            passed=ok,
            message=(
                f"{cls.__name__} extends BaseAdapter"
                if ok
                else f"{cls.__name__} does NOT extend BaseAdapter"
            ),
        )

    def _check_required_class_attributes(self, cls: Type) -> list[CheckResult]:
        results = []
        for attr_name, expected_type in _REQUIRED_CLASS_ATTRIBUTES:
            value = getattr(cls, attr_name, None)
            ok = isinstance(value, expected_type) and bool(value)
            results.append(CheckResult(
                name=f"class_attr_{attr_name}",
                passed=ok,
                message=(
                    f"{attr_name} = {value!r}"
                    if ok
                    else f"{attr_name} is missing or empty (got {value!r})"
                ),
            ))
        return results

    def _check_required_methods(self, cls: Type) -> list[CheckResult]:
        results = []
        all_required = _BASE_ADAPTER_REQUIRED_METHODS + _PROTOCOL_REQUIRED_METHODS
        for method_name in all_required:
            has_method = hasattr(cls, method_name) and callable(getattr(cls, method_name))
            # Also check it is not still abstract (i.e., actually implemented)
            is_abstract = method_name in getattr(cls, "__abstractmethods__", set())
            ok = has_method and not is_abstract
            results.append(CheckResult(
                name=f"implements_{method_name}",
                passed=ok,
                message=(
                    f"{cls.__name__}.{method_name}() implemented"
                    if ok
                    else f"{cls.__name__}.{method_name}() missing or still abstract"
                ),
            ))
        return results

    def _check_lifecycle_correctness(self, cls: Type) -> list[CheckResult]:
        """Instantiate the adapter, run connect/disconnect, verify state transitions."""
        results = []
        if not issubclass(cls, BaseProtocolAdapter):
            results.append(CheckResult(
                name="instantiation",
                passed=False,
                message=f"{cls.__name__} is not a BaseProtocolAdapter subclass; skipping lifecycle checks",
            ))
            return results
        try:
            adapter = cls()
        except Exception as exc:
            results.append(CheckResult(
                name="instantiation",
                passed=False,
                message=f"Failed to instantiate {cls.__name__}: {exc}",
            ))
            return results

        results.append(CheckResult(
            name="instantiation",
            passed=True,
            message=f"{cls.__name__}() instantiated successfully",
        ))

        # Check initial state
        results.append(CheckResult(
            name="initial_state_disconnected",
            passed=not adapter.is_connected,
            message=(
                "Starts disconnected"
                if not adapter.is_connected
                else "Adapter should start disconnected"
            ),
        ))

        # Connect
        try:
            adapter.connect()
            results.append(CheckResult(
                name="connect_succeeds",
                passed=adapter.is_connected,
                message=(
                    "connect() sets is_connected=True"
                    if adapter.is_connected
                    else "connect() did not set is_connected=True"
                ),
            ))
        except Exception as exc:
            results.append(CheckResult(
                name="connect_succeeds",
                passed=False,
                message=f"connect() raised: {exc}",
            ))

        # Disconnect
        try:
            adapter.disconnect()
            results.append(CheckResult(
                name="disconnect_succeeds",
                passed=not adapter.is_connected,
                message=(
                    "disconnect() sets is_connected=False"
                    if not adapter.is_connected
                    else "disconnect() did not set is_connected=False"
                ),
            ))
        except Exception as exc:
            results.append(CheckResult(
                name="disconnect_succeeds",
                passed=False,
                message=f"disconnect() raised: {exc}",
            ))

        return results

    def _check_error_handling(self, cls: Type) -> list[CheckResult]:
        """Verify connect() handles missing framework imports gracefully."""
        results = []
        try:
            adapter = cls()
            # connect() should not raise even if the underlying framework
            # package is not installed — adapters must catch ImportError
            adapter.connect()
            results.append(CheckResult(
                name="connect_handles_missing_framework",
                passed=True,
                message="connect() handles missing framework gracefully",
            ))
            adapter.disconnect()
        except ImportError as exc:
            results.append(CheckResult(
                name="connect_handles_missing_framework",
                passed=False,
                message=f"connect() leaks ImportError: {exc}",
            ))
        except Exception:
            # Other exceptions are acceptable — the point is ImportError is caught
            results.append(CheckResult(
                name="connect_handles_missing_framework",
                passed=True,
                message="connect() does not leak ImportError",
            ))
        return results

    def _check_adapter_info_returns_type(self, cls: Type) -> CheckResult:
        """Verify get_adapter_info() returns AdapterInfo."""
        from layerlens.instrument.adapters._base import AdapterInfo

        try:
            adapter = cls()
            info = adapter.get_adapter_info()
            ok = isinstance(info, AdapterInfo)
            return CheckResult(
                name="get_adapter_info_returns_AdapterInfo",
                passed=ok,
                message=(
                    f"get_adapter_info() returns AdapterInfo(name={info.name!r})"
                    if ok
                    else f"get_adapter_info() returned {type(info).__name__}, expected AdapterInfo"
                ),
            )
        except Exception as exc:
            return CheckResult(
                name="get_adapter_info_returns_AdapterInfo",
                passed=False,
                message=f"get_adapter_info() raised: {exc}",
            )

    def _check_probe_health_returns_dict(self, cls: Type) -> CheckResult:
        """Verify probe_health() returns a dict with expected keys."""
        try:
            adapter = cls()
            result = adapter.probe_health()
            ok = (
                isinstance(result, dict)
                and "reachable" in result
                and "latency_ms" in result
                and "protocol_version" in result
            )
            return CheckResult(
                name="probe_health_returns_valid_dict",
                passed=ok,
                message=(
                    "probe_health() returns dict with reachable, latency_ms, protocol_version"
                    if ok
                    else f"probe_health() returned {result!r} — missing required keys"
                ),
            )
        except Exception as exc:
            return CheckResult(
                name="probe_health_returns_valid_dict",
                passed=False,
                message=f"probe_health() raised: {exc}",
            )

    def _check_serialize_for_replay_returns_type(self, cls: Type) -> CheckResult:
        """Verify serialize_for_replay() returns ReplayableTrace."""
        from layerlens.instrument.adapters._base import ReplayableTrace

        try:
            adapter = cls()
            trace = adapter.serialize_for_replay()
            ok = isinstance(trace, ReplayableTrace)
            return CheckResult(
                name="serialize_for_replay_returns_ReplayableTrace",
                passed=ok,
                message=(
                    f"serialize_for_replay() returns ReplayableTrace(adapter_name={trace.adapter_name!r})"
                    if ok
                    else f"serialize_for_replay() returned {type(trace).__name__}"
                ),
            )
        except Exception as exc:
            return CheckResult(
                name="serialize_for_replay_returns_ReplayableTrace",
                passed=False,
                message=f"serialize_for_replay() raised: {exc}",
            )
