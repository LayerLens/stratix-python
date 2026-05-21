"""Protocol adapter certification suite.

Validates that a protocol adapter class meets the contract required for
GA: extends :class:`BaseProtocolAdapter`, sets ``PROTOCOL`` /
``PROTOCOL_VERSION``, implements the required lifecycle methods,
returns the right types from ``adapter_info`` / ``probe_health``, and
negotiates versions sensibly.

Usage::

    from layerlens.instrument.adapters.protocols import ProtocolCertificationSuite
    from layerlens.instrument.adapters.protocols.a2a.adapter import A2AAdapter

    suite = ProtocolCertificationSuite()
    result = suite.certify(A2AAdapter)
    print(result.summary())
    assert result.passed
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, List, Optional
from dataclasses import field, asdict, dataclass

from .._base import AdapterInfo, BaseAdapter
from ._base_protocol import ProtocolHealth, BaseProtocolAdapter

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class CheckResult:
    """Outcome of a single certification check."""

    name: str
    passed: bool
    message: str
    severity: str = "error"  # "error" | "warning"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CertificationResult:
    """Aggregate result for one adapter."""

    adapter_name: str
    protocol: str
    protocol_version: str
    passed: bool
    checks: List[CheckResult] = field(default_factory=list)

    def summary(self) -> str:
        total = len(self.checks)
        passed = sum(1 for c in self.checks if c.passed)
        errors = sum(1 for c in self.checks if not c.passed and c.severity == "error")
        warnings = sum(1 for c in self.checks if not c.passed and c.severity == "warning")
        status = "PASSED" if self.passed else "FAILED"
        return (
            f"{self.adapter_name} ({self.protocol} v{self.protocol_version}) "
            f"certification: {status} — {passed}/{total} checks ({errors} errors, {warnings} warnings)"
        )

    def to_dict(self) -> dict:
        return {
            "adapter_name": self.adapter_name,
            "protocol": self.protocol,
            "protocol_version": self.protocol_version,
            "passed": self.passed,
            "checks": [c.to_dict() for c in self.checks],
        }


# ---------------------------------------------------------------------------
# Required surface area
# ---------------------------------------------------------------------------

_REQUIRED_METHODS = ("connect", "disconnect", "adapter_info")
_OPTIONAL_RECOMMENDED_METHODS = ("probe_health", "negotiate_version")
_REQUIRED_CLASS_ATTRS = ("PROTOCOL", "PROTOCOL_VERSION")


# ---------------------------------------------------------------------------
# Suite
# ---------------------------------------------------------------------------


class ProtocolCertificationSuite:
    """Run the GA certification checks against protocol adapter classes."""

    def certify(self, adapter_class: type) -> CertificationResult:
        """Certify a single adapter class. Returns an aggregate result."""
        checks: List[CheckResult] = []
        checks.append(self._check_inherits_base_protocol(adapter_class))
        checks.append(self._check_inherits_base_adapter(adapter_class))
        checks.extend(self._check_required_class_attrs(adapter_class))
        checks.extend(self._check_required_methods(adapter_class))
        checks.extend(self._check_optional_methods(adapter_class))
        checks.append(self._check_adapter_info_shape(adapter_class))
        checks.append(self._check_probe_health_shape(adapter_class))
        checks.append(self._check_negotiate_version_logic(adapter_class))

        passed = all(c.passed for c in checks if c.severity == "error")
        protocol = getattr(adapter_class, "PROTOCOL", "") or ""
        protocol_version = getattr(adapter_class, "PROTOCOL_VERSION", "") or ""
        return CertificationResult(
            adapter_name=getattr(adapter_class, "__name__", str(adapter_class)),
            protocol=protocol,
            protocol_version=protocol_version,
            passed=passed,
            checks=checks,
        )

    def certify_all(self, adapter_classes: List[type]) -> List[CertificationResult]:
        """Run :meth:`certify` against multiple classes."""
        return [self.certify(cls) for cls in adapter_classes]

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_inherits_base_protocol(self, cls: type) -> CheckResult:
        ok = isinstance(cls, type) and issubclass(cls, BaseProtocolAdapter)
        return CheckResult(
            name="inherits_base_protocol_adapter",
            passed=bool(ok),
            message=("extends BaseProtocolAdapter" if ok else "does NOT extend BaseProtocolAdapter"),
        )

    def _check_inherits_base_adapter(self, cls: type) -> CheckResult:
        ok = isinstance(cls, type) and issubclass(cls, BaseAdapter)
        return CheckResult(
            name="inherits_base_adapter",
            passed=bool(ok),
            message="extends BaseAdapter" if ok else "does NOT extend BaseAdapter",
        )

    def _check_required_class_attrs(self, cls: type) -> List[CheckResult]:
        results: List[CheckResult] = []
        for attr in _REQUIRED_CLASS_ATTRS:
            value = getattr(cls, attr, "")
            ok = isinstance(value, str) and bool(value)
            results.append(
                CheckResult(
                    name=f"class_attr.{attr}",
                    passed=ok,
                    message=f"{attr}={value!r}" if ok else f"missing or empty {attr}",
                )
            )
        return results

    def _check_required_methods(self, cls: type) -> List[CheckResult]:
        results: List[CheckResult] = []
        for method in _REQUIRED_METHODS:
            ok = callable(getattr(cls, method, None))
            results.append(
                CheckResult(
                    name=f"method.{method}",
                    passed=ok,
                    message=("implemented" if ok else f"missing required method {method}"),
                )
            )
        return results

    def _check_optional_methods(self, cls: type) -> List[CheckResult]:
        results: List[CheckResult] = []
        for method in _OPTIONAL_RECOMMENDED_METHODS:
            present = callable(getattr(cls, method, None))
            results.append(
                CheckResult(
                    name=f"method.{method}",
                    passed=present,
                    message=("implemented" if present else f"missing recommended method {method}"),
                    severity="error" if not present else "error",
                )
            )
        return results

    def _check_adapter_info_shape(self, cls: type) -> CheckResult:
        """``adapter_info()`` should return an :class:`AdapterInfo` with
        ``adapter_type='protocol'``."""
        try:
            instance = self._safe_instantiate(cls)
            if instance is None:
                return CheckResult(
                    name="adapter_info.returns_adapter_info",
                    passed=False,
                    message="could not instantiate adapter for inspection",
                    severity="warning",
                )
            info = instance.adapter_info()
        except Exception as exc:
            return CheckResult(
                name="adapter_info.returns_adapter_info",
                passed=False,
                message=f"adapter_info() raised: {exc}",
            )
        if not isinstance(info, AdapterInfo):
            return CheckResult(
                name="adapter_info.returns_adapter_info",
                passed=False,
                message=f"adapter_info() returned {type(info).__name__}, expected AdapterInfo",
            )
        if info.adapter_type != "protocol":
            return CheckResult(
                name="adapter_info.returns_adapter_info",
                passed=False,
                message=f"adapter_info().adapter_type={info.adapter_type!r}, expected 'protocol'",
            )
        return CheckResult(
            name="adapter_info.returns_adapter_info",
            passed=True,
            message=f"AdapterInfo(name={info.name!r}, type='protocol', version={info.version!r})",
        )

    def _check_probe_health_shape(self, cls: type) -> CheckResult:
        """``probe_health()`` should return a :class:`ProtocolHealth`."""
        try:
            instance = self._safe_instantiate(cls)
            if instance is None:
                return CheckResult(
                    name="probe_health.returns_protocol_health",
                    passed=False,
                    message="could not instantiate adapter for inspection",
                    severity="warning",
                )
            health = instance.probe_health()
        except Exception as exc:
            return CheckResult(
                name="probe_health.returns_protocol_health",
                passed=False,
                message=f"probe_health() raised: {exc}",
                severity="warning",
            )
        if not isinstance(health, ProtocolHealth):
            return CheckResult(
                name="probe_health.returns_protocol_health",
                passed=False,
                message=f"probe_health() returned {type(health).__name__}, expected ProtocolHealth",
            )
        return CheckResult(
            name="probe_health.returns_protocol_health",
            passed=True,
            message=f"ProtocolHealth(reachable={health.reachable})",
        )

    def _check_negotiate_version_logic(self, cls: type) -> CheckResult:
        """``negotiate_version`` should pick the exact version when offered,
        or fall back to a major-version match."""
        try:
            instance = self._safe_instantiate(cls)
            if instance is None:
                return CheckResult(
                    name="negotiate_version.behavior",
                    passed=False,
                    message="could not instantiate adapter",
                    severity="warning",
                )
            own = getattr(cls, "PROTOCOL_VERSION", "")
            if not own:
                return CheckResult(
                    name="negotiate_version.behavior",
                    passed=False,
                    message="PROTOCOL_VERSION not set; cannot test negotiate_version",
                )
            picked = instance.negotiate_version([own])
            if picked != own:
                return CheckResult(
                    name="negotiate_version.behavior",
                    passed=False,
                    message=f"with exact match in server list, picked {picked!r} not own version {own!r}",
                )
            none_picked = instance.negotiate_version(["nonexistent-99.99.99"])
            if none_picked is not None and not none_picked.startswith(own.split(".")[0]):
                return CheckResult(
                    name="negotiate_version.behavior",
                    passed=False,
                    message=f"with no match, picked unrelated version {none_picked!r}",
                )
            return CheckResult(
                name="negotiate_version.behavior",
                passed=True,
                message="exact-match + no-match behavior correct",
            )
        except Exception as exc:
            return CheckResult(
                name="negotiate_version.behavior",
                passed=False,
                message=f"negotiate_version raised: {exc}",
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_instantiate(target_cls: type) -> Optional[Any]:
        """Construct an instance without arguments if possible.

        Protocol adapters typically take only optional kwargs in
        ``__init__``. If construction needs required args we return
        ``None`` and downstream checks skip with a warning.

        ``BaseProtocolAdapter.__init__`` creates an ``asyncio.Semaphore``
        which historically needed an event loop. Ensure one exists so
        we can instantiate even after a previous test closed its loop.
        """
        try:
            sig = inspect.signature(target_cls.__init__)
            for name, param in sig.parameters.items():
                if name == "self":
                    continue
                if param.default is inspect.Parameter.empty and param.kind not in (
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                ):
                    # Required arg with no default — bail.
                    return None

            # Guarantee an event loop for asyncio-touching constructors.
            import asyncio

            try:
                asyncio.get_event_loop()
            except RuntimeError:
                asyncio.set_event_loop(asyncio.new_event_loop())

            return target_cls()
        except Exception as exc:
            log.debug(
                "layerlens.certification: instantiation failed for %s: %s",
                target_cls.__name__,
                exc,
            )
            return None
