"""
browser_use adapter lifecycle (M7 placeholder).

The full instrumentation strategy will land in M7 — wrapping
``Browser.act`` / ``Browser.go_to_url`` / ``Browser.click`` etc. to
capture:

  * navigation events           → tool.call (L5a)
  * page screenshots / DOM      → tool.environment (L5c, but TRUNCATED)
  * model invocations           → model.invoke (L3)
  * agent task input/output     → agent.input / agent.output (L1)

This placeholder defines the minimum surface required by
:class:`AdapterRegistry` to load the adapter without
``ModuleNotFoundError`` AND wires the field-specific truncation
policy ahead of M7. See the package docstring for context.
"""

from __future__ import annotations

import uuid
import logging
from typing import Any

from layerlens.instrument.adapters._base.adapter import (
    AdapterInfo,
    BaseAdapter,
    AdapterHealth,
    AdapterStatus,
    ReplayableTrace,
    AdapterCapability,
)
from layerlens.instrument.adapters._base.truncation import DEFAULT_POLICY
from layerlens.instrument.adapters._base.pydantic_compat import PydanticCompat

logger = logging.getLogger(__name__)


class BrowserUseAdapter(BaseAdapter):
    """LayerLens adapter for browser_use (placeholder, full impl in M7).

    The placeholder declares the framework, version, and capability
    set so the registry's introspection paths work, AND wires the
    truncation policy from M5 so screenshot / image_data / html / dom
    payloads are correctly handled the moment instrumentation methods
    are added in M7.
    """

    FRAMEWORK = "browser_use"
    VERSION = "0.0.1-placeholder"
    # The placeholder source has no direct ``pydantic`` imports.
    # browser_use itself uses Pydantic v2 internally; the M7
    # implementation will revisit this declaration as appropriate.
    requires_pydantic = PydanticCompat.V1_OR_V2

    def __init__(
        self,
        stratix: Any | None = None,
        capture_config: Any | None = None,
        stratix_instance: Any | None = None,
    ) -> None:
        resolved = stratix or stratix_instance
        super().__init__(stratix=resolved, capture_config=capture_config)
        # Per-adapter wiring of the field-specific truncation policy
        # (cross-pollination audit §2.4 — CRITICAL for browser_use).
        # browser navigation captures multi-megabyte screenshots and
        # large DOM HTML payloads; without the policy a single emit
        # can blow past the ingestion sink limits.
        self._truncation_policy = DEFAULT_POLICY
        self._framework_version: str | None = None

    def connect(self) -> None:
        """Probe ``browser_use`` availability without importing the SDK heavily."""
        try:
            import browser_use  # type: ignore[import-not-found,unused-ignore]

            self._framework_version = getattr(browser_use, "__version__", "unknown")
        except ImportError:
            logger.debug("browser_use not installed")
        self._connected = True
        self._status = AdapterStatus.HEALTHY

    def disconnect(self) -> None:
        self._connected = False
        self._status = AdapterStatus.DISCONNECTED
        self._close_sinks()

    def health_check(self) -> AdapterHealth:
        return AdapterHealth(
            status=self._status,
            framework_name=self.FRAMEWORK,
            framework_version=self._framework_version,
            adapter_version=self.VERSION,
            error_count=self._error_count,
            circuit_open=self._circuit_open,
        )

    def get_adapter_info(self) -> AdapterInfo:
        return AdapterInfo(
            name="BrowserUseAdapter",
            version=self.VERSION,
            framework=self.FRAMEWORK,
            framework_version=self._framework_version,
            capabilities=[
                # Capabilities will broaden in M7 once instrumentation
                # methods land. The placeholder only claims
                # TRACE_TOOLS so the registry / capability lint is
                # consistent.
                AdapterCapability.TRACE_TOOLS,
            ],
            description="LayerLens adapter for browser_use (placeholder; full impl in M7)",
        )

    def serialize_for_replay(self) -> ReplayableTrace:
        return ReplayableTrace(
            adapter_name="BrowserUseAdapter",
            framework=self.FRAMEWORK,
            trace_id=str(uuid.uuid4()),
            events=list(self._trace_events),
            state_snapshots=[],
            config={"capture_config": self._capture_config.model_dump()},
        )
