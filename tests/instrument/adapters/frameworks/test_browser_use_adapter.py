"""Truncation-policy + placeholder-shape tests for browser_use adapter.

The full browser_use adapter lands in M7. The placeholder under
``layerlens.instrument.adapters.frameworks.browser_use`` exists today
to (a) satisfy the AdapterRegistry entry without ``ModuleNotFoundError``
and (b) wire the field-specific truncation policy ahead of M7 — see
cross-pollination audit §2.4 (CRITICAL for browser_use).

These tests validate the pre-wiring contract.
"""

from __future__ import annotations

from typing import Any, Dict, List

from layerlens.instrument.adapters._base import (
    DEFAULT_POLICY,
    AdapterStatus,
    CaptureConfig,
    AdapterCapability,
)
from layerlens.instrument.adapters.frameworks.browser_use import (
    ADAPTER_CLASS,
    BrowserUseAdapter,
)


class _RecordingStratix:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:
        if len(args) == 2 and isinstance(args[0], str):
            self.events.append({"event_type": args[0], "payload": args[1]})


def test_adapter_class_export() -> None:
    assert ADAPTER_CLASS is BrowserUseAdapter


def test_lifecycle_round_trip() -> None:
    """``connect`` → ``disconnect`` round-trip works without M7 SDK."""
    adapter = BrowserUseAdapter()
    adapter.connect()
    assert adapter.status == AdapterStatus.HEALTHY
    adapter.disconnect()
    assert adapter.status == AdapterStatus.DISCONNECTED


def test_adapter_info_marks_placeholder() -> None:
    adapter = BrowserUseAdapter()
    info = adapter.get_adapter_info()
    assert info.framework == "browser_use"
    assert "placeholder" in info.description
    assert AdapterCapability.TRACE_TOOLS in info.capabilities


def test_truncation_policy_is_default_after_construction() -> None:
    """Pre-M7: the placeholder MUST already wire DEFAULT_POLICY.

    Without this the moment M7 adds instrumentation methods (page
    navigation, screenshot capture, DOM inspection) the events would
    flow to ``Stratix.emit`` with multi-megabyte screenshot bytes
    embedded directly.
    """
    adapter = BrowserUseAdapter()
    assert adapter._truncation_policy is DEFAULT_POLICY


def test_screenshot_dropped_to_hash_reference() -> None:
    """Screenshots become deterministic SHA-256 references."""
    stratix = _RecordingStratix()
    adapter = BrowserUseAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    fake_png = b"\x89PNG\r\n\x1a\n" + b"PIXEL_DATA" * 5000  # ~50 KB blob
    adapter.emit_dict_event(
        "tool.call",
        {"tool_name": "browser.screenshot", "screenshot": fake_png},
    )

    payload = stratix.events[-1]["payload"]
    assert isinstance(payload["screenshot"], str)
    assert payload["screenshot"].startswith("<dropped:screenshot:sha256:")
    audit = payload.get("_truncated_fields", [])
    assert any("screenshot:dropped" in entry for entry in audit)


def test_html_body_truncated_to_16kb_cap() -> None:
    """Browser_use DOM/HTML payloads are capped at 16 KiB by default."""
    stratix = _RecordingStratix()
    adapter = BrowserUseAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    big_html = "<div>noise</div>" * 5000  # ~80 KB
    adapter.emit_dict_event(
        "tool.call",
        {"tool_name": "browser.snapshot_dom", "html": big_html},
    )
    payload = stratix.events[-1]["payload"]
    assert isinstance(payload["html"], str)
    # 16384 cap + suffix length.
    assert len(payload["html"]) <= 16384 + 100


def test_short_payload_not_audited() -> None:
    """Short payloads emit unchanged with no audit list attached."""
    stratix = _RecordingStratix()
    adapter = BrowserUseAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    adapter.emit_dict_event(
        "tool.call", {"tool_name": "browser.click", "url": "https://example.com"}
    )
    payload = stratix.events[-1]["payload"]
    assert "_truncated_fields" not in payload
