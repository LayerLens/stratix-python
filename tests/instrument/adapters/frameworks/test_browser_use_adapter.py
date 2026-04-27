"""Full lifecycle tests for the browser_use adapter.

Replaces the 7-test placeholder scaffold from PR #116. Covers:

* Adapter wiring (export, alias, capabilities, info).
* Lifecycle (connect / disconnect / health_check).
* Each public hook (session, navigation, action, screenshot,
  DOM extraction, LLM call).
* Truncation policy (screenshot drop, HTML cap, short-payload
  no-audit).
* PR #117 resilience contract (a callback exception NEVER crashes
  the agent and bumps the resilience counter).
* PR #118 multi-tenancy contract (org_id stamped on every emit).
* PR #115 error-aware emission (framework exceptions surface as
  structured *.error events before re-raise).
* Replay round-trip (serialize_for_replay produces a valid
  ReplayableTrace).
* STRATIX legacy alias (DeprecationWarning on access).
* Sync + async wrapping.
"""

from __future__ import annotations

import asyncio
import warnings
from typing import Any, Dict, List, Optional

import pytest

from layerlens.instrument.adapters._base import (
    DEFAULT_POLICY,
    AdapterStatus,
    CaptureConfig,
    AdapterCapability,
)
from layerlens.instrument.adapters._base.pydantic_compat import PydanticCompat
from layerlens.instrument.adapters.frameworks.browser_use import (
    ADAPTER_CLASS,
    BrowserUseAdapter,
    instrument_agent,
)

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


class _RecordingStratix:
    """Stand-in LayerLens client that captures every emitted event."""

    def __init__(self, org_id: str = "org_test_42") -> None:
        self.events: List[Dict[str, Any]] = []
        self.org_id = org_id

    def emit(self, *args: Any, **kwargs: Any) -> None:
        if len(args) == 2 and isinstance(args[0], str):
            self.events.append({"event_type": args[0], "payload": args[1]})


class _FakeAction:
    def __init__(self, type_: str, target: Optional[str] = None) -> None:
        self.type = type_
        self.target = target


class _FakeUsage:
    def __init__(self, prompt: int, completion: int) -> None:
        self.input_tokens = prompt
        self.output_tokens = completion
        self.prompt_tokens = prompt
        self.completion_tokens = completion


class _FakeStep:
    def __init__(
        self,
        url: Optional[str] = None,
        action: Any = None,
        screenshot: Optional[bytes] = None,
        usage: Any = None,
    ) -> None:
        self.url = url
        self.action = action
        self.screenshot = screenshot
        self.model_usage = usage


class _FakeHistory:
    def __init__(self, steps: List[_FakeStep]) -> None:
        self.history = steps


class _FakeBrowser:
    def __init__(self) -> None:
        self.headless = True
        self.user_data_dir = "/tmp/profile"


class _FakeAgent:
    """Duck-typed browser_use Agent with the surface the adapter touches."""

    def __init__(
        self,
        name: str = "demo-bot",
        task: str = "go to example.com",
        model: str = "gpt-4o-mini",
        result: Any = None,
        async_mode: bool = True,
        raise_on_run: Optional[BaseException] = None,
    ) -> None:
        self.name = name
        self.task = task
        self.model = model
        self.browser = _FakeBrowser()
        self._result = result if result is not None else _FakeHistory(steps=[])
        self._async = async_mode
        self._raise = raise_on_run

        if async_mode:
            async def run(*args: Any, **kwargs: Any) -> Any:
                if self._raise:
                    raise self._raise
                return self._result

            self.run = run
        else:
            def run(*args: Any, **kwargs: Any) -> Any:
                if self._raise:
                    raise self._raise
                return self._result

            self.run = run


def _make_adapter(stratix: Optional[_RecordingStratix] = None) -> BrowserUseAdapter:
    client = stratix or _RecordingStratix()
    adapter = BrowserUseAdapter(
        stratix=client,
        capture_config=CaptureConfig.full(),
        org_id=client.org_id,
    )
    adapter.connect()
    return adapter


# ---------------------------------------------------------------------------
# Wiring & registration
# ---------------------------------------------------------------------------


def test_adapter_class_export() -> None:
    assert ADAPTER_CLASS is BrowserUseAdapter


def test_pydantic_compat_v2_only() -> None:
    """browser_use is a Pydantic v2 library — declaration must be honest."""
    assert BrowserUseAdapter.requires_pydantic == PydanticCompat.V2_ONLY


def test_get_adapter_info_full_capability_set() -> None:
    adapter = _make_adapter()
    info = adapter.get_adapter_info()
    assert info.framework == "browser_use"
    assert info.name == "BrowserUseAdapter"
    caps = set(info.capabilities)
    for required in (
        AdapterCapability.TRACE_TOOLS,
        AdapterCapability.TRACE_MODELS,
        AdapterCapability.TRACE_STATE,
        AdapterCapability.STREAMING,
        AdapterCapability.REPLAY,
    ):
        assert required in caps, f"missing {required}"
    assert "placeholder" not in info.description.lower()


def test_legacy_strix_alias_deprecation_warning() -> None:
    """Constructing the legacy STRATIX-branded class MUST emit DeprecationWarning.

    The alias is a subclass of BrowserUseAdapter so isinstance checks
    still work, while construction surfaces the deprecation.
    """
    from layerlens.instrument.adapters.frameworks.browser_use import (
        STRATIXBrowserUseAdapter,
    )

    assert issubclass(STRATIXBrowserUseAdapter, BrowserUseAdapter)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        legacy_instance = STRATIXBrowserUseAdapter(org_id="org_legacy")
    assert isinstance(legacy_instance, BrowserUseAdapter)
    assert any(issubclass(w.category, DeprecationWarning) for w in caught), (
        "STRATIXBrowserUseAdapter construction did not emit DeprecationWarning"
    )


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_connect_disconnect_round_trip() -> None:
    adapter = BrowserUseAdapter(org_id="org_lifecycle")
    adapter.connect()
    assert adapter.status == AdapterStatus.HEALTHY
    health = adapter.health_check()
    assert health.framework_name == "browser_use"
    assert health.adapter_version == "0.1.0"
    adapter.disconnect()
    assert adapter.status == AdapterStatus.DISCONNECTED


def test_disconnect_unwraps_agents_idempotent() -> None:
    adapter = _make_adapter()
    agent = _FakeAgent()
    original_run = agent.run
    adapter.instrument_agent(agent)
    assert agent.run is not original_run
    # Repeat instrumentation is a no-op.
    adapter.instrument_agent(agent)
    adapter.disconnect()
    # After disconnect the agent's run is restored.
    assert agent.run is original_run


# ---------------------------------------------------------------------------
# Truncation policy (CRITICAL for browser_use)
# ---------------------------------------------------------------------------


def test_truncation_policy_default_after_construction() -> None:
    adapter = BrowserUseAdapter(org_id="org_t")
    assert adapter._truncation_policy is DEFAULT_POLICY


def test_screenshot_dropped_to_hash_reference() -> None:
    stratix = _RecordingStratix()
    adapter = _make_adapter(stratix)
    fake_png = b"\x89PNG\r\n\x1a\n" + b"PIXEL_DATA" * 5000  # ~50 KB
    adapter.on_screenshot(screenshot=fake_png, url="https://example.com", session_id="s1")
    payload = stratix.events[-1]["payload"]
    assert payload["url"] == "https://example.com"
    assert isinstance(payload["screenshot"], str)
    assert payload["screenshot"].startswith("<dropped:screenshot:sha256:")
    audit = payload.get("_truncated_fields", [])
    assert any("screenshot:dropped" in entry for entry in audit)


def test_screenshot_hash_is_deterministic() -> None:
    """Same input bytes → same hash reference (replay correlation)."""
    stratix = _RecordingStratix()
    adapter = _make_adapter(stratix)
    fake_png = b"\x89PNG\r\n\x1a\n" + b"PIXEL_DATA" * 100
    adapter.on_screenshot(screenshot=fake_png, session_id="s1")
    adapter.on_screenshot(screenshot=fake_png, session_id="s2")
    first = stratix.events[0]["payload"]["screenshot"]
    second = stratix.events[1]["payload"]["screenshot"]
    assert first == second


def test_html_capped_to_16kb() -> None:
    stratix = _RecordingStratix()
    adapter = _make_adapter(stratix)
    big_html = "<div>noise</div>" * 5000  # ~80 KB
    adapter.on_dom_extraction(html=big_html, url="https://example.com", session_id="s1")
    payload = stratix.events[-1]["payload"]
    assert isinstance(payload["html"], str)
    assert len(payload["html"]) <= 16384 + 100  # cap + suffix


def test_short_payload_emits_without_audit() -> None:
    stratix = _RecordingStratix()
    adapter = _make_adapter(stratix)
    adapter.on_action(action_type="click", target="#submit", session_id="s1")
    # The browser.action emit comes first, then the mirrored tool.call.
    action_payload = stratix.events[0]["payload"]
    assert "_truncated_fields" not in action_payload


# ---------------------------------------------------------------------------
# Multi-tenant org_id propagation (PR #118 contract)
# ---------------------------------------------------------------------------


def test_org_id_stamped_on_every_emit() -> None:
    stratix = _RecordingStratix(org_id="org_acme")
    adapter = _make_adapter(stratix)
    adapter.on_navigation(url="https://example.com", session_id="s1")
    adapter.on_action(action_type="click", target="#a", session_id="s1")
    adapter.on_screenshot(screenshot=b"data", session_id="s1")
    adapter.on_dom_extraction(html="<html/>", session_id="s1")
    adapter.on_llm_call(model="gpt-4o-mini", tokens_prompt=10, tokens_completion=20, session_id="s1")
    assert stratix.events, "no events captured"
    for evt in stratix.events:
        assert evt["payload"].get("org_id") == "org_acme", (
            f"missing org_id on {evt['event_type']}: {evt['payload']}"
        )


def test_org_id_resolved_from_stratix_attribute() -> None:
    """When org_id kwarg is omitted, the adapter resolves from the client."""
    stratix = _RecordingStratix(org_id="org_from_client")
    adapter = BrowserUseAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    adapter.on_navigation(url="https://x.com", session_id="s1")
    assert stratix.events[-1]["payload"]["org_id"] == "org_from_client"


def test_caller_supplied_org_id_overwritten_defensively() -> None:
    """Cross-tenant leak prevention: caller payloads cannot override binding."""
    stratix = _RecordingStratix(org_id="org_legit")
    adapter = _make_adapter(stratix)
    # Simulate a caller trying to inject a different org_id.
    adapter._emit("browser.navigate", {"url": "https://x.com", "org_id": "org_attacker"})
    assert stratix.events[-1]["payload"]["org_id"] == "org_legit"


# ---------------------------------------------------------------------------
# Resilience contract (PR #117): observability errors NEVER crash the agent
# ---------------------------------------------------------------------------


def test_resilience_callback_exception_does_not_propagate() -> None:
    """A bug in our emit path MUST NOT raise out of a callback."""
    stratix = _RecordingStratix()

    class _PoisonStratix:
        org_id = "org_p"

        def emit(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("downstream broken")

    adapter = BrowserUseAdapter(
        stratix=_PoisonStratix(),
        capture_config=CaptureConfig.full(),
        org_id="org_p",
    )
    adapter.connect()
    # All hooks should swallow the downstream error.
    adapter.on_session_start(agent_name="x", task="t")
    adapter.on_navigation(url="https://x.com", session_id="s1")
    adapter.on_action(action_type="click", session_id="s1")
    adapter.on_screenshot(screenshot=b"x", session_id="s1")
    adapter.on_dom_extraction(html="<x/>", session_id="s1")
    adapter.on_llm_call(model="gpt-4o-mini", session_id="s1")
    adapter.on_session_end(agent_name="x", session_id="s1", output="ok")
    # The BaseAdapter circuit breaker counts emit failures; the
    # adapter-level resilience tracker counts handler exceptions.
    # In this scenario the inner emit raises, BaseAdapter swallows
    # it via _post_emit_failure — no resilience-handler exception
    # bubbles up to our hook level. The contract being asserted here
    # is "no exception escapes the public hook".


def test_resilience_tracker_counts_handler_failures() -> None:
    """When _safe_serialize / inner code raises, the counter MUST increment."""
    adapter = _make_adapter()

    # Force an exception inside _emit_agent_config by passing an agent
    # whose attribute access raises.
    class _ExplodingAgent:
        name = "boom"

        @property
        def model(self) -> Any:
            raise RuntimeError("attribute access exploded")

        @property
        def task(self) -> Any:
            raise RuntimeError("task exploded")

        @property
        def browser(self) -> Any:
            raise RuntimeError("browser exploded")

        @property
        def controller(self) -> Any:
            raise RuntimeError("controller exploded")

    # Direct call; instrument_agent calls _emit_agent_config which
    # walks attributes. The tracker bumps because attribute reads
    # raise inside the try block.
    try:
        adapter._emit_agent_config("boom", _ExplodingAgent())
    except Exception:  # pragma: no cover — must NOT escape
        pytest.fail("resilience contract violated: exception escaped hook")
    snap = adapter.resilience_snapshot()
    assert snap["resilience_failures_total"] >= 1
    assert "_emit_agent_config" in snap["resilience_failures_by_callback"]


# ---------------------------------------------------------------------------
# Error-aware emission (PR #115 contract)
# ---------------------------------------------------------------------------


def test_run_failure_emits_agent_error_before_reraise() -> None:
    """Async run() that raises MUST emit agent.error before propagating."""
    stratix = _RecordingStratix(org_id="org_err")
    adapter = _make_adapter(stratix)
    boom = ValueError("rate limit exceeded")
    agent = _FakeAgent(async_mode=True, raise_on_run=boom)
    adapter.instrument_agent(agent)

    with pytest.raises(ValueError, match="rate limit exceeded"):
        asyncio.run(agent.run())

    error_events = [e for e in stratix.events if e["event_type"] == "agent.error"]
    assert error_events, "agent.error event was not emitted"
    payload = error_events[0]["payload"]
    assert payload["error_type"] == "ValueError"
    assert "rate limit" in payload["error_message"]
    assert payload["org_id"] == "org_err"
    # Session boundary events still fire so the dashboard sees a complete pair.
    types = [e["event_type"] for e in stratix.events]
    assert "agent.input" in types
    assert "agent.output" in types  # emitted in finally block


def test_action_with_error_emits_tool_error() -> None:
    stratix = _RecordingStratix()
    adapter = _make_adapter(stratix)
    adapter.on_action(
        action_type="click",
        target="#missing",
        session_id="s1",
        error=RuntimeError("element not found"),
    )
    error_events = [e for e in stratix.events if e["event_type"] == "tool.error"]
    assert error_events, "tool.error not emitted on action failure"
    assert error_events[0]["payload"]["error_type"] == "RuntimeError"


def test_llm_call_with_error_emits_model_error() -> None:
    stratix = _RecordingStratix()
    adapter = _make_adapter(stratix)
    adapter.on_llm_call(
        model="gpt-4o-mini",
        session_id="s1",
        error=ConnectionError("API down"),
    )
    error_events = [e for e in stratix.events if e["event_type"] == "model.error"]
    assert error_events, "model.error not emitted on LLM failure"
    assert error_events[0]["payload"]["error_type"] == "ConnectionError"


# ---------------------------------------------------------------------------
# Per-hook coverage
# ---------------------------------------------------------------------------


def test_on_session_start_emits_session_and_input_events() -> None:
    stratix = _RecordingStratix()
    adapter = _make_adapter(stratix)
    sid = adapter.on_session_start(agent_name="bot", task="visit example.com")
    assert sid  # uuid returned
    types = [e["event_type"] for e in stratix.events]
    assert "browser.session.start" in types
    assert "agent.input" in types
    session_evt = next(e for e in stratix.events if e["event_type"] == "browser.session.start")
    assert session_evt["payload"]["session_id"] == sid


def test_on_session_end_emits_output_and_state_change() -> None:
    stratix = _RecordingStratix()
    adapter = _make_adapter(stratix)
    sid = adapter.on_session_start(agent_name="bot", task="t")
    adapter.on_session_end(agent_name="bot", session_id=sid, output="done")
    types = [e["event_type"] for e in stratix.events]
    assert "agent.output" in types
    assert "agent.state.change" in types
    state_evt = next(e for e in stratix.events if e["event_type"] == "agent.state.change")
    assert state_evt["payload"]["event_subtype"] == "session_complete"


def test_on_navigation_emits_browser_navigate() -> None:
    stratix = _RecordingStratix()
    adapter = _make_adapter(stratix)
    adapter.on_navigation(
        url="https://example.com/login",
        session_id="s1",
        referrer="https://example.com/",
        status_code=200,
    )
    nav = stratix.events[-1]
    assert nav["event_type"] == "browser.navigate"
    assert nav["payload"]["url"] == "https://example.com/login"
    assert nav["payload"]["referrer"] == "https://example.com/"
    assert nav["payload"]["status_code"] == 200


def test_on_action_emits_browser_action_and_tool_call() -> None:
    stratix = _RecordingStratix()
    adapter = _make_adapter(stratix)
    adapter.on_action(
        action_type="type",
        target="#email",
        value="user@example.com",
        session_id="s1",
        latency_ms=42.5,
    )
    types = [e["event_type"] for e in stratix.events]
    assert "browser.action" in types
    assert "tool.call" in types
    tc = next(e for e in stratix.events if e["event_type"] == "tool.call")
    assert tc["payload"]["tool_name"] == "browser.type"
    assert tc["payload"]["latency_ms"] == 42.5


def test_on_dom_extraction_emits_dom_event() -> None:
    stratix = _RecordingStratix()
    adapter = _make_adapter(stratix)
    adapter.on_dom_extraction(
        html="<html><body>hi</body></html>",
        url="https://example.com",
        session_id="s1",
        element_count=3,
    )
    dom_evt = stratix.events[-1]
    assert dom_evt["event_type"] == "browser.dom.extract"
    assert dom_evt["payload"]["element_count"] == 3
    assert dom_evt["payload"]["html"] == "<html><body>hi</body></html>"


def test_on_llm_call_emits_invoke_and_cost() -> None:
    stratix = _RecordingStratix()
    adapter = _make_adapter(stratix)
    adapter.on_llm_call(
        model="claude-opus-4",
        tokens_prompt=100,
        tokens_completion=200,
        latency_ms=512.0,
        session_id="s1",
    )
    types = [e["event_type"] for e in stratix.events]
    assert "model.invoke" in types
    assert "cost.record" in types
    invoke = next(e for e in stratix.events if e["event_type"] == "model.invoke")
    assert invoke["payload"]["provider"] == "anthropic"  # auto-detected
    cost = next(e for e in stratix.events if e["event_type"] == "cost.record")
    assert cost["payload"]["tokens_total"] == 300


def test_environment_config_emitted_once_per_agent() -> None:
    stratix = _RecordingStratix()
    adapter = _make_adapter(stratix)
    agent = _FakeAgent()
    adapter.instrument_agent(agent)
    adapter.instrument_agent(agent)  # second call must not re-emit
    cfg_events = [e for e in stratix.events if e["event_type"] == "environment.config"]
    assert len(cfg_events) == 1
    payload = cfg_events[0]["payload"]
    assert payload["agent_name"] == "demo-bot"
    assert payload["model"] == "gpt-4o-mini"
    assert payload["browser"]["type"] == "_FakeBrowser"
    assert payload["browser"]["headless"] is True


# ---------------------------------------------------------------------------
# Wrapping (sync + async)
# ---------------------------------------------------------------------------


def test_async_run_wrapping_full_lifecycle() -> None:
    stratix = _RecordingStratix(org_id="org_async")
    adapter = _make_adapter(stratix)
    history = _FakeHistory(steps=[
        _FakeStep(
            url="https://example.com",
            action=_FakeAction("click", target="#submit"),
            screenshot=b"\x89PNG_PIXEL_DATA" * 100,
            usage=_FakeUsage(prompt=50, completion=75),
        )
    ])
    agent = _FakeAgent(async_mode=True, result=history)
    adapter.instrument_agent(agent)
    out = asyncio.run(agent.run())
    assert out is history
    types = [e["event_type"] for e in stratix.events]
    assert "agent.input" in types
    assert "agent.output" in types
    assert "browser.navigate" in types
    assert "browser.action" in types
    assert "browser.screenshot" in types
    assert "model.invoke" in types


def test_sync_run_wrapping_full_lifecycle() -> None:
    stratix = _RecordingStratix(org_id="org_sync")
    adapter = _make_adapter(stratix)
    agent = _FakeAgent(async_mode=False, result=_FakeHistory(steps=[]))
    # Provide run_sync so the sync wrapper path is taken.
    agent.run_sync = agent.run
    adapter.instrument_agent(agent)
    out = agent.run_sync("say hi")  # type: ignore[attr-defined]
    assert out is not None
    types = [e["event_type"] for e in stratix.events]
    # Either the async-shaped run or the sync run_sync path emits the same set.
    assert "agent.input" in types
    assert "agent.output" in types


# ---------------------------------------------------------------------------
# Replay round-trip
# ---------------------------------------------------------------------------


def test_serialize_for_replay_round_trip() -> None:
    stratix = _RecordingStratix(org_id="org_replay")
    adapter = _make_adapter(stratix)
    adapter.on_navigation(url="https://example.com", session_id="s1")
    adapter.on_action(action_type="click", target="#go", session_id="s1")
    trace = adapter.serialize_for_replay()
    assert trace.adapter_name == "BrowserUseAdapter"
    assert trace.framework == "browser_use"
    assert trace.trace_id  # uuid
    assert trace.events  # populated from emit success
    assert trace.config["org_id"] == "org_replay"
    # Round-trip via model_dump.
    from layerlens._compat.pydantic import model_dump

    dumped = model_dump(trace)
    assert dumped["adapter_name"] == "BrowserUseAdapter"
    assert dumped["framework"] == "browser_use"
    assert isinstance(dumped["events"], list)


# ---------------------------------------------------------------------------
# Convenience helper
# ---------------------------------------------------------------------------


def test_instrument_agent_helper_returns_connected_adapter() -> None:
    stratix = _RecordingStratix(org_id="org_helper")
    agent = _FakeAgent()
    adapter = instrument_agent(agent, stratix=stratix, org_id="org_helper")
    try:
        assert adapter.is_connected
        assert adapter._org_id == "org_helper"
        assert agent in adapter._wrapped_agents
    finally:
        adapter.disconnect()


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model,expected_provider",
    [
        ("gpt-4o-mini", "openai"),
        ("o1-preview", "openai"),
        ("o3-mini", "openai"),
        ("claude-opus-4", "anthropic"),
        ("gemini-1.5-pro", "google"),
        ("mistral-large", "mistral"),
        ("mixtral-8x7b", "mistral"),
        ("llama-3.1", "meta"),
        ("command-r-plus", "cohere"),
        ("unknown-model", None),
    ],
)
def test_provider_detection_table(model: str, expected_provider: Optional[str]) -> None:
    stratix = _RecordingStratix()
    adapter = _make_adapter(stratix)
    adapter.on_llm_call(model=model, session_id="s1")
    invoke = next(e for e in stratix.events if e["event_type"] == "model.invoke")
    assert invoke["payload"].get("provider") == expected_provider
