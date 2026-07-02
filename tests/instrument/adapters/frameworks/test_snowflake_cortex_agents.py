"""Tests for the Snowflake Cortex Agents adapter.

The adapter parses the ``agent:run`` SSE stream, so the tests drive it two
ways: with pre-parsed ``(event, data)`` tuples (``ingest_stream``) and with a
canned raw-line stream through an overridden transport (``run``). No Snowflake
account or network is required — httpx is the only dependency and it is never
actually called.
"""

from __future__ import annotations

from typing import Any, List, Tuple, Optional

import pytest

import layerlens.instrument.adapters.frameworks.snowflake_cortex_agents as _mod
from layerlens.instrument._capture_config import CaptureConfig
from layerlens.instrument.adapters.frameworks.snowflake_cortex_agents import (
    SnowflakeCortexAgentsAdapter,
    _iter_sse,
    _row_count,
    _last_user_text,
    _normalize_messages,
)

from .conftest import find_event, find_events, capture_framework_trace

Event = Tuple[str, dict]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _enable_httpx(monkeypatch: Any) -> None:
    monkeypatch.setattr(_mod, "_HAS_HTTPX", True)


def _setup(
    mock_client: Any,
    *,
    config: Optional[CaptureConfig] = None,
    agent: Optional[str] = "DB.SCHEMA.MY_AGENT",
) -> tuple:
    """Return ``(adapter, uploaded)`` connected in ingest-only mode."""
    uploaded = capture_framework_trace(mock_client)
    adapter = SnowflakeCortexAgentsAdapter(mock_client, capture_config=config)
    adapter.connect(agent=agent)  # no account_url -> ingest-only
    return adapter, uploaded


def _text_delta(text: str, idx: int = 0) -> Event:
    return ("response.text.delta", {"content_index": idx, "text": text})


def _thinking_delta(text: str) -> Event:
    return ("response.thinking.delta", {"content_index": 0, "text": text})


def _tool_use(tool_use_id: str, name: str, type_: str, input_: Any) -> Event:
    return (
        "response.tool_use",
        {"content_index": 0, "tool_use_id": tool_use_id, "type": type_, "name": name, "input": input_},
    )


def _tool_result(tool_use_id: str, name: str, content: Any, status: str = "success") -> Event:
    return (
        "response.tool_result",
        {"content_index": 0, "tool_use_id": tool_use_id, "type": "tool_results", "name": name,
         "content": content, "status": status},
    )


def _analyst_delta(tool_use_id: str, **delta: Any) -> Event:
    return (
        "response.tool_result.analyst.delta",
        {"content_index": 0, "tool_use_id": tool_use_id, "tool_type": "cortex_analyst_text_to_sql",
         "tool_name": "analyst", "delta": delta},
    )


def _final(tokens_consumed: Optional[List[dict]] = None, warnings: Optional[list] = None) -> Event:
    metadata = {}
    if tokens_consumed is not None:
        metadata["usage"] = {"tokens_consumed": tokens_consumed}
    data = {"role": "assistant", "content": [], "metadata": metadata}
    if warnings is not None:
        data["warnings"] = warnings
    return ("response", data)


_REQUEST = {
    "messages": [{"role": "user", "content": [{"type": "text", "text": "Q3 sales by region?"}]}],
    "models": {"orchestration": "claude-3-5-sonnet"},
    "thread_id": 7,
}


# ---------------------------------------------------------------------------
# Module-level parsing helpers
# ---------------------------------------------------------------------------


class TestSSEParsing:
    def test_iter_sse_basic(self) -> None:
        lines = [
            "event: response.text.delta",
            'data: {"content_index": 0, "text": "Hello"}',
            "",
            "event: response",
            'data: {"role": "assistant"}',
            "",
        ]
        events = list(_iter_sse(lines))
        assert events == [
            ("response.text.delta", {"content_index": 0, "text": "Hello"}),
            ("response", {"role": "assistant"}),
        ]

    def test_iter_sse_multiline_data_and_comments(self) -> None:
        lines = [
            ": keep-alive comment",
            "event: response.thinking.delta",
            'data: {"content_index": 0,',
            'data:  "text": "thinking"}',
            "",
        ]
        assert list(_iter_sse(lines)) == [
            ("response.thinking.delta", {"content_index": 0, "text": "thinking"}),
        ]

    def test_iter_sse_skips_done_and_bad_json(self) -> None:
        lines = [
            "event: response.text.delta",
            "data: [DONE]",
            "",
            "event: response.text.delta",
            "data: not json",
            "",
        ]
        assert list(_iter_sse(lines)) == []

    def test_iter_sse_flushes_trailing_event_without_blank_line(self) -> None:
        lines = ["event: response", 'data: {"role": "assistant"}']
        assert list(_iter_sse(lines)) == [("response", {"role": "assistant"})]

    def test_iter_sse_accepts_bytes(self) -> None:
        lines = [b"event: response.text.delta", b'data: {"text": "hi"}', b""]
        assert list(_iter_sse(lines)) == [("response.text.delta", {"text": "hi"})]

    def test_last_user_text(self) -> None:
        assert _last_user_text(_normalize_messages("hi there")) == "hi there"
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "first"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "reply"}]},
            {"role": "user", "content": [{"type": "text", "text": "second"}]},
        ]
        assert _last_user_text(msgs) == "second"

    def test_normalize_messages_wraps_string(self) -> None:
        assert _normalize_messages("hello") == [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]}
        ]

    def test_row_count(self) -> None:
        assert _row_count({"data": [[1], [2], [3]]}) == 3
        assert _row_count({"resultSetMetaData": {"numRows": 42}}) == 42
        assert _row_count("nope") is None


# ---------------------------------------------------------------------------
# Ingestion → events
# ---------------------------------------------------------------------------


class TestIngest:
    def test_input_and_output(self, mock_client: Any) -> None:
        adapter, uploaded = _setup(mock_client)
        stream = [_text_delta("The answer "), _text_delta("is 42."), _final()]
        final = adapter.ingest_stream(stream, request=_REQUEST)

        events = uploaded["events"]
        inp = find_event(events, "agent.input")
        assert inp["payload"]["input"] == "Q3 sales by region?"
        assert inp["payload"]["model"] == "claude-3-5-sonnet"
        assert inp["payload"]["thread_id"] == 7

        out = find_event(events, "agent.output")
        assert out["payload"]["output"] == "The answer is 42."
        assert out["payload"]["latency_ms"] is not None
        assert isinstance(final, dict)

    def test_thinking_captured_as_reasoning(self, mock_client: Any) -> None:
        adapter, uploaded = _setup(mock_client)
        adapter.ingest_stream(
            [_thinking_delta("Let me "), _thinking_delta("reason."), _text_delta("done"), _final()],
            request=_REQUEST,
        )
        out = find_event(uploaded["events"], "agent.output")
        assert out["payload"]["reasoning"] == "Let me reason."

    def test_content_gating_off(self, mock_client: Any) -> None:
        adapter, uploaded = _setup(mock_client, config=CaptureConfig(capture_content=False))
        adapter.ingest_stream(
            [_text_delta("secret"), _analyst_delta("t1", sql="SELECT 1"),
             _tool_result("t1", "analyst", [{"json": {}}]), _final()],
            request=_REQUEST,
        )
        events = uploaded["events"]
        assert "input" not in find_event(events, "agent.input")["payload"]
        assert "output" not in find_event(events, "agent.output")["payload"]
        tc = find_event(events, "tool.call")
        assert "sql" not in tc["payload"]
        assert "output" not in tc["payload"]

    def test_tool_use_and_result_emit_single_tool_call(self, mock_client: Any) -> None:
        adapter, uploaded = _setup(mock_client)
        adapter.ingest_stream(
            [
                _tool_use("tu-1", "cortex_search", "cortex_search", {"query": "policies"}),
                _tool_result("tu-1", "cortex_search", [{"text": "doc-1"}], status="success"),
                _final(),
            ],
            request=_REQUEST,
        )
        calls = find_events(uploaded["events"], "tool.call")
        assert len(calls) == 1
        p = calls[0]["payload"]
        assert p["tool_name"] == "cortex_search"
        assert p["tool_type"] == "cortex_search"
        assert p["status"] == "success"
        assert p["input"] == {"query": "policies"}
        assert p["output"] == [{"text": "doc-1"}]

    def test_analyst_delta_surfaces_sql_and_row_count(self, mock_client: Any) -> None:
        adapter, uploaded = _setup(mock_client)
        adapter.ingest_stream(
            [
                _tool_use("a-1", "analyst", "cortex_analyst_text_to_sql", {"question": "sales?"}),
                _analyst_delta("a-1", sql="SELECT region, "),
                _analyst_delta("a-1", sql="SUM(sales) FROM t", sql_explanation="Sums sales by region",
                               query_id="01ab", result_set={"data": [[1], [2]]}),
                _tool_result("a-1", "analyst", [{"json": {"sql": "..."}}]),
                _final(),
            ],
            request=_REQUEST,
        )
        tc = find_event(uploaded["events"], "tool.call")
        p = tc["payload"]
        assert p["sql"] == "SELECT region, SUM(sales) FROM t"
        assert p["sql_explanation"] == "Sums sales by region"
        assert p["query_id"] == "01ab"
        assert p["num_rows"] == 2

    def test_row_count_surfaced_even_when_content_gated(self, mock_client: Any) -> None:
        adapter, uploaded = _setup(mock_client, config=CaptureConfig(capture_content=False))
        adapter.ingest_stream(
            [
                _analyst_delta("a-1", sql="SELECT 1", result_set={"data": [[1], [2], [3]]}),
                _tool_result("a-1", "analyst", [], status="success"),
                _final(),
            ],
            request=_REQUEST,
        )
        tc = find_event(uploaded["events"], "tool.call")
        assert tc["payload"]["num_rows"] == 3
        assert "sql" not in tc["payload"]

    def test_usage_emits_model_invoke_and_cost(self, mock_client: Any) -> None:
        adapter, uploaded = _setup(mock_client)
        adapter.ingest_stream(
            [_text_delta("hi"), _final(tokens_consumed=[
                {"model": "claude-3-5-sonnet", "input_tokens": 100, "output_tokens": 40},
            ])],
            request=_REQUEST,
        )
        mi = find_event(uploaded["events"], "model.invoke")
        assert mi["payload"]["model"] == "claude-3-5-sonnet"
        assert mi["payload"]["provider"] == "snowflake_cortex"
        assert mi["payload"]["tokens_prompt"] == 100
        assert mi["payload"]["tokens_completion"] == 40
        assert mi["payload"]["tokens_total"] == 140

        cost = find_event(uploaded["events"], "cost.record")
        assert cost["payload"]["tokens_total"] == 140
        assert cost["payload"]["model"] == "claude-3-5-sonnet"

    def test_no_usage_no_cost_events(self, mock_client: Any) -> None:
        adapter, uploaded = _setup(mock_client)
        adapter.ingest_stream([_text_delta("hi"), _final()], request=_REQUEST)
        assert find_events(uploaded["events"], "model.invoke") == []
        assert find_events(uploaded["events"], "cost.record") == []

    def test_error_event(self, mock_client: Any) -> None:
        adapter, uploaded = _setup(mock_client)
        adapter.ingest_stream(
            [("error", {"code": "429", "message": "rate limited", "request_id": "req-9"})],
            request=_REQUEST,
        )
        err = find_event(uploaded["events"], "agent.error")
        assert err["payload"]["code"] == "429"
        assert err["payload"]["request_id"] == "req-9"
        assert err["payload"]["message"] == "rate limited"

    def test_incomplete_tool_use_is_flushed(self, mock_client: Any) -> None:
        adapter, uploaded = _setup(mock_client)
        adapter.ingest_stream(
            [_tool_use("tu-x", "cortex_search", "cortex_search", {"q": "x"}), _final()],
            request=_REQUEST,
        )
        tc = find_event(uploaded["events"], "tool.call")
        assert tc["payload"]["status"] == "incomplete"
        assert tc["payload"]["tool_name"] == "cortex_search"

    def test_warnings_attached_to_output(self, mock_client: Any) -> None:
        adapter, uploaded = _setup(mock_client)
        adapter.ingest_stream(
            [_text_delta("ok"), _final(warnings=[{"message": "truncated"}])],
            request=_REQUEST,
        )
        out = find_event(uploaded["events"], "agent.output")
        assert out["payload"]["warnings"] == [{"message": "truncated"}]

    def test_raw_sse_lines_accepted_by_ingest_stream(self, mock_client: Any) -> None:
        adapter, uploaded = _setup(mock_client)
        lines = [
            "event: response.text.delta",
            'data: {"text": "raw"}',
            "",
            "event: response",
            'data: {"role": "assistant", "content": []}',
            "",
        ]
        adapter.ingest_stream(lines, request=_REQUEST)
        assert find_event(uploaded["events"], "agent.output")["payload"]["output"] == "raw"


# ---------------------------------------------------------------------------
# run() + transport + lifecycle
# ---------------------------------------------------------------------------


class TestRunAndLifecycle:
    def test_run_uses_transport_and_builds_url(self, mock_client: Any) -> None:
        uploaded = capture_framework_trace(mock_client)
        adapter = SnowflakeCortexAgentsAdapter(mock_client)
        adapter.connect(
            account_url="https://acct.snowflakecomputing.com/",
            auth_token="tok-123",
            agent="MY_DB.MY_SCHEMA.MY_AGENT",
        )
        captured: dict = {}

        def _fake_post_sse(url: str, headers: dict, body: dict, timeout: float):
            captured["url"] = url
            captured["headers"] = headers
            captured["body"] = body
            return [_text_delta("42"), _final(tokens_consumed=[{"model": "m", "input_tokens": 5, "output_tokens": 2}])]

        adapter._post_sse = _fake_post_sse  # type: ignore[method-assign]
        final = adapter.run("What is 6x7?", model="claude-3-5-sonnet")

        assert captured["url"] == (
            "https://acct.snowflakecomputing.com/api/v2/databases/MY_DB/schemas/MY_SCHEMA/agents/MY_AGENT:run"
        )
        assert captured["headers"]["Authorization"] == "Bearer tok-123"
        assert captured["headers"]["Accept"] == "text/event-stream"
        assert captured["body"]["stream"] is True
        assert captured["body"]["models"] == {"orchestration": "claude-3-5-sonnet"}
        assert captured["body"]["messages"][0]["content"][0]["text"] == "What is 6x7?"

        assert find_event(uploaded["events"], "agent.output")["payload"]["output"] == "42"
        assert find_events(uploaded["events"], "cost.record")
        assert isinstance(final, dict)
        adapter.disconnect()

    def test_run_stateless_url_when_no_agent(self, mock_client: Any) -> None:
        adapter = SnowflakeCortexAgentsAdapter(mock_client)
        adapter.connect(account_url="https://acct.snowflakecomputing.com", auth_token="t")
        captured: dict = {}

        def _fake(url: str, headers: dict, body: dict, timeout: float):
            captured["url"] = url
            return [_final()]

        adapter._post_sse = _fake  # type: ignore[method-assign]
        adapter.run("hi")
        assert captured["url"] == "https://acct.snowflakecomputing.com/api/v2/cortex/agent:run"

    def test_run_requires_transport(self, mock_client: Any) -> None:
        adapter, _ = _setup(mock_client)  # ingest-only, no account_url
        with pytest.raises(RuntimeError, match="HTTP transport"):
            adapter.run("hi")

    def test_run_requires_auth_token(self, mock_client: Any) -> None:
        adapter = SnowflakeCortexAgentsAdapter(mock_client)
        adapter.connect(account_url="https://acct.snowflakecomputing.com")  # no token
        with pytest.raises(ValueError, match="auth_token"):
            adapter.run("hi")

    def test_missing_httpx_raises(self, mock_client: Any, monkeypatch: Any) -> None:
        monkeypatch.setattr(_mod, "_HAS_HTTPX", False)
        with pytest.raises(ImportError, match="httpx"):
            SnowflakeCortexAgentsAdapter(mock_client).connect(account_url="https://x")

    def test_ingest_before_connect_raises(self, mock_client: Any) -> None:
        adapter = SnowflakeCortexAgentsAdapter(mock_client)
        with pytest.raises(RuntimeError, match="not connected"):
            adapter.ingest_stream([_final()])

    def test_disconnect_closes_owned_client(self, mock_client: Any) -> None:
        adapter = SnowflakeCortexAgentsAdapter(mock_client)
        adapter.connect(account_url="https://acct.snowflakecomputing.com", auth_token="t")
        http = adapter._http
        assert http is not None
        adapter.disconnect()
        assert adapter._http is None

    def test_user_supplied_transport_not_owned(self, mock_client: Any) -> None:
        supplied = _mod.httpx.Client()
        adapter = SnowflakeCortexAgentsAdapter(mock_client)
        adapter.connect(target=supplied, auth_token="t", account_url="https://acct.snowflakecomputing.com")
        assert adapter._http is supplied
        assert adapter._owns_http is False
        adapter.disconnect()
        supplied.close()

    def test_adapter_info(self, mock_client: Any) -> None:
        adapter, _ = _setup(mock_client)
        info = adapter.adapter_info()
        assert info.name == "snowflake_cortex_agents"
        assert info.adapter_type == "framework"
        assert info.connected is True
