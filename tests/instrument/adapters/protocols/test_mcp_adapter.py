"""Unit tests for the MCP Extensions adapter.

MCP emits ``tool.call`` (l5a), ``protocol.tool.structured_output`` (l5a),
``protocol.elicitation.request`` / ``.response`` (l5a),
``protocol.async_task`` (always-enabled), ``protocol.mcp_app.invocation``
(l5a), and ``environment.config`` (l4a). All of these pass the default
:class:`CaptureConfig` layer-gate, so the canonical
``_RecordingStratix`` pattern works.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from layerlens.instrument.adapters._base.adapter import (
    AdapterInfo,
    AdapterStatus,
    ReplayableTrace,
)
from layerlens.instrument.adapters.protocols.mcp import (
    ADAPTER_CLASS,
    MCPExtensionsAdapter,
)


class _RecordingStratix:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:
        if args:
            payload = args[0]
            self.events.append(
                {
                    "event_type": getattr(payload, "event_type", None),
                    "payload": payload,
                }
            )


def test_adapter_class_export() -> None:
    assert ADAPTER_CLASS is MCPExtensionsAdapter


def test_adapter_class_constants() -> None:
    assert MCPExtensionsAdapter.FRAMEWORK == "mcp_extensions"
    assert MCPExtensionsAdapter.PROTOCOL == "mcp"
    assert MCPExtensionsAdapter.PROTOCOL_VERSION == "1.0.0"


def test_lifecycle_transitions() -> None:
    adapter = MCPExtensionsAdapter()
    assert adapter.status == AdapterStatus.DISCONNECTED
    adapter.connect()
    assert adapter.status == AdapterStatus.HEALTHY
    adapter.disconnect()
    assert adapter.status == AdapterStatus.DISCONNECTED


def test_disconnect_clears_state() -> None:
    adapter = MCPExtensionsAdapter(stratix=_RecordingStratix())
    adapter.connect()
    adapter.on_async_task("task-1", status="created")
    assert "task-1" in adapter._async_tasks
    adapter.disconnect()
    assert adapter._async_tasks == {}
    assert adapter._originals == {}


def test_get_adapter_info_shape() -> None:
    adapter = MCPExtensionsAdapter()
    info = adapter.get_adapter_info()
    assert isinstance(info, AdapterInfo)
    assert info.framework == "mcp_extensions"
    assert info.name == "MCPExtensionsAdapter"


def test_probe_health_default_no_endpoint() -> None:
    adapter = MCPExtensionsAdapter()
    adapter.connect()
    h = adapter.probe_health()
    assert h["reachable"] is True
    assert "latency_ms" in h
    assert "protocol_version" in h


def test_on_tool_call_emits_tool_call_event() -> None:
    stratix = _RecordingStratix()
    adapter = MCPExtensionsAdapter(stratix=stratix)
    adapter.connect()

    adapter.on_tool_call(
        tool_name="search",
        input_data={"q": "weather"},
        output_data={"answer": "sunny"},
        latency_ms=12.3,
    )
    types = [e["event_type"] for e in stratix.events]
    assert "tool.call" in types
    payload = next(e for e in stratix.events if e["event_type"] == "tool.call")["payload"]
    assert payload.tool.name == "search"
    assert payload.input == {"q": "weather"}
    assert payload.output == {"answer": "sunny"}
    assert payload.latency_ms == pytest.approx(12.3)


def test_on_tool_call_with_error_records_error() -> None:
    stratix = _RecordingStratix()
    adapter = MCPExtensionsAdapter(stratix=stratix)
    adapter.connect()

    adapter.on_tool_call(
        tool_name="broken",
        input_data={"x": 1},
        error="connection refused",
    )
    payload = next(e for e in stratix.events if e["event_type"] == "tool.call")["payload"]
    assert payload.error == "connection refused"


def test_on_structured_output_emits_event() -> None:
    stratix = _RecordingStratix()
    adapter = MCPExtensionsAdapter(stratix=stratix)
    adapter.connect()

    adapter.on_structured_output(
        tool_name="get_user",
        output={"id": 1, "name": "alice"},
        schema={"$id": "schema:user/v1", "type": "object"},
        validation_passed=True,
    )
    types = [e["event_type"] for e in stratix.events]
    assert "protocol.tool.structured_output" in types
    payload = next(
        e for e in stratix.events if e["event_type"] == "protocol.tool.structured_output"
    )["payload"]
    assert payload.tool_name == "get_user"
    assert payload.schema_id == "schema:user/v1"
    assert payload.validation_passed is True


def test_on_structured_output_validation_failure() -> None:
    stratix = _RecordingStratix()
    adapter = MCPExtensionsAdapter(stratix=stratix)
    adapter.connect()

    adapter.on_structured_output(
        tool_name="get_user",
        output={"id": "not-int"},
        schema={"type": "object"},
        validation_passed=False,
        validation_errors=["id: must be integer"],
    )
    payload = next(
        e for e in stratix.events if e["event_type"] == "protocol.tool.structured_output"
    )["payload"]
    assert payload.validation_passed is False
    assert payload.validation_errors == ["id: must be integer"]


def test_elicitation_request_response_lifecycle() -> None:
    stratix = _RecordingStratix()
    adapter = MCPExtensionsAdapter(stratix=stratix)
    adapter.connect()

    adapter.on_elicitation_request(
        elicitation_id="el-1",
        server_name="auth-server",
        schema={"$id": "schema:cred/v1", "type": "object"},
        title="Credentials needed",
    )
    adapter.on_elicitation_response(
        elicitation_id="el-1",
        action="accepted",
        response={"username": "alice"},
        latency_ms=420.0,
    )

    types = [e["event_type"] for e in stratix.events]
    assert "protocol.elicitation.request" in types
    assert "protocol.elicitation.response" in types

    response = next(
        e for e in stratix.events if e["event_type"] == "protocol.elicitation.response"
    )["payload"]
    assert response.action == "accepted"
    assert response.latency_ms == pytest.approx(420.0)


def test_async_task_lifecycle_tracks_elapsed_time() -> None:
    stratix = _RecordingStratix()
    adapter = MCPExtensionsAdapter(stratix=stratix)
    adapter.connect()

    adapter.on_async_task("task-1", status="created")
    assert "task-1" in adapter._async_tasks

    adapter.on_async_task("task-1", status="completed")
    # Removed on terminal state
    assert "task-1" not in adapter._async_tasks

    types = [e["event_type"] for e in stratix.events]
    # Both submitted/completed are protocol.async_task (always-enabled)
    assert types.count("protocol.async_task") == 2
    completed = stratix.events[-1]["payload"]
    assert completed.status == "completed"
    assert completed.elapsed_ms is not None and completed.elapsed_ms >= 0


def test_async_task_failed_status_clears_tracker() -> None:
    stratix = _RecordingStratix()
    adapter = MCPExtensionsAdapter(stratix=stratix)
    adapter.connect()
    adapter.on_async_task("t", status="created")
    adapter.on_async_task("t", status="failed")
    assert "t" not in adapter._async_tasks


def test_mcp_app_invocation_emits_event() -> None:
    stratix = _RecordingStratix()
    adapter = MCPExtensionsAdapter(stratix=stratix)
    adapter.connect()

    adapter.on_mcp_app_invocation(
        app_id="app-1",
        component_type="form",
        interaction_result="submitted",
        parameters={"field": "value"},
        result={"ok": True},
    )
    types = [e["event_type"] for e in stratix.events]
    assert "protocol.mcp_app.invocation" in types


def test_auth_event_emits_environment_config() -> None:
    stratix = _RecordingStratix()
    adapter = MCPExtensionsAdapter(stratix=stratix)
    adapter.connect()

    adapter.on_auth_event(
        auth_type="oauth2.token_refresh",
        success=True,
        details={"scope": "read"},
    )
    types = [e["event_type"] for e in stratix.events]
    assert "environment.config" in types
    payload = next(e for e in stratix.events if e["event_type"] == "environment.config")["payload"]
    assert payload.environment.attributes["auth_event"] == "oauth2.token_refresh"
    assert payload.environment.attributes["auth_success"] is True


def test_serialize_for_replay_shape() -> None:
    adapter = MCPExtensionsAdapter(stratix=_RecordingStratix())
    adapter.connect()
    rt = adapter.serialize_for_replay()
    assert isinstance(rt, ReplayableTrace)
    assert rt.adapter_name == "MCPExtensionsAdapter"
    assert rt.framework == "mcp_extensions"
    assert "capture_config" in rt.config
