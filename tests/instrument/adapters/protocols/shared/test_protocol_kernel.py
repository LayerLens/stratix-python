"""
Tests for the protocol adapter shared kernel.

Covers BaseProtocolAdapter, ProtocolConnectionPool, exceptions,
health probes, and protocol version negotiation.
"""

import pytest
from unittest.mock import MagicMock, patch

from layerlens.instrument.adapters.protocols.base import BaseProtocolAdapter
from layerlens.instrument.adapters.protocols.connection_pool import (
    ProtocolConnectionPool,
    ConnectionSlot,
)
from layerlens.instrument.adapters.protocols.exceptions import (
    ProtocolError,
    ProtocolConnectionError,
    ProtocolTimeoutError,
    A2ATaskError,
    MCPToolError,
    resolve_protocol_error,
)
from layerlens.instrument.adapters.protocols.health import (
    HealthProbeResult,
    probe_http_endpoint,
)
from layerlens.instrument.adapters._base import AdapterHealth, AdapterInfo, AdapterStatus, ReplayableTrace


# ---------------------------------------------------------------------------
# Concrete subclass for testing the ABC
# ---------------------------------------------------------------------------


class _TestProtocolAdapter(BaseProtocolAdapter):
    """Concrete implementation for testing."""

    FRAMEWORK = "test_protocol"
    PROTOCOL = "test"
    PROTOCOL_VERSION = "1.0.0"
    VERSION = "0.1.0"

    def connect(self):
        self._connected = True
        self._status = AdapterStatus.HEALTHY

    def disconnect(self):
        self._connected = False
        self._status = AdapterStatus.DISCONNECTED

    def get_adapter_info(self):
        return AdapterInfo(
            name="TestProtocolAdapter",
            version=self.VERSION,
            framework=self.FRAMEWORK,
        )

    def serialize_for_replay(self):
        return ReplayableTrace(
            adapter_name="TestProtocolAdapter",
            framework=self.FRAMEWORK,
            trace_id="test-trace",
            events=[],
        )

    def probe_health(self, endpoint=None):
        return {"reachable": True, "latency_ms": 1.0, "protocol_version": "1.0.0"}


# ---------------------------------------------------------------------------
# BaseProtocolAdapter tests
# ---------------------------------------------------------------------------


class TestBaseProtocolAdapter:
    def test_init_defaults(self):
        adapter = _TestProtocolAdapter()
        assert adapter.PROTOCOL == "test"
        assert adapter._max_connections == 10
        assert adapter._retry_max_attempts == 3
        assert adapter._pool_active_count == 0

    def test_connect_disconnect(self):
        adapter = _TestProtocolAdapter()
        adapter.connect()
        assert adapter.is_connected
        assert adapter.status == AdapterStatus.HEALTHY

        adapter.disconnect()
        assert not adapter.is_connected
        assert adapter.status == AdapterStatus.DISCONNECTED

    def test_health_check(self):
        adapter = _TestProtocolAdapter()
        adapter.connect()
        health = adapter.health_check()
        assert isinstance(health, AdapterHealth)
        assert health.status == AdapterStatus.HEALTHY

    def test_version_negotiation_exact_match(self):
        adapter = _TestProtocolAdapter()
        result = adapter.negotiate_version(["1.0.0", "2.0.0"])
        assert result == "1.0.0"

    def test_version_negotiation_major_match(self):
        adapter = _TestProtocolAdapter()
        result = adapter.negotiate_version(["1.1.0", "1.2.0"])
        assert result == "1.2.0"  # Highest in same major

    def test_version_negotiation_no_match(self):
        adapter = _TestProtocolAdapter()
        result = adapter.negotiate_version(["2.0.0", "3.0.0"])
        assert result is None

    def test_connection_pool_acquire_release(self):
        adapter = _TestProtocolAdapter(max_connections=2)
        conn1 = adapter._acquire_connection("endpoint1")
        assert conn1 is not None
        assert adapter._pool_active_count == 1

        conn2 = adapter._acquire_connection("endpoint2")
        assert conn2 is not None
        assert adapter._pool_active_count == 2

        # Pool exhausted
        conn3 = adapter._acquire_connection("endpoint3")
        assert conn3 is None

        adapter._release_connection("endpoint1")
        assert adapter._pool_active_count == 1

    def test_get_adapter_info(self):
        adapter = _TestProtocolAdapter()
        info = adapter.get_adapter_info()
        assert info.name == "TestProtocolAdapter"
        assert info.framework == "test_protocol"

    def test_serialize_for_replay(self):
        adapter = _TestProtocolAdapter()
        replay = adapter.serialize_for_replay()
        assert isinstance(replay, ReplayableTrace)
        assert replay.framework == "test_protocol"


# ---------------------------------------------------------------------------
# ProtocolConnectionPool tests
# ---------------------------------------------------------------------------


class TestProtocolConnectionPool:
    def test_acquire_slot(self):
        pool = ProtocolConnectionPool(max_per_endpoint=2, max_total=10)
        slot = pool.acquire("a2a", "http://agent1")
        assert slot is not None
        assert slot.active
        assert slot.protocol == "a2a"
        assert slot.endpoint == "http://agent1"

    def test_per_endpoint_limit(self):
        pool = ProtocolConnectionPool(max_per_endpoint=1, max_total=10)
        slot1 = pool.acquire("a2a", "http://agent1")
        assert slot1 is not None
        slot2 = pool.acquire("a2a", "http://agent1")
        assert slot2 is None

    def test_total_limit(self):
        pool = ProtocolConnectionPool(max_per_endpoint=5, max_total=2)
        pool.acquire("a2a", "http://agent1")
        pool.acquire("a2a", "http://agent2")
        slot3 = pool.acquire("a2a", "http://agent3")
        assert slot3 is None

    def test_release(self):
        pool = ProtocolConnectionPool(max_per_endpoint=1, max_total=10)
        slot = pool.acquire("a2a", "http://agent1")
        pool.release(slot)
        assert not slot.active

    def test_stats(self):
        pool = ProtocolConnectionPool(max_per_endpoint=5, max_total=10)
        pool.acquire("a2a", "http://agent1")
        pool.acquire("mcp", "http://server1")
        stats = pool.stats()
        assert stats["active"] == 2
        assert stats["max_total"] == 10

    def test_close_all(self):
        pool = ProtocolConnectionPool()
        pool.acquire("a2a", "http://agent1")
        pool.acquire("mcp", "http://server1")
        pool.close_all()
        assert pool.total_active == 0


# ---------------------------------------------------------------------------
# Exception tests
# ---------------------------------------------------------------------------


class TestProtocolExceptions:
    def test_base_protocol_error(self):
        err = ProtocolError("test error", protocol="a2a", error_code="E001")
        assert str(err) == "test error"
        assert err.protocol == "a2a"
        assert err.error_code == "E001"

    def test_a2a_task_error(self):
        err = A2ATaskError("task failed", task_id="task-123", error_code="-32001")
        assert err.task_id == "task-123"
        assert err.protocol == "a2a"

    def test_resolve_protocol_error(self):
        err = resolve_protocol_error("a2a", "-32001", "Task not found")
        assert isinstance(err, A2ATaskError)

    def test_resolve_unknown_error(self):
        err = resolve_protocol_error("unknown", "E999", "Unknown error")
        assert isinstance(err, ProtocolError)

    def test_mcp_tool_error(self):
        err = MCPToolError("tool failed", protocol="mcp")
        assert err.protocol == "mcp"


# ---------------------------------------------------------------------------
# Health probe tests
# ---------------------------------------------------------------------------


class TestHealthProbes:
    def test_health_probe_result_to_dict(self):
        result = HealthProbeResult(
            reachable=True,
            latency_ms=42.5,
            protocol_version="1.0.0",
            endpoint="http://example.com",
        )
        d = result.to_dict()
        assert d["reachable"] is True
        assert d["latency_ms"] == 42.5
        assert d["protocol_version"] == "1.0.0"
