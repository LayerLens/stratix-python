from __future__ import annotations

from typing import Any

from layerlens.instrument.adapters._base import AdapterInfo, BaseAdapter
from layerlens.instrument.adapters._registry import (
    get,
    register,
    _adapters,
    unregister,
    list_adapters,
    disconnect_all,
)


class StubAdapter(BaseAdapter):
    def __init__(self) -> None:
        self._connected = False

    def connect(self, target: Any = None, **kwargs: Any) -> Any:
        self._connected = True
        return target

    def disconnect(self) -> None:
        self._connected = False

    def adapter_info(self) -> AdapterInfo:
        return AdapterInfo(name="stub", adapter_type="provider", connected=self._connected)


class TestBaseAdapter:
    def test_is_connected_delegates_to_info(self):
        a = StubAdapter()
        assert not a.is_connected
        a.connect()
        assert a.is_connected

    def test_adapter_info_returns_dataclass(self):
        a = StubAdapter()
        info = a.adapter_info()
        assert info.name == "stub"
        assert info.adapter_type == "provider"


class TestRegistry:
    def setup_method(self):
        _adapters.clear()

    def teardown_method(self):
        _adapters.clear()

    def test_register_and_get(self):
        adapter = StubAdapter()
        register("test", adapter)
        assert get("test") is adapter

    def test_get_missing(self):
        assert get("nonexistent") is None

    def test_unregister(self):
        adapter = StubAdapter()
        adapter.connect()
        register("test", adapter)
        result = unregister("test")
        assert result is adapter
        assert not adapter.is_connected
        assert get("test") is None

    def test_unregister_missing(self):
        assert unregister("nonexistent") is None

    def test_register_replaces_existing(self):
        old = StubAdapter()
        old.connect()
        register("test", old)
        new = StubAdapter()
        register("test", new)
        assert get("test") is new
        assert not old.is_connected

    def test_list_adapters(self):
        a = StubAdapter()
        a.connect()
        b = StubAdapter()
        b.connect()
        register("a", a)
        register("b", b)
        infos = list_adapters()
        assert len(infos) == 2
        assert all(i.name == "stub" for i in infos)  # both are StubAdapter
        assert all(i.connected for i in infos)

    def test_disconnect_all(self):
        a = StubAdapter()
        a.connect()
        b = StubAdapter()
        b.connect()
        register("a", a)
        register("b", b)
        disconnect_all()
        assert not a.is_connected
        assert not b.is_connected
        assert list_adapters() == []

    def test_disconnect_all_empty_is_safe(self):
        disconnect_all()  # should not raise
