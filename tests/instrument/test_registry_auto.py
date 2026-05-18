"""Tests for AdapterRegistry auto-detection (``discover_installed`` + ``auto``)."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from layerlens.instrument import auto, discover_installed
from layerlens.instrument.adapters._registry import (
    _PROVIDER_PACKAGES,
    _FRAMEWORK_ADAPTERS,
    _FRAMEWORK_PACKAGES,
    get,
    _adapters,
    disconnect_all,
)


@pytest.fixture(autouse=True)
def _clear_registry():
    """Wipe the module-level registry before/after each test."""
    disconnect_all()
    _adapters.clear()
    yield
    disconnect_all()
    _adapters.clear()


class TestDiscoverInstalled:
    def test_returns_split_dict(self):
        with patch("layerlens.instrument.adapters._registry._is_installed", return_value=False):
            result = discover_installed()
        assert set(result.keys()) == {"frameworks", "providers"}
        assert result["frameworks"] == []
        assert result["providers"] == []

    def test_detects_installed_packages(self):
        installed = {"langchain_core", "openai", "anthropic"}

        def fake_is_installed(pkg: str) -> bool:
            return pkg in installed

        with patch("layerlens.instrument.adapters._registry._is_installed", side_effect=fake_is_installed):
            result = discover_installed()

        assert "langchain" in result["frameworks"]
        assert "openai" in result["providers"]
        assert "anthropic" in result["providers"]
        # Not installed -> not present
        assert "crewai" not in result["frameworks"]
        assert "bedrock" not in result["providers"]

    def test_results_are_sorted(self):
        # Pretend everything is installed
        with patch("layerlens.instrument.adapters._registry._is_installed", return_value=True):
            result = discover_installed()
        assert result["frameworks"] == sorted(result["frameworks"])
        assert result["providers"] == sorted(result["providers"])


class TestAuto:
    def test_skips_when_nothing_installed(self):
        client = Mock()
        with patch("layerlens.instrument.adapters._registry._is_installed", return_value=False):
            connected = auto(client)
        assert connected == {}

    def test_wires_only_installed_frameworks(self):
        client = Mock()

        # Only langchain_core is "installed"
        def fake_is_installed(pkg: str) -> bool:
            return pkg == "langchain_core"

        # Fake adapter — instantiated with (client) and supports connect()
        fake_adapter_instance = Mock()
        fake_adapter_cls = Mock(return_value=fake_adapter_instance)
        fake_module = Mock()
        fake_module.LangChainCallbackHandler = fake_adapter_cls

        with patch("layerlens.instrument.adapters._registry._is_installed", side_effect=fake_is_installed), patch(
            "layerlens.instrument.adapters._registry.importlib.import_module", return_value=fake_module
        ):
            connected = auto(client)

        assert "langchain" in connected
        assert "crewai" not in connected
        fake_adapter_cls.assert_called_once_with(client)
        fake_adapter_instance.connect.assert_called_once_with()
        # registered globally
        assert get("langchain") is fake_adapter_instance

    def test_skip_parameter_excludes_named_adapters(self):
        client = Mock()
        fake_adapter_cls = Mock(return_value=Mock())
        fake_module = Mock()
        fake_module.LangChainCallbackHandler = fake_adapter_cls
        fake_module.CrewAIAdapter = fake_adapter_cls

        with patch("layerlens.instrument.adapters._registry._is_installed", return_value=True), patch(
            "layerlens.instrument.adapters._registry.importlib.import_module", return_value=fake_module
        ):
            connected = auto(client, skip=["langchain"])

        assert "langchain" not in connected
        # All other detectable frameworks should be present
        assert "crewai" in connected

    def test_connect_failure_is_logged_and_skipped(self, caplog):
        client = Mock()

        def fake_is_installed(pkg: str) -> bool:
            return pkg == "langchain_core"

        # connect() raises -> adapter must NOT appear in the result
        broken_instance = Mock()
        broken_instance.connect.side_effect = RuntimeError("boom")
        broken_cls = Mock(return_value=broken_instance)
        fake_module = Mock()
        fake_module.LangChainCallbackHandler = broken_cls

        with patch("layerlens.instrument.adapters._registry._is_installed", side_effect=fake_is_installed), patch(
            "layerlens.instrument.adapters._registry.importlib.import_module", return_value=fake_module
        ):
            connected = auto(client)

        assert connected == {}
        assert get("langchain") is None
        assert any("langchain" in rec.message for rec in caplog.records)

    def test_capture_config_passed_through_when_provided(self):
        client = Mock()
        fake_config = Mock()
        fake_adapter_cls = Mock(return_value=Mock())
        fake_module = Mock()
        fake_module.LangChainCallbackHandler = fake_adapter_cls

        def fake_is_installed(pkg: str) -> bool:
            return pkg == "langchain_core"

        with patch("layerlens.instrument.adapters._registry._is_installed", side_effect=fake_is_installed), patch(
            "layerlens.instrument.adapters._registry.importlib.import_module", return_value=fake_module
        ):
            auto(client, capture_config=fake_config)

        fake_adapter_cls.assert_called_once_with(client, capture_config=fake_config)


class TestRegistryTablesAreConsistent:
    """Guard against drift between the three module-level mappings."""

    def test_every_framework_adapter_has_a_package(self):
        for name in _FRAMEWORK_ADAPTERS:
            assert name in _FRAMEWORK_PACKAGES, f"{name} is in _FRAMEWORK_ADAPTERS but missing from _FRAMEWORK_PACKAGES"

    def test_every_framework_package_has_an_adapter(self):
        for name in _FRAMEWORK_PACKAGES:
            assert name in _FRAMEWORK_ADAPTERS, f"{name} is in _FRAMEWORK_PACKAGES but missing from _FRAMEWORK_ADAPTERS"

    def test_no_overlap_between_framework_and_provider_keys(self):
        overlap = set(_FRAMEWORK_PACKAGES) & set(_PROVIDER_PACKAGES)
        assert not overlap, f"Names overlap between framework and provider tables: {overlap}"
