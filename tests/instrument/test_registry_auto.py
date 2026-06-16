"""Tests for AdapterRegistry auto-detection (``discover_installed`` + ``auto``)."""

from __future__ import annotations

import sys
import importlib
import importlib.util
from typing import Any
from unittest.mock import Mock, patch

import pytest

from layerlens.instrument import CaptureConfig, auto, discover_installed
from layerlens.instrument.adapters._base import AdapterInfo, BaseAdapter
from layerlens.instrument.adapters._registry import (
    _PROVIDER_PACKAGES,
    _FRAMEWORK_ADAPTERS,
    _FRAMEWORK_PACKAGES,
    get,
    register,
    _adapters,
    list_adapters,
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

        with patch(
            "layerlens.instrument.adapters._registry._is_installed",
            side_effect=fake_is_installed,
        ):
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

    def test_detects_autogen_agentchat_package_layout(self):
        """The ``autogen`` extra installs autogen-agentchat, which ships the
        ``autogen_core``/``autogen_agentchat`` modules and NO top-level
        ``autogen`` module — detection must still find it (LAY-3567 B5)."""
        installed = {"autogen_core", "autogen_agentchat"}

        with patch(
            "layerlens.instrument.adapters._registry._is_installed",
            side_effect=lambda pkg: pkg in installed,
        ):
            result = discover_installed()

        assert "autogen" in result["frameworks"]

    def test_reports_packages_installed_in_this_env(self):
        """Live (unmocked) probe against the base venv this suite runs in.

        Installed here: langchain-core, langgraph, openai-agents (``agents``),
        pydantic-ai, plus the boto3/openai/anthropic/litellm/ollama SDKs.
        """
        result = discover_installed()

        for name in ("langchain", "openai_agents", "pydantic_ai"):
            assert name in result["frameworks"], name
        for name in ("openai", "anthropic", "litellm", "ollama", "bedrock"):
            assert name in result["providers"], name


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

        with patch(
            "layerlens.instrument.adapters._registry._is_installed",
            side_effect=fake_is_installed,
        ), patch(
            "layerlens.instrument.adapters._registry.importlib.import_module",
            return_value=fake_module,
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
            "layerlens.instrument.adapters._registry.importlib.import_module",
            return_value=fake_module,
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

        with patch(
            "layerlens.instrument.adapters._registry._is_installed",
            side_effect=fake_is_installed,
        ), patch(
            "layerlens.instrument.adapters._registry.importlib.import_module",
            return_value=fake_module,
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

        with patch(
            "layerlens.instrument.adapters._registry._is_installed",
            side_effect=fake_is_installed,
        ), patch(
            "layerlens.instrument.adapters._registry.importlib.import_module",
            return_value=fake_module,
        ):
            auto(client, capture_config=fake_config)

        fake_adapter_cls.assert_called_once_with(client, capture_config=fake_config)


# Adapters auto() can ALWAYS wire in this repo's CI/base envs (langchain-core,
# langgraph, and the openai-agents SDK are in the dev lock). On py3.10+ rows CI
# also resolves the interpreter-gated extras (crewai, autogen, ...), so the
# wired set is env-dependent — assertions below use membership, never exact
# equality. pydantic_ai and bedrock_agents are *detected* but their connect()
# requires a target, so auto() warns and drops them — pinned in
# test_auto_attempts_but_cannot_wire_target_requiring_adapters below.
_ALWAYS_AUTO_WIRED = {"langchain", "langgraph", "openai_agents"}


class TestAutoLiveWiring:
    """Unmocked ``auto()`` runs against the real adapters in this venv.

    These tests connect real adapters with global side effects (the OpenAI
    Agents adapter registers itself as a global trace processor). The
    autouse ``_clear_registry`` fixture calls ``disconnect_all()`` after
    every test, which unhooks all of that before the next test runs.
    """

    def test_wires_expected_adapters_for_this_env(self):
        connected = auto(Mock())

        assert _ALWAYS_AUTO_WIRED <= set(connected)
        for name, adapter in connected.items():
            assert adapter.is_connected, name
            assert get(name) is adapter, name

    def test_capture_config_threaded_to_every_wired_adapter(self):
        cfg = CaptureConfig.standard()

        connected = auto(Mock(), capture_config=cfg)

        assert _ALWAYS_AUTO_WIRED <= set(connected)
        for name, adapter in connected.items():
            assert adapter._config is cfg, f"{name} did not receive the shared capture_config"

    def test_skip_excludes_named_adapter_but_wires_the_rest(self):
        connected = auto(Mock(), skip=["langchain"])

        assert "langchain" not in connected
        assert get("langchain") is None
        assert _ALWAYS_AUTO_WIRED - {"langchain"} <= set(connected)

    def test_skip_unknown_name_is_harmless(self):
        connected = auto(Mock(), skip=["definitely_not_an_adapter"])

        assert _ALWAYS_AUTO_WIRED <= set(connected)

    def test_auto_attempts_but_cannot_wire_target_requiring_adapters(self, caplog):
        """pydantic_ai and bedrock_agents are discovered here (pydantic-ai and
        boto3 are installed) and listed in ``_FRAMEWORK_ADAPTERS``, but their
        ``connect()`` raises ``ValueError`` when called without a target, so
        EVERY ``auto()`` call in an env like this one logs a warning for them
        and omits them from the result.

        Pins CURRENT behavior: these two can never be auto-wired, yet they
        are attempted anyway instead of being excluded from auto() the way
        agentforce/langfuse are.
        """
        connected = auto(Mock())

        assert "pydantic_ai" not in connected
        assert "bedrock_agents" not in connected
        assert get("pydantic_ai") is None
        assert get("bedrock_agents") is None
        for name in ("pydantic_ai", "bedrock_agents"):
            assert any(name in rec.message for rec in caplog.records), name

    def test_connect_failure_in_one_adapter_does_not_block_others(self, monkeypatch, caplog):
        # LangGraphCallbackHandler subclasses LangChainCallbackHandler, so
        # break openai_agents instead of langchain to keep the blast radius
        # to a single adapter.
        from layerlens.instrument.adapters.frameworks.openai_agents import OpenAIAgentsAdapter

        def _boom(self, target: Any = None, **kwargs: Any) -> Any:
            raise RuntimeError("synthetic connect failure")

        monkeypatch.setattr(OpenAIAgentsAdapter, "connect", _boom)

        connected = auto(Mock())

        assert "openai_agents" not in connected
        assert get("openai_agents") is None
        assert {"langchain", "langgraph"} <= set(connected)
        assert any("openai_agents" in rec.message for rec in caplog.records)


class TestProbeFalsePositives:
    """Detection probes match *module names*, not the actual SDKs.

    Both tests pin CURRENT behavior. Probe tightening was deliberately
    deferred — per LAY-3585 it may ride along only if the team agrees in PR
    review. Do not "fix" these without that sign-off.
    """

    def test_generic_agents_probe_false_positive_current_behavior(self, tmp_path, monkeypatch):
        """``openai_agents`` is probed via the generic top-level module name
        ``agents`` (``_FRAMEWORK_PACKAGES["openai_agents"] == "agents"``), so
        ANY unrelated package that happens to be called ``agents`` makes
        ``discover_installed()`` report the OpenAI Agents SDK as installed.

        TEAM DECISION PENDING (LAY-3585): a tighter probe (e.g.
        ``agents.tracing``, which is what the adapter actually imports) was
        deliberately deferred to PR review. Until then this test pins the
        false positive.
        """
        impostor_pkg = tmp_path / "agents"
        impostor_pkg.mkdir()
        (impostor_pkg / "__init__.py").write_text("# definitely not the openai-agents SDK\n")

        # Shadow the real SDK so the path-based probe resolves the impostor.
        monkeypatch.delitem(sys.modules, "agents", raising=False)
        monkeypatch.syspath_prepend(str(tmp_path))
        importlib.invalidate_caches()

        # Prove the probe's positive signal comes from the impostor, not the
        # real openai-agents SDK that is also installed in this venv.
        spec = importlib.util.find_spec("agents")
        assert spec is not None
        assert spec.origin == str(impostor_pkg / "__init__.py")

        assert "openai_agents" in discover_installed()["frameworks"]

    def test_bedrock_agents_probe_matches_bare_boto3_current_behavior(self):
        """``bedrock_agents`` is probed via ``boto3``
        (``_FRAMEWORK_PACKAGES["bedrock_agents"] == "boto3"``), so every
        environment with boto3 installed — effectively every AWS user — is
        reported as having Bedrock Agents, and ``auto()`` instantiates and
        attempts the adapter there (the connect then fails for lack of a
        bedrock-agent-runtime client; see
        test_auto_attempts_but_cannot_wire_target_requiring_adapters).

        Same deferred probe-tightening caveat as the ``agents`` test above.
        """
        assert importlib.util.find_spec("boto3") is not None, "precondition: boto3 in this venv"

        assert "bedrock_agents" in discover_installed()["frameworks"]


class _ExplodingDisconnectAdapter(BaseAdapter):
    """Stub whose ``disconnect()`` always raises."""

    def connect(self, target: Any = None, **kwargs: Any) -> Any:
        return target

    def disconnect(self) -> None:
        raise RuntimeError("disconnect failure")

    def adapter_info(self) -> AdapterInfo:
        return AdapterInfo(name="exploding", adapter_type="framework", connected=True)


class TestDisconnectAll:
    """register/get/unregister/list_adapters basics are covered in
    tests/instrument/adapters/test_registry.py; only the error-swallowing
    path of ``disconnect_all()`` is pinned here."""

    def test_swallows_disconnect_errors_and_still_clears(self, caplog):
        bad = _ExplodingDisconnectAdapter()
        good = Mock(spec=BaseAdapter)
        register("exploding", bad)
        register("ok", good)

        disconnect_all()  # must not raise

        assert list_adapters() == []
        assert get("exploding") is None
        assert get("ok") is None
        good.disconnect.assert_called_once_with()
        assert any("Error disconnecting" in rec.message for rec in caplog.records)


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
