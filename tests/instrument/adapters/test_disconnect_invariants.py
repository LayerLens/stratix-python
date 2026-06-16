"""Disconnect "leave-no-trace" invariants (LAY-3577 / T3).

Every adapter must, on ``disconnect()``:

* restore every patched attribute to the exact original callable,
* deregister only its OWN callbacks/handlers — third-party state registered
  before (or after) connect survives,
* tolerate a second ``disconnect()``,
* support a full connect → disconnect → connect → disconnect cycle.

The openai_agents variant of these invariants (global processor registry,
N6/N2) lives in ``tests/instrument/adapters/frameworks/test_openai_agents.py``.
Framework adapters that need a real framework object to connect (crewai,
smolagents, ...) carry their invariant tests in their own per-framework
modules, which skip when the framework isn't installed.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Callable, Optional
from dataclasses import field, dataclass

import pytest


def _stub(name: str) -> Callable[..., Dict[str, str]]:
    def fn(*args: Any, **kwargs: Any) -> Dict[str, str]:
        return {"stub": name}

    fn.__name__ = name
    return fn


def _get_path(obj: Any, path: str) -> Any:
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


# ---------------------------------------------------------------------------
# Case table
# ---------------------------------------------------------------------------


@dataclass
class Case:
    id: str
    make_adapter: Callable[[Any], Any]  # mock_client -> adapter instance
    make_target: Callable[[], Any]
    patched_paths: List[str]
    requires: Optional[str] = None  # pytest.importorskip module
    connect_kwargs: Dict[str, Any] = field(default_factory=dict)


def _openai_provider(_client: Any) -> Any:
    from layerlens.instrument.adapters.providers.openai import OpenAIProvider

    return OpenAIProvider()


def _anthropic_provider(_client: Any) -> Any:
    from layerlens.instrument.adapters.providers.anthropic import AnthropicProvider

    return AnthropicProvider()


def _azure_provider(_client: Any) -> Any:
    from layerlens.instrument.adapters.providers.azure_openai import (
        AzureOpenAIProvider,
    )

    return AzureOpenAIProvider()


def _vertex_provider(_client: Any) -> Any:
    from layerlens.instrument.adapters.providers.google_vertex import (
        GoogleVertexProvider,
    )

    return GoogleVertexProvider()


def _bedrock_provider(_client: Any) -> Any:
    from layerlens.instrument.adapters.providers.bedrock import BedrockProvider

    return BedrockProvider()


def _ollama_provider(_client: Any) -> Any:
    from layerlens.instrument.adapters.providers.ollama import OllamaProvider

    return OllamaProvider()


def _mcp_adapter(_client: Any) -> Any:
    from layerlens.instrument.adapters.protocols.mcp.adapter import MCPProtocolAdapter

    return MCPProtocolAdapter()


def _agui_adapter(_client: Any) -> Any:
    from layerlens.instrument.adapters.protocols.agui.adapter import (
        AGUIProtocolAdapter,
    )

    return AGUIProtocolAdapter()


def _ap2_adapter(_client: Any) -> Any:
    from layerlens.instrument.adapters.protocols.ap2 import AP2ProtocolAdapter

    return AP2ProtocolAdapter()


def _ucp_adapter(_client: Any) -> Any:
    from layerlens.instrument.adapters.protocols.ucp import UCPProtocolAdapter

    return UCPProtocolAdapter()


def _a2ui_adapter(_client: Any) -> Any:
    from layerlens.instrument.adapters.protocols.a2ui import A2UIProtocolAdapter

    return A2UIProtocolAdapter()


def _a2a_adapter(_client: Any) -> Any:
    from layerlens.instrument.adapters.protocols.a2a.adapter import (
        A2AProtocolAdapter,
    )

    return A2AProtocolAdapter()


def _embedding_adapter(client: Any) -> Any:
    from layerlens.instrument.adapters.frameworks.embedding import EmbeddingAdapter

    return EmbeddingAdapter(client)


def _vector_store_adapter(client: Any) -> Any:
    from layerlens.instrument.adapters.frameworks.vector_store import (
        VectorStoreAdapter,
    )

    return VectorStoreAdapter(client)


def _langchain_adapter(client: Any) -> Any:
    from layerlens.instrument.adapters.frameworks.langchain import (
        LangChainCallbackHandler,
    )

    return LangChainCallbackHandler(client)


def _langgraph_adapter(client: Any) -> Any:
    from layerlens.instrument.adapters.frameworks.langgraph import (
        LangGraphCallbackHandler,
    )

    return LangGraphCallbackHandler(client)


CASES: List[Case] = [
    # -- providers (duck-typed fake clients, no SDK objects needed) --
    Case(
        id="provider-openai",
        make_adapter=_openai_provider,
        make_target=lambda: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=_stub("chat"))),
            responses=SimpleNamespace(create=_stub("responses")),
            embeddings=SimpleNamespace(create=_stub("embeddings")),
        ),
        patched_paths=["chat.completions.create", "responses.create", "embeddings.create"],
    ),
    Case(
        id="provider-anthropic",
        make_adapter=_anthropic_provider,
        make_target=lambda: SimpleNamespace(
            messages=SimpleNamespace(create=_stub("create"), stream=_stub("stream")),
        ),
        patched_paths=["messages.create", "messages.stream"],
    ),
    Case(
        id="provider-azure-openai",
        make_adapter=_azure_provider,
        make_target=lambda: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=_stub("chat"))),
            base_url="https://example-resource.openai.azure.com/openai/v1/",
        ),
        patched_paths=["chat.completions.create"],
    ),
    Case(
        id="provider-google-vertex",
        make_adapter=_vertex_provider,
        make_target=lambda: SimpleNamespace(
            generate_content=_stub("generate_content"),
            model_name="gemini-1.5-flash",
        ),
        patched_paths=["generate_content"],
    ),
    Case(
        id="provider-bedrock",
        make_adapter=_bedrock_provider,
        make_target=lambda: SimpleNamespace(
            invoke_model=_stub("invoke_model"),
            converse=_stub("converse"),
        ),
        patched_paths=["invoke_model", "converse"],
    ),
    Case(
        id="provider-ollama",
        make_adapter=_ollama_provider,
        make_target=lambda: SimpleNamespace(
            chat=_stub("chat"),
            generate=_stub("generate"),
            embeddings=_stub("embeddings"),
            embed=_stub("embed"),
        ),
        patched_paths=["chat", "generate", "embeddings", "embed"],
    ),
    # -- protocols (duck-typed fake SDK clients) --
    Case(
        id="protocol-mcp",
        make_adapter=_mcp_adapter,
        make_target=lambda: SimpleNamespace(
            call_tool=_stub("call_tool"),
            list_tools=_stub("list_tools"),
            elicit=_stub("elicit"),
        ),
        patched_paths=["call_tool", "list_tools", "elicit"],
    ),
    Case(
        id="protocol-agui",
        make_adapter=_agui_adapter,
        make_target=lambda: SimpleNamespace(
            dispatch_event=_stub("dispatch_event"),
            emit_event=_stub("emit_event"),
            publish=_stub("publish"),
        ),
        patched_paths=["dispatch_event", "emit_event", "publish"],
    ),
    Case(
        id="protocol-ap2",
        make_adapter=_ap2_adapter,
        make_target=lambda: SimpleNamespace(
            create_intent_mandate=_stub("create_intent_mandate"),
            sign_payment_mandate=_stub("sign_payment_mandate"),
            issue_receipt=_stub("issue_receipt"),
        ),
        patched_paths=["create_intent_mandate", "sign_payment_mandate", "issue_receipt"],
    ),
    Case(
        id="protocol-ucp",
        make_adapter=_ucp_adapter,
        make_target=lambda: SimpleNamespace(
            discover_suppliers=_stub("discover_suppliers"),
            browse_catalog=_stub("browse_catalog"),
            start_checkout=_stub("start_checkout"),
            complete_checkout=_stub("complete_checkout"),
            issue_refund=_stub("issue_refund"),
        ),
        patched_paths=[
            "discover_suppliers",
            "browse_catalog",
            "start_checkout",
            "complete_checkout",
            "issue_refund",
        ],
    ),
    Case(
        id="protocol-a2ui",
        make_adapter=_a2ui_adapter,
        make_target=lambda: SimpleNamespace(
            on_surface_created=_stub("on_surface_created"),
            on_user_action=_stub("on_user_action"),
        ),
        patched_paths=["on_surface_created", "on_user_action"],
    ),
    Case(
        id="protocol-a2a",
        make_adapter=_a2a_adapter,
        make_target=lambda: SimpleNamespace(
            send_task=_stub("send_task"),
            get_task=_stub("get_task"),
            cancel_task=_stub("cancel_task"),
            get_agent_card=_stub("get_agent_card"),
            register_handler=_stub("register_handler"),
        ),
        patched_paths=[
            "send_task",
            "get_task",
            "cancel_task",
            "get_agent_card",
            "register_handler",
        ],
    ),
    # -- target-wrapping framework adapters --
    Case(
        id="framework-embedding",
        make_adapter=_embedding_adapter,
        make_target=lambda: SimpleNamespace(
            embeddings=SimpleNamespace(create=_stub("create")),
        ),
        patched_paths=["embeddings.create"],
    ),
    Case(
        id="framework-vector-store",
        make_adapter=_vector_store_adapter,
        make_target=lambda: SimpleNamespace(query=_stub("query")),
        patched_paths=["query"],
    ),
    # -- callback-handler frameworks (no patching; lifecycle invariants only) --
    Case(
        id="framework-langchain",
        make_adapter=_langchain_adapter,
        make_target=lambda: None,
        patched_paths=[],
        requires="langchain_core",
    ),
    Case(
        id="framework-langgraph",
        make_adapter=_langgraph_adapter,
        make_target=lambda: None,
        patched_paths=[],
        requires="langchain_core",
    ),
]

_CASE_IDS = [c.id for c in CASES]


@pytest.fixture(params=CASES, ids=_CASE_IDS)
def case(request: Any) -> Case:
    c: Case = request.param
    if c.requires is not None:
        pytest.importorskip(c.requires)
    return c


# ---------------------------------------------------------------------------
# Generic invariants
# ---------------------------------------------------------------------------


class TestDisconnectInvariants:
    def test_disconnect_restores_originals(self, case: Case, mock_client: Any) -> None:
        target = case.make_target()
        originals = {p: _get_path(target, p) for p in case.patched_paths}

        adapter = case.make_adapter(mock_client)
        adapter.connect(target=target, **case.connect_kwargs)

        for path, original in originals.items():
            assert _get_path(target, path) is not original, f"{path} was never patched by connect()"

        adapter.disconnect()

        for path, original in originals.items():
            assert _get_path(target, path) is original, f"{path} not restored to the original after disconnect()"

    def test_double_disconnect_is_safe(self, case: Case, mock_client: Any) -> None:
        target = case.make_target()
        originals = {p: _get_path(target, p) for p in case.patched_paths}

        adapter = case.make_adapter(mock_client)
        adapter.connect(target=target, **case.connect_kwargs)
        adapter.disconnect()
        adapter.disconnect()  # must not raise or re-patch

        for path, original in originals.items():
            assert _get_path(target, path) is original

    def test_reconnect_cycle_works(self, case: Case, mock_client: Any) -> None:
        target = case.make_target()
        originals = {p: _get_path(target, p) for p in case.patched_paths}

        adapter = case.make_adapter(mock_client)
        adapter.connect(target=target, **case.connect_kwargs)
        adapter.disconnect()
        adapter.connect(target=target, **case.connect_kwargs)

        for path, original in originals.items():
            assert _get_path(target, path) is not original, f"{path} not re-patched on reconnect"

        adapter.disconnect()

        for path, original in originals.items():
            assert _get_path(target, path) is original, f"{path} not restored after the second disconnect"


# ---------------------------------------------------------------------------
# litellm — module-level patch must restore the real module's functions
# ---------------------------------------------------------------------------


class TestLiteLLMModuleRestore:
    def test_disconnect_restores_module_functions(self, mock_client: Any) -> None:
        litellm = pytest.importorskip("litellm")
        from layerlens.instrument.adapters.providers.litellm import LiteLLMProvider

        orig_completion = litellm.completion
        orig_acompletion = litellm.acompletion
        adapter = LiteLLMProvider()
        try:
            adapter.connect()
            assert litellm.completion is not orig_completion
            adapter.disconnect()
            assert litellm.completion is orig_completion
            assert litellm.acompletion is orig_acompletion
        finally:
            # Never leak a patched module into other tests, even on failure.
            litellm.completion = orig_completion
            litellm.acompletion = orig_acompletion


# ---------------------------------------------------------------------------
# pydantic_ai — agent attributes restored exactly
# ---------------------------------------------------------------------------


class TestPydanticAIRestore:
    def test_disconnect_restores_agent_surface(self, mock_client: Any) -> None:
        pytest.importorskip("pydantic_ai")
        from pydantic_ai import Agent
        from pydantic_ai.models.test import TestModel

        from layerlens.instrument.adapters.frameworks.pydantic_ai import (
            PydanticAIAdapter,
        )

        model = TestModel()
        agent = Agent(model=model, name="invariant_agent")
        orig_run = agent.run

        adapter = PydanticAIAdapter(mock_client)
        adapter.connect(target=agent)
        assert "run" in agent.__dict__ or agent.model is not model

        adapter.disconnect()
        assert agent.model is model
        for attr in ("run", "run_sync", "run_stream"):
            assert attr not in agent.__dict__, f"{attr} still shadowed on the agent after disconnect"
        assert agent.run == orig_run or agent.run.__func__ is orig_run.__func__

        adapter.disconnect()  # double disconnect safe
        assert agent.model is model


# ---------------------------------------------------------------------------
# bedrock_agents — boto3 event handlers: ours removed, third-party kept
# ---------------------------------------------------------------------------


class _FakeBotoEvents:
    def __init__(self) -> None:
        self.handlers: Dict[str, List[Any]] = {}

    def register(self, name: str, fn: Any) -> None:
        self.handlers.setdefault(name, []).append(fn)

    def unregister(self, name: str, fn: Any) -> None:
        self.handlers[name].remove(fn)

    def all_handlers(self) -> List[Any]:
        return [fn for fns in self.handlers.values() for fn in fns]


class TestBedrockAgentsEventSystem:
    def test_disconnect_removes_only_own_handlers(self, mock_client: Any) -> None:
        pytest.importorskip("boto3")
        from layerlens.instrument.adapters.frameworks.bedrock_agents import (
            BedrockAgentsAdapter,
        )

        events = _FakeBotoEvents()
        third_party = _stub("user_handler")
        events.register("before-call.bedrock-agent-runtime.InvokeAgent", third_party)
        target = SimpleNamespace(meta=SimpleNamespace(events=events))

        adapter = BedrockAgentsAdapter(mock_client)
        adapter.connect(target=target)
        assert len(events.all_handlers()) > 1, "connect() registered no handlers"

        adapter.disconnect()
        remaining = events.all_handlers()
        assert third_party in remaining, "third-party boto3 handler removed by disconnect()"
        assert remaining == [third_party], f"adapter handlers left behind: {remaining}"

        adapter.disconnect()  # double disconnect safe
        assert events.all_handlers() == [third_party]
