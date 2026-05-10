"""Typed-event regression tests for the langchain adapter.

Bundle #6 of the typed-events migration ports the langchain
:class:`LayerLensCallbackHandler` from the legacy
:meth:`emit_dict_event` dispatcher to the typed
:meth:`BaseAdapter.emit_event` path against the canonical Pydantic
models from :mod:`layerlens.instrument._compat.events`.

The full langchain test suite is untracked on PR #129's foundation
branch (``tests/instrument/adapters/frameworks/test_langchain_adapter.py``
does not exist on this branch). This regression module mirrors the
shape of PR #138 / #151 / #152's per-adapter typed-event tests and
asserts:

1. Every emit site produces a canonical Pydantic instance
   (:class:`ModelInvokeEvent` / :class:`ToolCallEvent` /
   :class:`AgentInputEvent` / :class:`AgentOutputEvent`).
2. ``filterwarnings('error', DeprecationWarning)`` catches any
   residual ``emit_dict_event`` call — a re-introduction would
   immediately fail this test.
"""

from __future__ import annotations

import uuid
import warnings
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from layerlens._compat.pydantic import (
    BaseModel as _CompatBaseModel,
    model_dump as _compat_model_dump,
)
from layerlens.instrument._compat.events import (
    ToolCallEvent,
    AgentInputEvent,
    AgentOutputEvent,
    ModelInvokeEvent,
)
from layerlens.instrument.adapters._base.capture import CaptureConfig
from layerlens.instrument.adapters.frameworks.langchain.callbacks import (
    LayerLensCallbackHandler,
)


class _RecordingStratix:
    org_id: str = "test-org"

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self.typed_payloads: List[Any] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:
        if len(args) == 2 and isinstance(args[0], str):
            self.events.append({"event_type": args[0], "payload": args[1]})
            return
        if args and isinstance(args[0], _CompatBaseModel):
            payload_model = args[0]
            self.typed_payloads.append(payload_model)
            event_type = getattr(payload_model, "event_type", "<unknown>")
            self.events.append(
                {"event_type": event_type, "payload": _compat_model_dump(payload_model)}
            )


@pytest.fixture
def handler() -> LayerLensCallbackHandler:
    h = LayerLensCallbackHandler(
        stratix=_RecordingStratix(),
        capture_config=CaptureConfig.full(),
        org_id="test-org",
    )
    h.connect()
    return h


def _llm_response(content: str = "hi", prompt_tokens: int = 10, completion_tokens: int = 5) -> Any:
    """Build a minimal ``LLMResult``-shaped duck type."""
    gen = SimpleNamespace(text=content)
    return SimpleNamespace(
        generations=[[gen]],
        llm_output={"token_usage": {"prompt_tokens": prompt_tokens,
                                     "completion_tokens": completion_tokens,
                                     "total_tokens": prompt_tokens + completion_tokens}},
    )


class TestLangchainTypedEvents:
    def test_on_llm_end_emits_typed_model_invoke(
        self, handler: LayerLensCallbackHandler
    ) -> None:
        run_id = uuid.uuid4()
        # ``serialized`` per LangChain's protocol: ``id`` is a fully-
        # qualified module path list whose 3rd element is the
        # provider name; ``model_name`` lives in ``kwargs``.
        handler.on_llm_start(
            serialized={"model_name": "gpt-4o", "kwargs": {}, "id": ["langchain", "llms", "openai"]},
            prompts=["hi"],
            run_id=run_id,
        )
        handler.on_llm_end(response=_llm_response(), run_id=run_id)

        stratix = handler._stratix
        invoke_payloads = [p for p in stratix.typed_payloads if isinstance(p, ModelInvokeEvent)]
        assert len(invoke_payloads) == 1
        invoke = stratix.events[-1]
        assert invoke["event_type"] == "model.invoke"
        # Canonical shape — model nested under ``model.{provider,
        # name, version, parameters}``.
        assert invoke["payload"]["model"]["name"] == "gpt-4o"
        assert invoke["payload"]["model"]["provider"] == "openai"
        # Token slots are top-level on the envelope.
        assert invoke["payload"]["prompt_tokens"] == 10
        assert invoke["payload"]["completion_tokens"] == 5
        assert invoke["payload"]["total_tokens"] == 15

    def test_on_tool_end_emits_typed_tool_call(
        self, handler: LayerLensCallbackHandler
    ) -> None:
        run_id = uuid.uuid4()
        handler.on_tool_start(
            serialized={"name": "calculator"},
            input_str="2+2",
            run_id=run_id,
            inputs={"expression": "2+2"},
        )
        handler.on_tool_end(output="4", run_id=run_id)

        stratix = handler._stratix
        tool_payloads = [p for p in stratix.typed_payloads if isinstance(p, ToolCallEvent)]
        assert len(tool_payloads) == 1
        ev = stratix.events[-1]
        assert ev["event_type"] == "tool.call"
        assert ev["payload"]["tool"]["name"] == "calculator"
        # Inputs preserved on canonical ``input`` slot.
        assert ev["payload"]["input"]["expression"] == "2+2"
        # Output wrapped on canonical ``output`` slot.
        assert ev["payload"]["output"] == {"value": "4"}

    def test_on_chain_start_emits_typed_agent_input(
        self, handler: LayerLensCallbackHandler
    ) -> None:
        run_id = uuid.uuid4()
        handler.on_chain_start(
            serialized={"name": "graph"},
            inputs={"messages": ["hi"]},
            run_id=run_id,
            metadata={"langgraph_node": "agent_a", "langgraph_step": 0},
        )

        stratix = handler._stratix
        input_payloads = [p for p in stratix.typed_payloads if isinstance(p, AgentInputEvent)]
        assert len(input_payloads) == 1
        ev = stratix.events[-1]
        assert ev["event_type"] == "agent.input"
        # node_name on canonical metadata, not as a top-level field.
        assert ev["payload"]["content"]["metadata"]["node_name"] == "agent_a"

    def test_on_agent_finish_emits_typed_agent_output(
        self, handler: LayerLensCallbackHandler
    ) -> None:
        run_id = uuid.uuid4()
        finish = SimpleNamespace(return_values={"output": "done"}, log="thinking…")
        handler.on_agent_finish(finish, run_id=run_id)

        stratix = handler._stratix
        output_payloads = [p for p in stratix.typed_payloads if isinstance(p, AgentOutputEvent)]
        assert len(output_payloads) == 1
        ev = stratix.events[-1]
        assert ev["event_type"] == "agent.output"
        assert ev["payload"]["content"]["metadata"]["log"] == "thinking…"

    def test_no_deprecation_warning_after_migration(
        self, handler: LayerLensCallbackHandler
    ) -> None:
        """Walking every emission path must produce zero DeprecationWarnings."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            run_id = uuid.uuid4()
            handler.on_llm_start(
                serialized={"model_name": "gpt-4o", "kwargs": {}, "id": ["langchain", "llms", "openai"]},
                prompts=["hi"],
                run_id=run_id,
            )
            handler.on_llm_end(response=_llm_response(), run_id=run_id)

            tool_run_id = uuid.uuid4()
            handler.on_tool_start(serialized={"name": "calc"}, input_str="x", run_id=tool_run_id, inputs={"x": 1})
            handler.on_tool_end(output="2", run_id=tool_run_id)

            chain_run_id = uuid.uuid4()
            handler.on_chain_start(
                serialized={"name": "g"}, inputs={"x": 1}, run_id=chain_run_id,
                metadata={"langgraph_node": "n"},
            )
            handler.on_chain_end(outputs={"y": 2}, run_id=chain_run_id)

            handler.on_agent_finish(SimpleNamespace(return_values={"r": 1}), run_id=uuid.uuid4())
