"""Regression tests for the typed-event migration of ``providers/_base/provider.py``.

Bundle #6 of the typed-events migration ports the four shared
``LLMProviderAdapter._emit_*`` helpers from
:meth:`emit_dict_event` to typed
:meth:`BaseAdapter.emit_event` calls. Every concrete LLM provider
adapter (openai, anthropic, azure_openai, aws_bedrock, google_vertex,
cohere, mistral, ollama, litellm) inherits these helpers — so this
file exercises the helpers directly via a minimal subclass and
asserts:

1. Each helper produces a canonical Pydantic instance.
2. No :class:`DeprecationWarning` fires after migration (proves zero
   ``emit_dict_event`` calls remain on the path).
3. The dict-shape view of the typed payload preserves the canonical
   schema slots (``payload["model"]["name"]``,
   ``payload["tool"]["name"]``, ``payload["cost"]["api_cost_usd"]``,
   ``payload["violation"]["root_cause"]``).

The full provider-adapter test suites
(``tests/instrument/adapters/providers/test_<provider>_adapter.py``)
have pre-existing collection errors on PR #129's foundation branch
because they import from untracked submodules
(``providers/_base/tokens.py``, ``providers/_base/pricing.py``).
``providers/_base/provider.py`` itself imports those untracked
modules at line 70 / 71 — so this regression test is gated by the
same import. We skip the entire module via :func:`pytest.importorskip`
when the foundation branch is missing the sibling modules; the test
runs cleanly once the submodules land (which they will when this
branch is rebased onto a target that carries them).
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List
from dataclasses import dataclass

import pytest

from layerlens._compat.pydantic import (
    BaseModel as _CompatBaseModel,
    model_dump as _compat_model_dump,
)
from layerlens.instrument._compat.events import (
    ToolCallEvent,
    ViolationType,
    CostRecordEvent,
    IntegrationType,
    ModelInvokeEvent,
    PolicyViolationEvent,
)
from layerlens.instrument.adapters._base.capture import CaptureConfig

# providers/_base/{tokens,pricing}.py are untracked on PR #129's
# foundation branch — provider.py imports them at module load. Use
# importorskip to defer collection to environments where the
# submodules are present (any branch downstream of merge to master
# carries them; this regression confirms the typed-event migration
# does not regress that path).
LLMProviderAdapter = pytest.importorskip(
    "layerlens.instrument.adapters.providers._base.provider"
).LLMProviderAdapter


@dataclass
class _MinimalNormalizedTokenUsage:
    """Duck-typed stand-in for :class:`NormalizedTokenUsage`.

    The real ``providers/_base/tokens.py`` is untracked on the
    foundation branch — this dataclass mimics the attribute surface
    the four ``_emit_*`` helpers read (``prompt_tokens``,
    ``completion_tokens``, ``total_tokens``, ``cached_tokens``,
    ``reasoning_tokens``).
    """

    prompt_tokens: int = 10
    completion_tokens: int = 5
    total_tokens: int = 15
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None


class _StubProvider(LLMProviderAdapter):
    """Minimal concrete subclass — connects without a real client."""

    FRAMEWORK = "_stub_provider"
    VERSION = "0.0.0"

    def connect_client(self, client: Any) -> Any:
        return client


class _RecordingStratix:
    """Records both legacy dict and typed Pydantic emissions.

    Mirrors the ``_RecordingStratix`` doubles in PR #138 / #151 / #152
    so the assertions below can read the typed payload as a dict
    (``model_dump``) and as the original Pydantic instance
    (``typed_payloads``).
    """

    org_id: str = "test-org"

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self.typed_payloads: List[Any] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:
        # Dict-path (legacy emit_dict_event): emit(event_type, payload).
        if len(args) == 2 and isinstance(args[0], str):
            self.events.append({"event_type": args[0], "payload": args[1]})
            return
        # Typed-path (emit_event): emit(payload_model[, privacy_level]).
        if args and isinstance(args[0], _CompatBaseModel):
            payload_model = args[0]
            self.typed_payloads.append(payload_model)
            event_type = getattr(payload_model, "event_type", "<unknown>")
            self.events.append(
                {
                    "event_type": event_type,
                    "payload": _compat_model_dump(payload_model),
                }
            )


@pytest.fixture
def adapter() -> _StubProvider:
    """Build a connected stub provider with full content capture."""
    a = _StubProvider(
        stratix=_RecordingStratix(),
        capture_config=CaptureConfig.full(),
        org_id="test-org",
    )
    a.connect()
    return a


# ---------------------------------------------------------------------------
# _emit_model_invoke
# ---------------------------------------------------------------------------


class TestEmitModelInvoke:
    def test_produces_typed_model_invoke_event(self, adapter: _StubProvider) -> None:
        adapter._emit_model_invoke(
            provider="openai",
            model="gpt-4o",
            parameters={"temperature": 0.7, "tools_count": 2},
            usage=_MinimalNormalizedTokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            latency_ms=42.5,
            input_messages=[{"role": "user", "content": "hi"}],
            output_message={"role": "assistant", "content": "hello"},
            metadata={"finish_reason": "stop", "response_id": "resp-1"},
        )
        stratix = adapter._stratix
        assert isinstance(stratix.typed_payloads[0], ModelInvokeEvent)

    def test_canonical_dict_shape(self, adapter: _StubProvider) -> None:
        adapter._emit_model_invoke(
            provider="anthropic",
            model="claude-sonnet-4",
            parameters={"max_tokens": 1024},
            usage=_MinimalNormalizedTokenUsage(),
            latency_ms=7.0,
            metadata={"finish_reason": "end_turn"},
        )
        stratix = adapter._stratix
        invoke = stratix.events[0]
        assert invoke["event_type"] == "model.invoke"
        # Canonical: model is nested under ``model.{provider, name, version, parameters}``.
        assert invoke["payload"]["model"]["provider"] == "anthropic"
        assert invoke["payload"]["model"]["name"] == "claude-sonnet-4"
        assert invoke["payload"]["model"]["version"] == "unavailable"
        # Adapter-specific provenance flows onto ``model.parameters``.
        assert invoke["payload"]["model"]["parameters"]["max_tokens"] == 1024
        assert invoke["payload"]["model"]["parameters"]["finish_reason"] == "end_turn"
        # Token slots are top-level on the canonical envelope.
        assert invoke["payload"]["prompt_tokens"] == 10
        assert invoke["payload"]["completion_tokens"] == 5
        assert invoke["payload"]["total_tokens"] == 15
        assert invoke["payload"]["latency_ms"] == 7.0

    def test_emits_no_deprecation_warning(self, adapter: _StubProvider) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            adapter._emit_model_invoke(
                provider="openai",
                model="gpt-4o",
                usage=_MinimalNormalizedTokenUsage(),
            )

    def test_cached_and_reasoning_tokens_fold_onto_parameters(
        self, adapter: _StubProvider
    ) -> None:
        adapter._emit_model_invoke(
            provider="openai",
            model="o3",
            usage=_MinimalNormalizedTokenUsage(
                prompt_tokens=100, completion_tokens=50, total_tokens=150,
                cached_tokens=20, reasoning_tokens=30,
            ),
        )
        stratix = adapter._stratix
        params = stratix.events[0]["payload"]["model"]["parameters"]
        assert params["cached_tokens"] == 20
        assert params["reasoning_tokens"] == 30


# ---------------------------------------------------------------------------
# _emit_cost_record
# ---------------------------------------------------------------------------


class TestEmitCostRecord:
    def test_produces_typed_cost_record_event(self, adapter: _StubProvider) -> None:
        adapter._emit_cost_record(
            model="gpt-4o",
            usage=_MinimalNormalizedTokenUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500),
            provider="openai",
        )
        stratix = adapter._stratix
        assert isinstance(stratix.typed_payloads[0], CostRecordEvent)

    def test_canonical_dict_shape(self, adapter: _StubProvider) -> None:
        adapter._emit_cost_record(
            model="gpt-4o",
            usage=_MinimalNormalizedTokenUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500),
            provider="openai",
        )
        stratix = adapter._stratix
        cost = stratix.events[0]
        assert cost["event_type"] == "cost.record"
        assert cost["payload"]["cost"]["prompt_tokens"] == 1000
        assert cost["payload"]["cost"]["completion_tokens"] == 500
        assert cost["payload"]["cost"]["tokens"] == 1500

    def test_unavailable_pricing_uses_canonical_string_sentinel(
        self, adapter: _StubProvider
    ) -> None:
        # No pricing table → calculate_cost returns None → canonical
        # ``api_cost_usd="unavailable"`` (the schema's union string member).
        adapter._emit_cost_record(
            model="unknown-model-xyz",
            usage=_MinimalNormalizedTokenUsage(),
            provider="custom",
        )
        stratix = adapter._stratix
        cost = stratix.events[0]
        assert cost["payload"]["cost"]["api_cost_usd"] == "unavailable"


# ---------------------------------------------------------------------------
# _emit_tool_calls
# ---------------------------------------------------------------------------


class TestEmitToolCalls:
    def test_produces_typed_tool_call_events(self, adapter: _StubProvider) -> None:
        adapter._emit_tool_calls(
            [
                {"name": "get_weather", "arguments": {"city": "SF"}, "id": "call-1"},
                {"name": "get_time", "arguments": {"tz": "UTC"}, "id": "call-2"},
            ],
            parent_model="gpt-4o",
        )
        stratix = adapter._stratix
        assert len(stratix.typed_payloads) == 2
        assert all(isinstance(p, ToolCallEvent) for p in stratix.typed_payloads)

    def test_canonical_dict_shape(self, adapter: _StubProvider) -> None:
        adapter._emit_tool_calls(
            [{"name": "get_weather", "arguments": {"city": "SF"}, "id": "call-1"}],
            parent_model="gpt-4o",
        )
        stratix = adapter._stratix
        ev = stratix.events[0]
        assert ev["event_type"] == "tool.call"
        # Canonical: tool nested under ``tool.{name, version, integration}``.
        assert ev["payload"]["tool"]["name"] == "get_weather"
        assert ev["payload"]["tool"]["version"] == "unavailable"
        assert ev["payload"]["tool"]["integration"] == IntegrationType.LIBRARY.value
        # Tool input on canonical ``input`` slot. Provenance keys are namespaced.
        assert ev["payload"]["input"]["city"] == "SF"
        assert ev["payload"]["input"]["_tool_call_id"] == "call-1"
        assert ev["payload"]["input"]["_parent_model"] == "gpt-4o"
        assert ev["payload"]["input"]["_provider"] == "_stub_provider"

    def test_non_dict_arguments_wrap_on_value_key(self, adapter: _StubProvider) -> None:
        adapter._emit_tool_calls(
            [{"name": "echo", "arguments": "raw-string-arg", "id": "call-1"}],
        )
        stratix = adapter._stratix
        assert stratix.events[0]["payload"]["input"]["value"] == "raw-string-arg"


# ---------------------------------------------------------------------------
# _emit_provider_error
# ---------------------------------------------------------------------------


class TestEmitProviderError:
    def test_produces_typed_policy_violation_event(self, adapter: _StubProvider) -> None:
        adapter._emit_provider_error(
            provider="anthropic",
            error="rate limited",
            model="claude-sonnet-4",
        )
        stratix = adapter._stratix
        assert isinstance(stratix.typed_payloads[0], PolicyViolationEvent)

    def test_canonical_dict_shape(self, adapter: _StubProvider) -> None:
        adapter._emit_provider_error(
            provider="openai",
            error="rate limited",
            model="gpt-4o",
            metadata={"retry_after": 60},
        )
        stratix = adapter._stratix
        ev = stratix.events[0]
        assert ev["event_type"] == "policy.violation"
        # Canonical: violation nested under ``violation.{type, root_cause, remediation, details}``.
        assert ev["payload"]["violation"]["type"] == ViolationType.SAFETY.value
        assert ev["payload"]["violation"]["root_cause"] == "rate limited"
        assert "retry" in ev["payload"]["violation"]["remediation"].lower()
        # Adapter provenance lives on ``details``.
        assert ev["payload"]["violation"]["details"]["provider"] == "openai"
        assert ev["payload"]["violation"]["details"]["model"] == "gpt-4o"
        assert ev["payload"]["violation"]["details"]["retry_after"] == 60


# ---------------------------------------------------------------------------
# Cross-cutting: zero emit_dict_event on the post-migration path
# ---------------------------------------------------------------------------


class TestNoLegacyEmitPath:
    def test_full_emit_cycle_does_not_warn(self, adapter: _StubProvider) -> None:
        """A provider's hottest emission cycle (model invoke + cost record +
        tool calls + provider error) flows through zero
        ``emit_dict_event`` calls — proven by escalating
        ``DeprecationWarning`` to error.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            adapter._emit_model_invoke(
                provider="openai",
                model="gpt-4o",
                usage=_MinimalNormalizedTokenUsage(),
            )
            adapter._emit_cost_record(
                model="gpt-4o",
                usage=_MinimalNormalizedTokenUsage(),
                provider="openai",
            )
            adapter._emit_tool_calls(
                [{"name": "x", "arguments": {}, "id": "1"}], parent_model="gpt-4o",
            )
            adapter._emit_provider_error("openai", "boom", model="gpt-4o")

        stratix = adapter._stratix
        # Four typed payloads, four canonical event types.
        assert [type(p).__name__ for p in stratix.typed_payloads] == [
            "ModelInvokeEvent",
            "CostRecordEvent",
            "ToolCallEvent",
            "PolicyViolationEvent",
        ]
