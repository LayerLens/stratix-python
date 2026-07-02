"""Every trace with a producer-DECLARED agent identity must carry ONE canonical,
attestation-covered ``agent.identity`` event — and a trace with no honest agent
identity must carry NONE (an honest "—", never a fabricated name).

Goal (agent-identity fill): the traces "Agent" column was 2.8% filled because the
honest identity a producer already declares (a @stratix.trace name, a crew/agent
name, a langgraph node) lived scattered across per-adapter payload keys
(``crew_name``/``agent_name``/``node``) that the server never reads, while
provider-only / model / span_name signals are NOT agent identities. The SDK now
canonicalizes the honest, declared name into ONE ``agent.identity`` event at flush
so the server + FE surface it from a single place — and REFUSES to synthesize one
from a model name, an API method name, a span_name, or a class default.

This is the DUAL of ``test_trace_root_span.py``: the root marker asserts
``"agent_name" not in payload`` (never fabricate); this asserts a real declared
name IS surfaced, and a non-identity signal is NOT.

Bite: revert the flush() identity hook and every "identity surfaced" assertion
here goes RED; loosen the anti-fabrication guards and the model-as-agent /
span_name / api-method cases go RED.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from layerlens.instrument import trace, trace_context
from layerlens.instrument._collector import TraceCollector
from layerlens.instrument._capture_config import CaptureConfig

from ._event_schema import KNOWN_EVENT_TYPES

# The synthesized identity is a real, uploaded event — it participates in the
# schema lock like everything else.
pytestmark = pytest.mark.invariant

IDENTITY_TYPE = "agent.identity"


def _identity_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [e for e in events if e["event_type"] == IDENTITY_TYPE]


def _agent_name(events: List[Dict[str, Any]]) -> Optional[str]:
    ids = _identity_events(events)
    if not ids:
        return None
    return (ids[0]["payload"] or {}).get("agent_name")


# ===========================================================================
# 1. HONEST, declared names ARE surfaced — one canonical agent.identity event.
# ===========================================================================


class TestDeclaredNameIsSurfaced:
    def test_trace_decorator_name_becomes_identity(self, mock_client: Any, capture_trace_list: Any) -> None:
        @trace(mock_client, name="customer-support")
        def run() -> str:
            return "done"

        run()
        events = capture_trace_list[0]["events"]
        assert _agent_name(events) == "customer-support"
        ident = _identity_events(events)[0]
        assert (ident["payload"] or {}).get("source") == "decorator"

    def test_crew_name_becomes_identity(self, mock_client: Any, capture_trace_list: Any) -> None:
        # crewai-shape: an event whose payload declares crew_name (+ framework).
        collector = TraceCollector(mock_client, CaptureConfig.standard())
        collector.emit(
            "agent.input",
            {"framework": "crewai", "crew_name": "research-crew"},
            span_id="root0",
            parent_span_id=None,
            span_name="research-crew",
        )
        collector.flush()
        events = capture_trace_list[0]["events"]
        assert _agent_name(events) == "research-crew"

    def test_agent_name_becomes_identity(self, mock_client: Any, capture_trace_list: Any) -> None:
        # agno/agentforce/strands-shape: payload declares agent_name.
        collector = TraceCollector(mock_client, CaptureConfig.standard())
        collector.emit(
            "agent.input",
            {"framework": "agno", "agent_name": "finance_agent"},
            span_id="root0",
            parent_span_id=None,
        )
        collector.flush()
        assert _agent_name(capture_trace_list[0]["events"]) == "finance_agent"

    def test_langgraph_node_becomes_identity(self, mock_client: Any, capture_trace_list: Any) -> None:
        collector = TraceCollector(mock_client, CaptureConfig.standard())
        collector.emit(
            "agent.node.enter",
            {"framework": "langgraph", "node": "planner"},
            span_id="n1",
            parent_span_id=None,
        )
        collector.emit(
            "model.invoke",
            {"provider": "openai", "model": "gpt-4o", "latency_ms": 1.0},
            span_id="leaf",
            parent_span_id="n1",
        )
        collector.flush()
        assert _agent_name(capture_trace_list[0]["events"]) == "planner"

    def test_exactly_one_identity_event(self, mock_client: Any, capture_trace_list: Any) -> None:
        # Two agents named across the trace -> still exactly one canonical identity
        # (the primary), never one-per-event.
        collector = TraceCollector(mock_client, CaptureConfig.standard())
        collector.emit("agent.input", {"framework": "crewai", "crew_name": "crew-A"}, span_id="a", parent_span_id=None)
        collector.emit("agent.step", {"framework": "crewai", "agent_name": "worker-1"}, span_id="b", parent_span_id="a")
        collector.flush()
        assert len(_identity_events(capture_trace_list[0]["events"])) == 1


# ===========================================================================
# 2. Anti-fabrication: a NON-identity signal must NOT become an agent.
# ===========================================================================


class TestNeverFabricates:
    def test_provider_only_gets_no_identity(self, mock_client: Any, capture_trace_list: Any) -> None:
        # A bare provider call (no @trace, no agent name) is honestly "—".
        with trace_context(mock_client):
            from layerlens.instrument._context import _current_span_id, _current_collector

            col = _current_collector.get()
            parent = _current_span_id.get()
            col.emit(
                "model.invoke",
                {"provider": "openai", "model": "gpt-4o-mini", "latency_ms": 5.0},
                span_id="leaf",
                parent_span_id=parent,
            )
        assert _identity_events(capture_trace_list[0]["events"]) == []

    def test_model_as_agent_name_is_rejected(self, mock_client: Any, capture_trace_list: Any) -> None:
        # pydantic-ai shape: it stuffs the MODEL into agent_name. That is
        # model-as-agent — the identity scan must NOT surface it.
        collector = TraceCollector(mock_client, CaptureConfig.standard())
        collector.emit(
            "agent.input",
            {"framework": "pydantic-ai", "agent_name": "gpt-4o-mini"},
            span_id="root0",
            parent_span_id=None,
        )
        collector.emit(
            "model.invoke",
            {"provider": "openai", "model": "gpt-4o-mini", "latency_ms": 1.0},
            span_id="leaf",
            parent_span_id="root0",
        )
        collector.flush()
        assert _identity_events(capture_trace_list[0]["events"]) == []

    def test_span_name_is_never_an_identity(self, mock_client: Any, capture_trace_list: Any) -> None:
        # A trace whose ONLY name-ish signal is span_name (the red-teamed fallback
        # that fabricated 137 fake agents incl. a planted secret) gets NO identity.
        collector = TraceCollector(mock_client, CaptureConfig.standard())
        collector.emit(
            "model.invoke",
            {"provider": "anthropic", "model": "claude-haiku", "latency_ms": 1.0},
            span_id="leaf",
            parent_span_id="ambient00000000",
            span_name="SECRET-sk-planted-do-not-surface",
        )
        collector.flush()
        assert _identity_events(capture_trace_list[0]["events"]) == []

    def test_api_method_name_on_model_invoke_is_not_identity(self, mock_client: Any, capture_trace_list: Any) -> None:
        # provider adapters set payload.name to the API method (openai.chat...);
        # that is a method label, not an agent — must not be surfaced.
        collector = TraceCollector(mock_client, CaptureConfig.standard())
        collector.emit(
            "model.invoke",
            {"provider": "openai", "model": "gpt-4o", "name": "openai.chat.completions.create", "latency_ms": 1.0},
            span_id="leaf",
            parent_span_id="ambient00000000",
        )
        collector.flush()
        assert _identity_events(capture_trace_list[0]["events"]) == []

    def test_empty_trace_gets_no_identity(self, mock_client: Any, capture_trace_list: Any) -> None:
        collector = TraceCollector(mock_client, CaptureConfig.standard())
        collector.flush()
        assert capture_trace_list == []  # nothing to flush

    @pytest.mark.parametrize(
        "generic",
        [
            "ToolCallingAgent",  # smolagents type(agent).__name__ fallback
            "CodeAgent",
            "AgentGroupChat",  # ms_agent_framework type(chat).__name__ fallback
            "Crew",  # crewai _get_name type().__name__ fallback
            "unknown",  # openai_agents / google_adk hardcoded placeholder
            "agno_agent",  # agno generic literal fallback
            "Strands Agents",  # strands generic framework-default name
            "AssistantAgent",  # autogen class name
        ],
    )
    def test_class_name_and_placeholder_fallbacks_are_rejected(
        self, mock_client: Any, capture_trace_list: Any, generic: str
    ) -> None:
        # A framework that falls back to type(agent).__name__ or a hardcoded
        # placeholder is declaring a TYPE, not a producer-chosen agent identity.
        # Surfacing it in the Agent column would be a generic label masquerading
        # as an agent — the panel confirmed this (ms_agent 'AgentGroupChat',
        # smolagents 'ToolCallingAgent'). Honest "—" beats a class name.
        collector = TraceCollector(mock_client, CaptureConfig.standard())
        collector.emit("agent.input", {"framework": "x", "agent_name": generic}, span_id="r", parent_span_id=None)
        collector.flush()
        assert _identity_events(capture_trace_list[0]["events"]) == [], f"{generic!r} must not surface as an agent"

    def test_real_distinctive_names_still_surface(self, mock_client: Any, capture_trace_list: Any) -> None:
        # The guard must NOT hide genuine, producer-chosen names.
        for name in ("customer-support", "research crew", "finance_agent", "Acme", "researcher"):
            collector = TraceCollector(mock_client, CaptureConfig.standard())
            collector.emit("agent.input", {"framework": "x", "agent_name": name}, span_id="r", parent_span_id=None)
            collector.flush()
        names = [_agent_name(t["events"]) for t in capture_trace_list]
        assert names == ["customer-support", "research crew", "finance_agent", "Acme", "researcher"]

    @pytest.mark.parametrize(
        "generic",
        [
            "SequentialAgent",  # google_adk workflow-agent class default
            "ParallelAgent",
            "LoopAgent",
            "BaseAgent",  # already covered, sanity
        ],
    )
    def test_adk_workflow_class_names_are_rejected(
        self, mock_client: Any, capture_trace_list: Any, generic: str
    ) -> None:
        # An unnamed google_adk SequentialAgent/ParallelAgent/LoopAgent falls back
        # to type(agent).__name__ — a workflow-container class, not a producer-chosen
        # identity. Live-verified this leaked (probe P-F). Honest "—" beats a class name.
        collector = TraceCollector(mock_client, CaptureConfig.standard())
        collector.emit(
            "agent.input", {"framework": "google_adk", "agent_name": generic}, span_id="r", parent_span_id=None
        )
        collector.flush()
        assert _identity_events(capture_trace_list[0]["events"]) == [], f"{generic!r} must not surface as an agent"

    @pytest.mark.parametrize(
        "src_key,evt",
        [
            ("agent_name", "agent.input"),  # tier 2
            ("crew_name", "agent.step"),  # tier 1
            ("node", "agent.node.enter"),  # tier 3
            ("from_agent", "a2a.delegation"),  # tier 5
        ],
    )
    def test_api_method_label_is_rejected_on_every_tier(
        self, mock_client: Any, capture_trace_list: Any, src_key: str, evt: str
    ) -> None:
        # A dotted, all-lowercase API-method label (openai.chat.completions.create)
        # is a provider method, never an agent — and the guard must hold on ALL
        # tiers, not just the @trace decorator (tier 4). Live-verified a tier-2
        # api-method label surfaced (probe P-C).
        collector = TraceCollector(mock_client, CaptureConfig.standard())
        collector.emit(
            evt, {"framework": "x", src_key: "openai.chat.completions.create"}, span_id="r", parent_span_id=None
        )
        collector.flush()
        assert _identity_events(capture_trace_list[0]["events"]) == [], (
            "an API-method label must never surface as an agent"
        )

    def test_whitespace_only_name_is_rejected(self, mock_client: Any, capture_trace_list: Any) -> None:
        # A whitespace-only declared name is not a name — it must not fill the
        # column with blanks. Live-verified it surfaced (probe P-D).
        collector = TraceCollector(mock_client, CaptureConfig.standard())
        collector.emit("agent.input", {"framework": "x", "agent_name": "   "}, span_id="r", parent_span_id=None)
        collector.flush()
        assert _identity_events(capture_trace_list[0]["events"]) == [], "a whitespace-only name must not surface"

    def test_name_with_surrounding_whitespace_is_trimmed(self, mock_client: Any, capture_trace_list: Any) -> None:
        # A genuine name with incidental surrounding whitespace still surfaces, trimmed.
        collector = TraceCollector(mock_client, CaptureConfig.standard())
        collector.emit(
            "agent.input", {"framework": "x", "agent_name": "  finance_agent  "}, span_id="r", parent_span_id=None
        )
        collector.flush()
        assert _agent_name(capture_trace_list[0]["events"]) == "finance_agent"

    def test_bidi_and_control_chars_are_stripped_from_identity(self, mock_client: Any, capture_trace_list: Any) -> None:
        # A declared name carrying a Unicode bidi-override (U+202E) or C0/C1 control
        # codepoint must not reach the Agent column verbatim — an RTL override can
        # visually reorder the label into a spoof (e.g. render "researcher" backwards
        # or hide a suffix). Strip ONLY control/bidi-format codepoints; NEVER charset:
        # the legitimate CJK stays. Live-verified the raw override surfaced (deep probe
        # identity_unicode_rtl). Defense-in-depth ahead of the FE render sink.
        collector = TraceCollector(mock_client, CaptureConfig.standard())
        collector.emit(
            "agent.input",
            {"framework": "x", "agent_name": "研究‮evil"},  # RTL override + BEL
            span_id="r",
            parent_span_id=None,
        )
        collector.flush()
        name = _agent_name(capture_trace_list[0]["events"])
        assert name is not None
        # legitimate CJK preserved; bidi-override + control byte removed
        assert name == "研究evil", f"got {name!r}"
        assert "‮" not in name and "" not in name

    def test_mixed_case_api_method_label_is_rejected(self, mock_client: Any, capture_trace_list: Any) -> None:
        # A capitalized/mixed-case dotted API-method label is STILL a provider method,
        # not an agent — the guard must be case-insensitive (re-vet residual).
        for label in ("OpenAI.Chat.Completions.Create", "Ollama.Chat", "openai.Chat.completions"):
            collector = TraceCollector(mock_client, CaptureConfig.standard())
            collector.emit("agent.input", {"framework": "x", "agent_name": label}, span_id="r", parent_span_id=None)
            collector.flush()
        for t in capture_trace_list:
            assert _identity_events(t["events"]) == [], "a mixed-case API-method label must not surface"

    def test_alm_and_line_separator_are_stripped_from_identity(self, mock_client: Any, capture_trace_list: Any) -> None:
        # U+061C (Arabic Letter Mark, a bidi control) and U+2028 (line separator) are
        # in-scope format codepoints — strip them (re-vet residual), keep the letters.
        collector = TraceCollector(mock_client, CaptureConfig.standard())
        collector.emit(
            "agent.input",
            {"framework": "x", "agent_name": "؜billing copilot"},
            span_id="r",
            parent_span_id=None,
        )
        collector.flush()
        name = _agent_name(capture_trace_list[0]["events"])
        assert name == "billingcopilot", f"got {name!r}"  # ALM + line-sep removed, letters kept

    def test_name_that_is_only_control_chars_is_rejected(self, mock_client: Any, capture_trace_list: Any) -> None:
        # After stripping control/bidi codepoints, a name that becomes empty is not
        # a declared identity — reject it (honest "—"), like whitespace-only.
        collector = TraceCollector(mock_client, CaptureConfig.standard())
        collector.emit(
            "agent.input",
            {"framework": "x", "agent_name": "‪‬‎"},  # only bidi-format
            span_id="r",
            parent_span_id=None,
        )
        collector.flush()
        assert _identity_events(capture_trace_list[0]["events"]) == [], "a control-char-only name must not surface"


class TestProtocolPeerIdentity:
    def test_a2a_delegation_from_agent_becomes_identity(self, mock_client: Any, capture_trace_list: Any) -> None:
        # The local delegating agent of an a2a.delegation is a producer-declared
        # agent identity (topology, like agent.handoff.from_agent).
        collector = TraceCollector(mock_client, CaptureConfig.standard())
        collector.emit(
            "a2a.delegation",
            {"protocol": "a2a", "from_agent": "orchestrator", "to_agent": "remote-worker"},
            span_id="d1",
            parent_span_id=None,
        )
        collector.flush()
        assert _agent_name(capture_trace_list[0]["events"]) == "orchestrator"


# ===========================================================================
# 3. The identity event is structural: survives redaction, is registered, and
#    is inside the attestation chain.
# ===========================================================================


class TestIdentityIsStructuralAndAttested:
    def test_identity_type_is_registered(self) -> None:
        assert IDENTITY_TYPE in KNOWN_EVENT_TYPES

    def test_agent_name_survives_capture_content_false(self, mock_client: Any, capture_trace_list: Any) -> None:
        cfg = CaptureConfig(capture_content=False)
        collector = TraceCollector(mock_client, cfg)
        collector.emit(
            "agent.input",
            {"framework": "crewai", "crew_name": "research-crew", "input": "secret prompt"},
            span_id="root0",
            parent_span_id=None,
        )
        collector.flush()
        # The declared name is structural identity (like from_agent/to_agent
        # topology) — it survives no-content redaction so the Agent column fills.
        assert _agent_name(capture_trace_list[0]["events"]) == "research-crew"

    def test_identity_carries_no_content(self, mock_client: Any, capture_trace_list: Any) -> None:
        collector = TraceCollector(mock_client, CaptureConfig.standard())
        collector.emit(
            "agent.input", {"framework": "agno", "agent_name": "a1", "input": "hi"}, span_id="r", parent_span_id=None
        )
        collector.flush()
        ident = _identity_events(capture_trace_list[0]["events"])[0]
        payload = ident["payload"] or {}
        for banned in ("input", "output", "messages", "content", "prompt"):
            assert banned not in payload

    def test_attestation_covers_identity_event(self, mock_client: Any, capture_trace_list: Any) -> None:
        from layerlens.attestation._chain import HashChain
        from layerlens.attestation._verify import verify_trial
        from layerlens.attestation._envelope import HashScope, AttestationEnvelope

        collector = TraceCollector(mock_client, CaptureConfig.standard())
        collector.emit("agent.input", {"framework": "crewai", "crew_name": "c1"}, span_id="r", parent_span_id=None)
        collector.flush()

        payload = capture_trace_list[0]
        assert _identity_events(payload["events"]), "identity event must be present"
        att = payload["attestation"]
        assert att.get("root_hash"), "trace must still be attested"
        rebuilt = HashChain()
        for e in payload["events"]:
            rebuilt.add_event(e)
        envelopes = rebuilt.envelopes
        trial = AttestationEnvelope(hash=att["root_hash"], scope=HashScope.TRIAL, previous_hash=envelopes[-1].hash)
        result = verify_trial(envelopes, trial)
        assert result.trial_hash_valid, f"root hash does not cover the identity event: {result.errors}"
        chain_events = att["chain"]["events"]
        assert len(chain_events) == len(payload["events"]), "identity event must be chained"

    def test_explicitly_emitted_identity_is_not_doubled(self, mock_client: Any, capture_trace_list: Any) -> None:
        # If an adapter ever emits agent.identity itself, flush must not add a 2nd.
        collector = TraceCollector(mock_client, CaptureConfig.standard())
        collector.emit(
            IDENTITY_TYPE,
            {"framework": "custom", "agent_name": "explicit-agent", "source": "adapter"},
            span_id="r",
            parent_span_id=None,
        )
        collector.emit("agent.input", {"framework": "crewai", "crew_name": "other"}, span_id="r2", parent_span_id="r")
        collector.flush()
        idents = _identity_events(capture_trace_list[0]["events"])
        assert len(idents) == 1
        assert (idents[0]["payload"] or {}).get("agent_name") == "explicit-agent"
