"""Honest agent-identity resolution for a trace's captured events.

The traces "Agent" column must fill from a producer-DECLARED name and NEVER be
fabricated. This module is the single decision point: given a trace's events, it
returns the one honest agent identity the producer already declared — or ``None``
(an honest "—"). The collector calls it at flush to synthesize the canonical
:data:`layerlens.instrument._events.AGENT_IDENTITY` event; the server + FE then
surface that one field instead of re-deriving.

Honest sources, in priority order (all are values a producer explicitly set):
  1. ``crew_name``      — a crew/team name (crewai).
  2. ``agent_name``     — a declared agent name (agno/agentforce/openai_agents/
     ms_agent_framework/strands/smolagents/google_adk), EXCLUDING the model-as-
     agent anti-pattern (pydantic-ai stuffs the model into agent_name).
  3. ``node``           — the entered graph node (langgraph ``agent.node.enter``).
  4. ``name``           — the @stratix.trace name (only on ``agent.input`` /
     ``agent.output`` / ``agent.error`` — the decorator's own events), EXCLUDING
     a model name or an API-method label (``openai.chat.completions.create``).
  5. ``from_agent``     — the delegating agent of an ``agent.handoff``.

NEVER a source: span_name, a model name, an API method name, a class default, or
a remote/discovered peer's ``name`` on a protocol event.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ._events import AGENT_IDENTITY

# A dotted, all-lowercase method label like ``openai.chat.completions.create`` or
# ``ollama.chat`` — a provider API method, not an agent name. (Real agent names
# use hyphens/underscores/capitals or single words: ``customer-support``,
# ``finance_agent``, ``researcher`` — none match this.)
_API_METHOD_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)+$")

# Events whose ``payload.name`` is the @stratix.trace decorator's own name. Other
# event types put an API method / class / node label in ``name`` — not an agent.
_DECORATOR_NAME_TYPES = frozenset({"agent.input", "agent.output", "agent.error"})

# Generic framework fallbacks that are NOT a producer-chosen agent identity: a
# ``type(agent).__name__`` class name or a hardcoded placeholder an adapter emits
# when the real name is unset. Surfacing one in the Agent column is a generic
# label masquerading as an agent (the adversarial panel confirmed the class-name
# cases: smolagents 'ToolCallingAgent', ms_agent_framework 'AgentGroupChat',
# crewai 'Crew'). An honest "—" beats a class name. Census/research-grounded;
# case-insensitive. NB: only clear generics — a genuine distinctive name a
# developer set (``customer-support``, ``research crew``, ``finance_agent``)
# must still surface, so this is a precise denylist, not a CamelCase heuristic
# (which would hide a legitimately CamelCase-named agent).
_GENERIC_IDENTITY_VALUES = frozenset(
    {
        # hardcoded placeholders
        "unknown",
        "none",
        "n/a",
        "default",
        "agent",
        # generic framework-default literals (every unnamed agent gets these)
        "agno_agent",
        "pydantic_ai_agent",
        "strands agents",
        "collaborator",
        # class names surfaced via type(x).__name__ fallbacks
        "toolcallingagent",
        "codeagent",
        "multistepagent",
        "agentgroupchat",
        # google_adk workflow-agent class defaults (type(agent).__name__ when
        # an orchestration agent is left unnamed) — containers, not identities.
        "sequentialagent",
        "parallelagent",
        "loopagent",
        "crew",
        "flow",
        "chatopenai",
        "llmagent",
        "baseagent",
        "assistantagent",
        "userproxyagent",
        "conversableagent",
        "roundrobingroupchat",
    }
)


def _is_generic(name: str) -> bool:
    return name.strip().lower() in _GENERIC_IDENTITY_VALUES


# Unicode codepoints that must never reach the Agent column: C0/C1 control
# characters and bidi override/embedding/isolate + directional/format marks. A
# name carrying U+202E (RTL override) can visually reorder or hide part of the
# label (a display-spoof), and control/format bytes corrupt the column or inject a
# line break. We strip ONLY control/format codepoints — NEVER charset — so a
# legitimate international name (CJK, accents, letters) surfaces intact. Zero-width
# joiners (U+200C/U+200D) ARE stripped as a spoofing/invisible-text vector, so a
# ZWJ emoji sequence is decomposed and a ZWNJ-spelled name is normalized — an
# intentional security-over-fidelity tradeoff for the Agent LABEL.
_CONTROL_BIDI_RE = re.compile(
    "["
    "\x00-\x1f\x7f-\x9f"  # C0 + C1 control characters
    "\u061c"  # Arabic Letter Mark (bidi control)
    "\u200b-\u200f"  # zero-width chars + LRM/RLM directional marks
    "\u2028\u2029"  # line / paragraph separators
    "\u202a-\u202e"  # bidi embeddings + overrides (incl. U+202E RLO)
    "\u2066-\u2069"  # bidi isolates
    "\ufeff"  # zero-width no-break space / BOM
    "]"
)


def _s(value: Any) -> Optional[str]:
    """A non-empty, whitespace-trimmed, control/bidi-sanitized string, else None.

    Trims surrounding whitespace so an incidental ``"  finance_agent  "`` still
    surfaces (as ``"finance_agent"``) while a whitespace-only value is rejected
    outright — a blank is not a declared identity. Also removes C0/C1 control and
    bidi-format codepoints (:data:`_CONTROL_BIDI_RE`) so a hostile declared name
    cannot spoof or corrupt the Agent column; a name that is ONLY such codepoints
    is rejected like a blank."""
    if isinstance(value, str):
        stripped = _CONTROL_BIDI_RE.sub("", value).strip()
        if stripped:
            return stripped
    return None


def _trace_models(events: List[Dict[str, Any]]) -> set[str]:
    """Lower-cased set of every model referenced in the trace — used to reject
    the model-as-agent anti-pattern (agent_name/name == the model)."""
    models: set[str] = set()
    for e in events:
        p = e.get("payload") or {}
        if isinstance(p, dict):
            m = _s(p.get("model"))
            if m:
                models.add(m.lower())
    return models


def honest_agent_identity(events: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the one honest agent identity for *events*, or None.

    Result: ``{"agent_name", "source", "framework", "span_id", "parent_span_id"}``
    — the ``span_id``/``parent_span_id`` are copied from the source event so the
    collector can co-locate the identity marker on an existing captured span
    (never adding a new tree node). Returns None when the trace already carries an
    ``agent.identity`` event (never double) or has no declared name.
    """
    if any(e.get("event_type") == AGENT_IDENTITY for e in events):
        return None

    models = _trace_models(events)

    def _hit(event: Dict[str, Any], name: str, source: str) -> Dict[str, Any]:
        p = event.get("payload") or {}
        return {
            "agent_name": name,
            "source": source,
            "framework": _s(p.get("framework")),
            "span_id": event.get("span_id"),
            "parent_span_id": event.get("parent_span_id"),
        }

    def _honest(name: str) -> bool:
        """A producer-chosen name: not the model (model-as-agent), not a generic
        class-name / placeholder fallback, and not a dotted API-method label
        (``openai.chat.completions.create``). The API-method guard applies on
        EVERY tier — a provider method is never an agent, whether it lands in a
        crew_name/agent_name/node/from_agent or a decorator name."""
        return name.lower() not in models and not _is_generic(name) and not _API_METHOD_RE.match(name.lower())

    # Tier 1 — a crew/team name.
    for e in events:
        p = e.get("payload") or {}
        name = _s(p.get("crew_name")) if isinstance(p, dict) else None
        if name and _honest(name):
            return _hit(e, name, "framework")

    # Tier 2 — a declared agent_name (rejects pydantic-ai model-as-agent and
    # class-name/placeholder fallbacks like 'ToolCallingAgent'/'unknown').
    for e in events:
        p = e.get("payload") or {}
        name = _s(p.get("agent_name")) if isinstance(p, dict) else None
        if name and _honest(name):
            return _hit(e, name, "framework")

    # Tier 3 — the entered graph node (langgraph).
    for e in events:
        if e.get("event_type") != "agent.node.enter":
            continue
        p = e.get("payload") or {}
        name = _s(p.get("node")) if isinstance(p, dict) else None
        if name and _honest(name):
            return _hit(e, name, "framework")

    # Tier 4 — the @stratix.trace name, only on the decorator's own events.
    # (_honest already rejects a model name and an API-method label.)
    for e in events:
        if e.get("event_type") not in _DECORATOR_NAME_TYPES:
            continue
        p = e.get("payload") or {}
        name = _s(p.get("name")) if isinstance(p, dict) else None
        if name and _honest(name):
            return _hit(e, name, "decorator")

    # Tier 5 — the delegating (local) agent of a handoff or an a2a delegation
    # (cross-agent topology, producer-declared).
    for e in events:
        if e.get("event_type") not in ("agent.handoff", "a2a.delegation"):
            continue
        p = e.get("payload") or {}
        name = _s(p.get("from_agent")) if isinstance(p, dict) else None
        if name and _honest(name):
            return _hit(e, name, "handoff")

    return None
