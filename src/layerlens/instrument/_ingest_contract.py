"""The per-event-type ingest field contract, declared ONCE (LAY-3622 F1).

Before this module the contract existed in two places and nowhere authoritative:
docstring prose in four adapters ("required at ingest") and a table hand-copied
into one test lane. That is precisely how it came to contradict itself —
``tests/instrument/adapters/frameworks/test_instructor.py`` asserts
``"model_name" not in payload`` and documents that absence as INTENTIONAL for the
framework family, while ``openinference``/``mirascope``/``instructor``/``marvin``
all claim the field is required. Both statements shipped, in the same tree.

WHO OWNS THIS CONTRACT

* **atlas does not.** ``validateTraceRecords``
  (``apps/backend/api/v1/organizations/traces/traces_create.go``) checks only that
  a record carries one recognized top-level field, and says so explicitly: "A
  per-event-type registry is deliberately out of scope (it belongs on the SDK
  side; none exists server-side)." The OTLP write path validates shape even less.
  So *nothing on our platform rejects an event for a missing field today.*
* **ateam does.** ``EventNormalizer.validate_schema``
  (``stratix/ingest/normalizer.py``, ``SCHEMA_REGISTRY``) is a strict registry and
  rejects both unknown event types and missing required fields. It is the
  reference table the ADVISORY tier below mirrors.

THE DECISION (2026-08-04, product): a small HARD core, the rest ADVISORY

Measured over every committed event corpus in this repo
(``tests/fixtures/**/*.json`` + ``samples/data/traces/**/*.jsonl``) on
2026-08-03: **717 of 905 events — 79% — do not satisfy ateam's table.** Declaring
that table hard platform-wide would mark 79% of our own committed events invalid;
enforcing it is a programme, not a commit. Declaring it fiction would discard the
one requirement we really do enforce. So the contract is split:

* :data:`HARD_REQUIRED` — enforced today, by the A11 fail-closed cost lock. One
  entry, and it is a DISJUNCTION rather than a plain field (see below).
* :data:`ADVISORY_REQUIRED` — required by ateam's normalizer, NOT by us. An
  adapter that omits these is not broken on this platform; it is non-portable to
  a strict consumer. Each entry carries its measured non-compliance rate so the
  gap can never again be mistaken for a clean slate.
* :data:`LENIENT_BOUNDARY` — registered types with nothing required, deliberately.
  Presence here is a decision, not an omission.

Reading a green test run against the ADVISORY tier as evidence that the platform
is ingest-valid is exactly the error this split exists to prevent. The
``openinference`` adapter holds itself to HARD + ADVISORY (its oracle lane asserts
both) because it was authored against ateam's bar; it is the compliant OUTLIER,
not the norm.

This module DECLARES. It deliberately does not enforce: wiring a validator into
the emit path would start the platform-wide programme silently, which is the one
outcome the decision ruled out.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Mapping, FrozenSet

#: One requirement = a group of ALTERNATIVES, satisfied when at least one member
#: is present and non-``None``. A single-member group is a plain required field.
Requirement = FrozenSet[str]

#: Enforced today. ``cost.record`` must carry a real cost OR an explicit reason it
#: has none — the disjunction the A11 lock actually enforces, not ateam's bare
#: ``cost_usd``. Cluster A (LAY-3622 / A4b) introduced the second legal state: a
#: priced model whose token shape cannot be priced carries
#: ``cost_status="unpriceable_token_shape"`` and no cost. Stating this as bare
#: ``{"cost_usd"}`` would declare the shared chokepoint's own output invalid.
#:
#: CONDITIONAL, and the condition is load-bearing: the A11 lock applies it only
#: when the model resolves to a rate. An unpriced local/custom model (ollama, a
#: bring-your-own endpoint) legitimately carries no cost at all, and enforcing
#: this unconditionally would condemn every such record. See
#: ``tests/instrument/_event_schema.py`` for the enforcement, which reads this
#: table so the two cannot drift.
HARD_REQUIRED: Dict[str, Tuple[Requirement, ...]] = {
    "cost.record": (frozenset({"cost_usd", "cost_status"}),),
}

#: Required by ateam's ``SCHEMA_REGISTRY``; advisory here. NOT enforced on this
#: platform — see the module docstring. The measured non-compliance for each is in
#: :data:`MEASURED_NON_COMPLIANCE`.
ADVISORY_REQUIRED: Dict[str, Tuple[Requirement, ...]] = {
    "model.invoke": (frozenset({"model_name"}),),
    "agent.input": (frozenset({"agent_id"}), frozenset({"input_text"})),
    "agent.output": (frozenset({"agent_id"}), frozenset({"output_text"})),
    "policy.violation": (frozenset({"policy_id"}), frozenset({"violation_type"})),
    "tool.call": (frozenset({"tool_name"}),),
}

#: Registered types with NOTHING required. A deliberate lenient boundary, not an
#: omission — an event type absent from this module entirely is rejected outright
#: by ateam's normalizer ("Unknown event_type"), so silence is not neutral.
LENIENT_BOUNDARY: FrozenSet[str] = frozenset(
    {
        "embedding.create",
        "retrieval.query",
        "evaluation.result",
        "agent.interaction",
    }
)

#: event_type -> (violating, total) over every committed event corpus, measured
#: 2026-08-03. This is the evidence the ADVISORY tier rests on; it is recorded
#: here rather than in a report so it cannot drift out of sight.
#:
#: Two honest caveats:
#:
#: 1. Measured against ateam's table, i.e. ``cost.record`` was checked for a bare
#:    ``cost_usd``, NOT for the disjunction :data:`HARD_REQUIRED` states and NOT
#:    conditioned on the model being priced. Its 44% is therefore an upper bound
#:    on non-compliance with the hard core, not a measurement of it.
#: 2. It is NOT an artifact of the redaction backstop: fixtures recorded with
#:    ``capture_content=True`` are equally non-compliant, so those adapters never
#:    emit the fields at all.
#:
#: ``policy.violation`` is absent because the corpus contains none — an unmeasured
#: cell, deliberately left empty rather than filled with a zero that would read as
#: "fully compliant".
MEASURED_NON_COMPLIANCE: Dict[str, Tuple[int, int]] = {
    "model.invoke": (211, 218),
    "agent.input": (211, 217),
    "agent.output": (199, 205),
    "cost.record": (91, 207),
    "tool.call": (5, 58),
}

#: Total across the measured types: 717 of 905 committed events, 79%.
MEASURED_TOTAL: Tuple[int, int] = (717, 905)

#: Every event type with a declared contract, at any tier. Deliberately NOT named
#: ``KNOWN_EVENT_TYPES``: ``tests/instrument/_event_schema.py`` already owns that
#: name for a different set (every event type any adapter emits, contract or not).
REGISTERED_EVENT_TYPES: FrozenSet[str] = frozenset(HARD_REQUIRED) | frozenset(ADVISORY_REQUIRED) | LENIENT_BOUNDARY


def is_satisfied(requirement: Requirement, payload: Mapping[str, Any]) -> bool:
    """True when *payload* carries at least one alternative of *requirement*.

    Presence means present AND non-``None``. An empty string counts as present:
    ``openinference`` deliberately emits ``input_text=""`` under
    ``capture_content=False`` so the turn is recorded rather than lost, and
    treating that as absent would call the privacy-preserving path a violation.
    """
    return any(payload.get(alt) is not None for alt in requirement)


def unsatisfied(requirements: Tuple[Requirement, ...], payload: Mapping[str, Any]) -> List[Requirement]:
    """The requirement groups *payload* fails to satisfy, in declaration order."""
    return [req for req in requirements if not is_satisfied(req, payload)]


def requirements_for(event_type: str, *, include_advisory: bool) -> Tuple[Requirement, ...]:
    """The requirement groups for *event_type*.

    ``include_advisory=False`` returns only the enforced core. ``True`` returns the
    strict (ateam) bar — what the ``openinference`` oracle lane holds itself to.
    An unregistered type returns an empty tuple; callers that need to treat an
    unregistered type as an error should check :data:`KNOWN_EVENT_TYPES`.
    """
    groups = HARD_REQUIRED.get(event_type, ())
    if include_advisory:
        groups = groups + ADVISORY_REQUIRED.get(event_type, ())
    return groups


def describe(requirement: Requirement) -> str:
    """Render a requirement group for an assertion message: ``a`` or ``a|b``."""
    return "|".join(sorted(requirement))
