"""Self-tests for the declared ingest contract (LAY-3622 F1).

Two jobs:

1. Pin the registry's own semantics — in particular that a present-but-EMPTY
   string SATISFIES a requirement. ``openinference`` emits ``input_text=""`` under
   ``capture_content=False`` precisely so the turn is recorded rather than lost;
   a registry that called that a violation would condemn the privacy-preserving
   path.
2. **The drift guard.** The enforced core is DECLARED in
   ``_ingest_contract.HARD_REQUIRED``, but the A11 fail-closed cost lock in
   ``tests/instrument/_event_schema.py`` implements the enforcement by hand,
   because it is CONDITIONAL (priced models only) and a naive "any alternative
   present" check would wrongly accept a marker with no cost. Two artifacts
   stating one contract is exactly the shape that let this contract contradict
   itself before F1, so these tests drive the real lock and assert its verdicts
   match the declaration.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from tests.instrument._event_schema import validate_event
from layerlens.instrument._ingest_contract import (
    HARD_REQUIRED,
    MEASURED_TOTAL,
    LENIENT_BOUNDARY,
    ADVISORY_REQUIRED,
    REGISTERED_EVENT_TYPES,
    MEASURED_NON_COMPLIANCE,
    describe,
    unsatisfied,
    is_satisfied,
    requirements_for,
)

# Run as a required CI gate via `-m invariant`: the drift guard protects a
# fail-closed billing lock, so it belongs in the invariant population.
pytestmark = pytest.mark.invariant


class TestRequirementSemantics:
    def test_an_empty_string_satisfies_a_requirement(self) -> None:
        # The openinference privacy convention. If this flips, the default
        # capture_content=False config becomes "contract-violating" by definition
        # and F2's whole justification inverts.
        assert is_satisfied(frozenset({"input_text"}), {"input_text": ""})

    def test_none_does_not_satisfy_a_requirement(self) -> None:
        assert not is_satisfied(frozenset({"input_text"}), {"input_text": None})

    def test_an_absent_key_does_not_satisfy_a_requirement(self) -> None:
        assert not is_satisfied(frozenset({"input_text"}), {})

    def test_a_disjunction_is_satisfied_by_either_alternative(self) -> None:
        group = frozenset({"cost_usd", "cost_status"})
        assert is_satisfied(group, {"cost_usd": 0.01})
        assert is_satisfied(group, {"cost_status": "unpriceable_token_shape"})
        assert not is_satisfied(group, {"total_tokens": 10})

    def test_a_zero_cost_satisfies_the_disjunction(self) -> None:
        # 0.0 is a legal cost (a call that truly billed nothing); the registry
        # tests PRESENCE, not truthiness. `payload.get(alt) is not None` is
        # load-bearing here — a falsy-check would read $0.00 as "no cost".
        assert is_satisfied(frozenset({"cost_usd", "cost_status"}), {"cost_usd": 0.0})

    def test_unsatisfied_reports_only_the_failing_groups(self) -> None:
        groups = (frozenset({"agent_id"}), frozenset({"input_text"}))
        assert unsatisfied(groups, {"agent_id": "a", "input_text": ""}) == []
        assert unsatisfied(groups, {"agent_id": "a"}) == [frozenset({"input_text"})]
        assert len(unsatisfied(groups, {})) == 2

    def test_describe_renders_a_disjunction_stably(self) -> None:
        assert describe(frozenset({"cost_usd", "cost_status"})) == "cost_status|cost_usd"
        assert describe(frozenset({"tool_name"})) == "tool_name"


class TestTierStructure:
    def test_requirements_for_excludes_advisory_by_default(self) -> None:
        # cost.record is the ONLY enforced entry; model.invoke's model_name is
        # advisory. Collapsing the two tiers would silently start the
        # platform-wide enforcement programme the F1 decision ruled out.
        assert requirements_for("cost.record", include_advisory=False) == HARD_REQUIRED["cost.record"]
        assert requirements_for("model.invoke", include_advisory=False) == ()
        assert requirements_for("model.invoke", include_advisory=True) == ADVISORY_REQUIRED["model.invoke"]

    def test_the_strict_bar_is_the_union_of_both_tiers(self) -> None:
        strict = requirements_for("cost.record", include_advisory=True)
        assert set(strict) == set(HARD_REQUIRED["cost.record"]) | set(ADVISORY_REQUIRED.get("cost.record", ()))

    def test_an_unregistered_type_has_no_requirements(self) -> None:
        assert requirements_for("not.a.real.type", include_advisory=True) == ()

    def test_registered_event_types_is_the_union_of_the_three_tiers(self) -> None:
        assert REGISTERED_EVENT_TYPES == set(HARD_REQUIRED) | set(ADVISORY_REQUIRED) | LENIENT_BOUNDARY

    def test_a_lenient_type_is_never_also_required(self) -> None:
        # "nothing required" and "these fields are required" are contradictory
        # claims about one event type.
        assert not LENIENT_BOUNDARY & set(HARD_REQUIRED)
        assert not LENIENT_BOUNDARY & set(ADVISORY_REQUIRED)

    def test_no_requirement_group_is_empty(self) -> None:
        # An empty group can never be satisfied, so it would be a permanent
        # violation. "Nothing required" is expressed by LENIENT_BOUNDARY.
        for table in (HARD_REQUIRED, ADVISORY_REQUIRED):
            for event_type, groups in table.items():
                assert groups, f"{event_type} has an empty requirement tuple"
                for group in groups:
                    assert group, f"{event_type} has an empty requirement group"


class TestMeasuredEvidence:
    """The ADVISORY tier rests on these numbers, so they must stay coherent."""

    def test_the_per_type_counts_sum_to_the_recorded_total(self) -> None:
        violating = sum(v for v, _ in MEASURED_NON_COMPLIANCE.values())
        total = sum(t for _, t in MEASURED_NON_COMPLIANCE.values())
        assert (violating, total) == MEASURED_TOTAL, (
            "MEASURED_TOTAL no longer matches the per-type breakdown — editing one "
            "cell without the other is how a measurement becomes a claim"
        )

    def test_no_type_reports_more_violations_than_events(self) -> None:
        for event_type, (violating, total) in MEASURED_NON_COMPLIANCE.items():
            assert 0 <= violating <= total, f"{event_type}: {violating} of {total} is not a rate"

    def test_every_measured_type_is_registered(self) -> None:
        assert not set(MEASURED_NON_COMPLIANCE) - REGISTERED_EVENT_TYPES


class TestA11DriftGuard:
    """The declaration and the hand-written enforcement must agree.

    Drives the REAL lock (``validate_event``) rather than restating its logic.
    """

    #: A model that resolves to a rate, and one that deliberately does not.
    PRICED = {"model": "gpt-4o", "provider": "openai"}
    UNPRICED = {"model": "totally-custom-model-xyz", "provider": "ollama"}

    @staticmethod
    def _cost_event(**payload: Any) -> Dict[str, Any]:
        return {"event_type": "cost.record", "payload": {"total_tokens": 1500, **payload}}

    def test_the_declared_core_is_exactly_the_cost_disjunction(self) -> None:
        # If someone drops an alternative from the registry, the hand-written lock
        # below keeps accepting it — this is the assertion that notices.
        assert set(HARD_REQUIRED) == {"cost.record"}, (
            "a new HARD entry was declared but nothing enforces it — either wire the enforcement or declare it ADVISORY"
        )
        assert HARD_REQUIRED["cost.record"] == (frozenset({"cost_usd", "cost_status"}),)

    def test_a_real_cost_satisfies_the_lock(self) -> None:
        assert validate_event(self._cost_event(**self.PRICED, cost_usd=0.01)) == []

    def test_the_unpriceable_marker_satisfies_the_lock(self) -> None:
        event = self._cost_event(**self.PRICED, cost_status="unpriceable_token_shape")
        assert validate_event(event) == []

    def test_a_priced_model_with_neither_alternative_is_REJECTED(self) -> None:
        # The bite: this is the A11 dropped-price bug, and the lock must catch it.
        problems = validate_event(self._cost_event(**self.PRICED))
        assert problems, "the A11 lock accepted a priced cost.record carrying neither alternative"
        assert any("no cost_usd" in p for p in problems)

    def test_the_requirement_is_CONDITIONAL_on_the_model_being_priced(self) -> None:
        # An unpriced local/custom model legitimately carries no cost at all.
        # Enforcing the core unconditionally would condemn every ollama record —
        # this conditionality is why the registry documents rather than enforces.
        assert validate_event(self._cost_event(**self.UNPRICED)) == []

    def test_the_marker_and_a_cost_together_are_REJECTED(self) -> None:
        # "no price" plus a price is self-contradictory; the lock says so.
        problems = validate_event(self._cost_event(**self.PRICED, cost_status="unpriceable_token_shape", cost_usd=0.01))
        assert any("marked unpriceable_token_shape yet carries cost_usd" in p for p in problems)

    def test_the_partial_marker_with_a_cost_and_a_magnitude_satisfies_the_lock(self) -> None:
        # LAY-3622 F4's marker is the MIRROR of unpriceable: a real cost that
        # UNDERSTATES the bill. Legal only with both a cost and the shortfall.
        event = self._cost_event(**self.PRICED, cost_usd=0.01, cost_status="partial_token_shape", unpriced_tokens=42)
        assert validate_event(event) == []

    def test_the_partial_marker_without_a_cost_is_REJECTED(self) -> None:
        # Nothing to understate — and it would masquerade as an honestly-withheld cost.
        problems = validate_event(
            self._cost_event(**self.PRICED, cost_status="partial_token_shape", unpriced_tokens=42)
        )
        assert any("no cost_usd to understate" in p for p in problems)

    def test_the_partial_marker_without_a_magnitude_is_REJECTED(self) -> None:
        # A marker with no number forces every reader to re-derive the pricing
        # arithmetic, whose obvious form is wrong on a fully-cached turn.
        problems = validate_event(self._cost_event(**self.PRICED, cost_usd=0.01, cost_status="partial_token_shape"))
        assert any("without a positive unpriced_tokens" in p for p in problems)
