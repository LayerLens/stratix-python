"""Recorded-real-response replay for the Instructor framework (LAY-3614).

Drives a REAL ``instructor.from_openai(OpenAI())`` patched client backed by a
real ``openai.OpenAI`` over ``httpx.MockTransport`` serving the captured OpenAI
response, with the real ``InstructorAdapter`` attached. This exercises the full
path — real provider ``chat.completion`` body -> real instructor ``Mode.TOOLS``
parser -> real Pydantic validation into ``ContractMetadata`` -> real adapter ->
emitted events — which the unit doubles (hand-built response objects) never
combine.

Unlike pydantic_ai/smolagents this does NOT reuse the shared ``openai`` corpus:
instructor's ``Mode.TOOLS`` asserts the returned ``tool_call.function.name``
equals the ``response_model``'s schema name, so a ``get_weather`` body can never
parse into a domain model. The fixture is instructor's own —
``tests/fixtures/recorded/instructor/contract_extract.json`` — captured UPSTREAM
of instructor's parser by ``samples/data/generators/instructor.py``, from the
SAME real gpt-4o-mini run that sealed the Family-B trace
``samples/data/traces/industry/legal_instructor_contract_extract.jsonl``.

The strong tells that the real provider shape flowed through:

* ``model.invoke`` reports ``tokens_prompt/completion/total = 947/125/1072``,
  which the adapter normalizes off ``response._raw_response.usage`` — the usage
  block instructor hangs off the REAL parsed ``ChatCompletion``, reachable only
  because instructor stashed the raw response on the validated Pydantic model.
  Nothing in this test configures those numbers.
* the returned object is a REAL validated ``ContractMetadata`` whose fields
  (36-month initial term, 12-month auto-renewal, 90-day non-renewal notice, New
  York law) are the values the live model actually extracted, decoded out of the
  recorded ``tool_call.function.arguments`` JSON by instructor's real parser.
* ``cost.record`` carries a real ``cost_usd`` priced from those real tokens.

(``model`` is the *requested* ``gpt-4o-mini`` rather than the response's
``gpt-4o-mini-2024-07-18``: the adapter reads the model off the caller's kwargs,
which is the honest source for instructor — the tokens are the response-derived
strong values.)
"""

from __future__ import annotations

from typing import List, Optional

import httpx
import pytest

import openai

pytest.importorskip("instructor")  # skips in the base venv (not installed there)

import instructor  # noqa: E402
from pydantic import Field, BaseModel  # noqa: E402

from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.instructor import InstructorAdapter  # noqa: E402

from .conftest import find_event, capture_framework_trace  # noqa: E402
from ..._recorded import load_recorded, mock_transport  # noqa: E402


# The response_model the recorded run was made with. Its class name IS the tool
# name in the recorded body ("ContractMetadata") and its fields ARE the recorded
# ``arguments`` keys — instructor's real parser reconciles the two, so a drift in
# either direction fails here rather than in production.
class Party(BaseModel):
    """A contracting party as the agreement's preamble defines it."""

    name: str = Field(description="Legal entity name of the party, as written in the agreement.")
    role: str = Field(description="The party's contract-defined role, e.g. 'Provider' or 'Customer'.")
    entity_type: Optional[str] = Field(
        default=None,
        description="Entity form as stated, e.g. 'Delaware corporation', 'New York limited liability company'.",
    )


class ContractMetadata(BaseModel):
    """The deal terms a legal-ops team abstracts off an inbound agreement."""

    agreement_title: str = Field(description="The agreement's title, e.g. 'Master Services Agreement'.")
    parties: List[Party] = Field(description="Every contracting party named in the preamble.")
    effective_date: str = Field(description="The stated Effective Date, as written in the agreement.")
    initial_term_months: int = Field(description="Length of the Initial Term in months.")
    governing_law: str = Field(description="The jurisdiction whose law governs, e.g. 'State of New York'.")
    exclusive_venue: Optional[str] = Field(
        default=None, description="The exclusive forum for disputes, if the agreement states one."
    )
    auto_renews: bool = Field(description="True if the agreement automatically renews at the end of a term.")
    renewal_term_months: Optional[int] = Field(
        default=None, description="Length of each Renewal Term in months, if stated."
    )
    non_renewal_notice_days: Optional[int] = Field(
        default=None, description="Days of advance written notice required to stop a renewal, if stated."
    )


def _patched_client(fixture):
    transport, requests = mock_transport(fixture)
    # instructor patches a real openai client; inject the MockTransport through
    # openai's documented ``http_client=`` seam so the real SDK does its real
    # deserialization of the recorded body before instructor parses it.
    raw = openai.OpenAI(api_key="test-key", http_client=httpx.Client(transport=transport))
    return instructor.from_openai(raw), requests


class TestInstructorRecorded:
    def test_extraction_over_recorded_openai(self, mock_client):
        fixture = load_recorded("instructor", "contract_extract")
        uploaded = capture_framework_trace(mock_client)

        patched, requests = _patched_client(fixture)
        adapter = InstructorAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=patched, agent_name="contract-metadata-extractor")
        result = patched.chat.completions.create(
            model="gpt-4o-mini",
            response_model=ContractMetadata,
            max_retries=2,
            temperature=0,
            messages=[
                {"role": "system", "content": "Abstract the agreement's deal terms."},
                {"role": "user", "content": "MASTER SERVICES AGREEMENT ..."},
            ],
        )
        adapter.disconnect()

        # ---- instructor's REAL Mode.TOOLS parser validated the recorded body ----
        # A real Pydantic object, not a dict: the tool_call arguments JSON really
        # round-tripped through ContractMetadata.model_validate.
        assert isinstance(result, ContractMetadata)
        assert result.agreement_title == "MASTER SERVICES AGREEMENT"
        assert result.initial_term_months == 36
        assert result.governing_law == "State of New York"
        assert result.exclusive_venue == "New York County, New York"
        assert result.auto_renews is True
        assert result.renewal_term_months == 12
        assert result.non_renewal_notice_days == 90
        # The nested list[Party] model really deserialized too.
        assert [p.role for p in result.parties] == ["Provider", "Customer"]
        assert result.parties[0].name == "NORTHWIND ANALYTICS, INC."
        assert result.parties[0].entity_type == "Delaware corporation"
        assert result.parties[1].name == "MERIDIAN HEALTH PARTNERS, LLC"

        # The real request really went out through the openai client's transport.
        assert len(requests) == 1
        assert requests[0].url.path == "/v1/chat/completions"

        events = uploaded["events"]

        # ---- the adapter read the REAL usage off response._raw_response ----
        # 947/125/1072 exist nowhere but the recorded provider body.
        mi = find_event(events, "model.invoke")
        assert mi["payload"]["framework"] == "instructor"
        assert mi["payload"]["model"] == "gpt-4o-mini"
        assert mi["payload"]["provider"] == "openai"
        assert mi["payload"]["response_model"] == "ContractMetadata"
        assert mi["payload"]["status"] == "ok"
        assert mi["payload"]["tokens_prompt"] == 947
        assert mi["payload"]["tokens_completion"] == 125
        assert mi["payload"]["tokens_total"] == 1072

        # The caller-DECLARED identity is what fills the Agent column (instructor
        # declares no agent of its own — see the adapter's _honest_agent_name).
        assert mi["payload"]["agent_name"] == "contract-metadata-extractor"

        # max_retries=2 is the CONFIGURED maximum; hooks observed ZERO real
        # retries (the recorded body parsed first try). The adapter must report
        # the observation, never backfill it from the configuration.
        assert mi["payload"]["max_retries_configured"] == 2
        assert mi["payload"]["retries_observed"] == 0

        # The extracted object rides the event as the real structured output.
        assert mi["payload"]["output_message"]["initial_term_months"] == 36

        # ---- cost.record priced from the REAL tokens + REAL model ----
        cost = find_event(events, "cost.record")
        assert cost["payload"]["framework"] == "instructor"
        assert cost["payload"]["model"] == "gpt-4o-mini"
        assert cost["payload"]["tokens_prompt"] == 947
        assert cost["payload"]["tokens_completion"] == 125
        assert cost["payload"]["tokens_total"] == 1072
        # gpt-4o-mini: 947 prompt + 125 completion, priced by the shared table.
        assert cost["payload"]["cost_usd"] == pytest.approx(0.00021705, rel=1e-6)

        # The declared identity surfaced into the canonical agent.identity event
        # (this is what the Agent column resolves from).
        ident = find_event(events, "agent.identity")
        assert ident["payload"]["agent_name"] == "contract-metadata-extractor"
