"""Recorded-real-response replay for the Mirascope v2 framework adapter.

Replays the REAL upstream ``chat.completions`` body that ``ollama/llama3:8b``
actually returned for the Insurance FNOL-intake lane — captured at the transport
by ``samples/data/generators/mirascope.py`` during the same run that produced
``samples/data/traces/industry/insurance_mirascope_fnol_intake.jsonl`` — through
a REAL ``mirascope.llm.providers.ollama.OllamaProvider``, the REAL ``@llm.call``
decorator, the REAL ``llm.Response``, and the REAL ``MirascopeAdapter``. The only
fake is the socket (``httpx.MockTransport``), so every ``model_id`` /
``provider_id`` / ``usage`` value asserted below is produced by mirascope parsing
a genuine wire body, never by a hand-built double.

Why this is the gate the unit suite is not: ``test_mirascope.py`` drives the
OpenAI provider with a synthesized completion body, so it only ever sees the
``openai/<model>[:completions]`` model-id shape. A real ollama model id
(``ollama/llama3:8b``) carries a TAG after the colon rather than a transport
suffix, and a real ollama body reports ``model: "llama3:8b"``. Both facts are
only observable against a recorded real response — and both were wrong until this
lane existed:

* ``_bare_model_id`` stripped ANY ``:suffix``, reporting ``llama3`` — a model
  that did not run, contradicted by the recorded body's own ``model`` field.
* ``_format_name`` reported the literal ``"Format"`` as the structured-output
  spec whenever the customer used ``llm.format(...)`` — which is mandatory to
  select a formatting mode, and the only way to get typed output from a model
  without tool support (llama3:8b rejects mirascope's default tool mode with a
  real 400).

The strong tells that the real shape flowed through: ``tokens_prompt/completion/
total = 986/189/1175`` are read by the real OpenAI SDK off the recorded body's
``usage`` into a real ``llm.Usage``, and ``model_id``/``provider`` are what the
real ``OllamaProvider`` stamped onto the real ``Response``.
"""

from __future__ import annotations

import sys

import httpx
import pytest

if sys.version_info < (3, 10):  # pragma: no cover - matrix pins 3.11
    pytest.skip("mirascope 2.x requires Python >= 3.10", allow_module_level=True)

pytest.importorskip("mirascope.llm", reason="mirascope not installed")
pytest.importorskip("openai", reason="mirascope[openai] not installed")

import mirascope.llm as llm  # noqa: E402  # pyright: ignore[reportMissingImports]
from pydantic import Field, BaseModel  # noqa: E402
from mirascope.llm.providers.ollama import OllamaProvider  # noqa: E402  # pyright: ignore[reportMissingImports]
from mirascope.llm.providers.provider_registry import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    PROVIDER_REGISTRY,
    provider_singleton,
    reset_provider_registry,
)

from openai import OpenAI  # noqa: E402
from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.mirascope import MirascopeAdapter  # noqa: E402

from .conftest import find_event, capture_framework_trace  # noqa: E402
from ..._recorded import load_recorded, mock_transport  # noqa: E402
from ._mirascope_support import restore_call_classes  # noqa: E402

#: The model id the recorded run actually used. The ``:8b`` is an ollama TAG —
#: part of the model's real name — NOT a mirascope transport suffix (mirascope's
#: own normaliser strips only ``:responses`` / ``:completions``).
RECORDED_MODEL_ID = "ollama/llama3:8b"
RECORDED_MODEL = "llama3:8b"


class FirstNoticeOfLoss(BaseModel):
    """The FNOL record the recorded run was asked to extract (same shape as the
    generator's, so the recorded body's JSON really does parse into it)."""

    policy_number: str = Field(description="The policyholder's stated policy number.")
    claimant_name: str = Field(description="Full name of the person reporting the loss.")
    loss_date: str = Field(description="The date the loss occurred, as stated by the caller.")
    loss_type: str = Field(description="The kind of loss.")
    loss_location: str = Field(description="Where the loss occurred.")
    insured_vehicle: str = Field(description="The insured vehicle's year, make and model.")
    damage_description: str = Field(description="The damage to the insured vehicle.")
    injuries_reported: bool = Field(description="True if ANY injury was reported.")
    injury_description: str = Field(description="The injuries reported, or 'none'.")
    police_report_number: str = Field(description="The police report number, or 'none'.")
    other_party_involved: bool = Field(description="True if another party was involved.")
    severity: str = Field(description="Triage severity: LOW, MEDIUM or HIGH.")


@pytest.fixture
def recorded_ollama():
    """A REAL mirascope ``OllamaProvider`` whose only fake is its HTTP transport.

    ``register_provider`` mutates a process-global registry, so it is restored
    afterwards — a leaked provider would silently serve later tests.
    """
    fixture = load_recorded("mirascope", "fnol_intake")
    transport, requests = mock_transport(fixture)
    provider = OllamaProvider()
    provider.client = OpenAI(
        api_key="test-key",
        base_url="http://localhost:11434/v1/",
        http_client=httpx.Client(transport=transport),
    )
    saved = dict(PROVIDER_REGISTRY)
    llm.register_provider(provider, scope="ollama/")
    try:
        yield requests
    finally:
        reset_provider_registry()
        PROVIDER_REGISTRY.update(saved)
        provider_singleton.cache_clear()
        restore_call_classes()


class TestMirascopeRecorded:
    def test_fnol_intake_over_recorded_ollama(self, mock_client, recorded_ollama):
        uploaded = capture_framework_trace(mock_client)

        adapter = MirascopeAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()

        @llm.call(RECORDED_MODEL_ID, format=llm.format(FirstNoticeOfLoss, mode="json"))
        def fnol_intake_agent(narrative: str):
            return f"Extract the FNOL record from:\n{narrative}"

        response = fnol_intake_agent("Rear-ended at a light on Lamar Blvd; policy AUTO-TX-4482910.")
        adapter.disconnect()

        # The real mirascope Response parsed the recorded body's JSON into the
        # real typed record — the framework's own deserialization, end to end.
        record = response.parse()
        assert isinstance(record, FirstNoticeOfLoss)
        assert record.policy_number == "AUTO-TX-4482910"
        assert record.claimant_name == "Denise Okonkwo"
        assert record.injuries_reported is True
        assert record.severity == "HIGH"

        events = uploaded["events"]

        mi = find_event(events, "model.invoke")["payload"]
        # The recorded body itself reports model "llama3:8b". Reporting "llama3"
        # here would name a model that did not run.
        assert mi["model"] == RECORDED_MODEL
        assert mi["model_id"] == RECORDED_MODEL_ID
        assert mi["provider"] == "ollama"
        assert mi["framework"] == "mirascope"
        # The decorated function is the honest Agent identity.
        assert mi["agent_name"] == "fnol_intake_agent"
        assert mi["function_name"] == "fnol_intake_agent"
        # The structured-output spec is the customer's model, not the wrapper.
        assert mi["response_model"] == "FirstNoticeOfLoss"
        # Real usage, read by the real OpenAI SDK off the recorded body.
        assert mi["tokens_prompt"] == 986
        assert mi["tokens_completion"] == 189
        assert mi["tokens_total"] == 1175

        cost = find_event(events, "cost.record")["payload"]
        assert cost["framework"] == "mirascope"
        assert cost["model"] == RECORDED_MODEL
        assert cost["tokens_prompt"] == 986
        assert cost["tokens_completion"] == 189
        assert cost["tokens_total"] == 1175

        # The call/result pair frames the real invocation.
        call = find_event(events, "tool.call")["payload"]
        assert call["tool_name"] == "mirascope.fnol_intake_agent"
        assert call["success"] is True
        assert call["response_model"] == "FirstNoticeOfLoss"

        result = find_event(events, "tool.result")["payload"]
        assert result["success"] is True
        assert "AUTO-TX-4482910" in str(result["output"])

    def test_request_carries_the_tagged_model_name(self, mock_client, recorded_ollama):
        """The tag is part of the name mirascope puts ON THE WIRE — which is why
        dropping it downstream misreports the model."""
        adapter = MirascopeAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()

        @llm.call(RECORDED_MODEL_ID, format=llm.format(FirstNoticeOfLoss, mode="json"))
        def fnol_intake_agent(narrative: str):
            return narrative

        fnol_intake_agent("Rear-ended at a light.")
        adapter.disconnect()

        import json as _json

        body = _json.loads(recorded_ollama[0].content)
        assert body["model"] == RECORDED_MODEL
