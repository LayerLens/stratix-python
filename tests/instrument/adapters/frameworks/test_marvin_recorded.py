"""Recorded-real-response replay for the Marvin framework adapter (LAY-3614).

Drives the REAL ``marvin.cast`` / ``marvin.extract`` primitives — Marvin's own
orchestrator, its own ``Task``/``Thread``/end-turn tooling, and the real
pydantic-ai ``OpenAIChatModel`` under it — against the recorded **OpenAI
chat.completion SSE stream** that Marvin actually received during the Real-Estate
MLS listing-intake run, with the real ``MarvinAdapter`` patched onto the real
``marvin`` module. Recorded by ``samples/data/generators/marvin.py``, which
sealed these bodies and the shipped
``samples/data/traces/industry/realestate_marvin_listing_extract.jsonl`` from the
SAME run — so this gate and that sample can never drift apart.

This is the layer the unit suite (``test_marvin.py``, a pydantic-ai ``TestModel``)
cannot reach: ``TestModel`` fabricates a canned result without ever producing a
provider body, so it proves nothing about Marvin deserializing a real streamed
tool call into a real typed object and the adapter reporting it.

The strong tell that the real body flowed through: the adapter's captured
``tool.call.output`` / ``model.invoke.response`` carry the actual listing values
(``1428 Oakridge Lane``, 4 beds, 2.5 baths, ``$749,000``) — these exist ONLY
inside the recorded stream's tool-call arguments, so they can only appear here by
Marvin really deserializing it.

THE HONEST-OMISSION CONTRACT (the reason this fixture is worth its weight)
-------------------------------------------------------------------------
The recorded stream's final chunk carries a REAL OpenAI usage block
(``prompt_tokens: 1275, completion_tokens: 91, total_tokens: 1366`` for the cast).
Marvin surfaces no usage on its primitives, so the adapter deliberately reports
**no tokens and no cost.record** — the pricing hook has nothing real to price at
this layer, and any figure here would be fabricated. Asserting that against a body
that demonstrably HAS the tokens is what makes the omission provably principled
rather than an accident: if someone later "helpfully" invented token/cost fields
on marvin's tokenless ``model.invoke``, this test goes red.

Likewise ``model`` is the model the developer configured on the ``marvin.Agent``
(``gpt-4o-mini``), NOT the ``gpt-4o-mini-2024-07-18`` the recorded *response*
echoes — the adapter resolves the model off ``Agent.model`` and never reads it
back off a response body. Pinning both sides here documents that boundary against
a body that contains the other value.

DETERMINISM SEAM: Marvin names its end-turn tool ``MarkTaskSuccessful_{task.id}``
where ``Task.id`` is a fresh ``uuid.uuid4().hex[:8]`` per construction
(``marvin/engine/end_turn.py``, ``marvin/tasks/task.py``), so the recorded
response's tool name is bound to the recording run's random id. ``_pin_task_id``
freezes that ONE random id by swapping the ``uuid`` module reference inside
``marvin.tasks.task``'s namespace only — the recorded body is replayed byte-for-
byte and untouched; nothing else (the SDK's own span ids included) is affected.
"""

from __future__ import annotations

import os
import sys
import json
import uuid as _real_uuid
import tempfile

import httpx
import pytest

if sys.version_info < (3, 10):
    pytest.skip("marvin requires Python >= 3.10", allow_module_level=True)

# ``import marvin`` calls ensure_db_tables_exist() at module scope — point it at a
# throwaway file BEFORE the import so a test run never touches the user's real
# Marvin database.
os.environ.setdefault(
    "MARVIN_DATABASE_URL",
    "sqlite+aiosqlite:///" + os.path.join(tempfile.mkdtemp(prefix="layerlens-marvin-rec-"), "marvin.db"),
)

marvin = pytest.importorskip("marvin", reason="marvin not installed")

from pydantic import Field, BaseModel  # noqa: E402
from pydantic_ai.models.openai import OpenAIChatModel  # noqa: E402
from pydantic_ai.providers.openai import OpenAIProvider  # noqa: E402

from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.marvin import MarvinAdapter  # noqa: E402

from .conftest import find_event, find_events, capture_framework_trace  # noqa: E402
from ..._recorded import load_recorded, mock_transport  # noqa: E402

# Marvin's rich console handler renders a live panel per call — pure test noise.
marvin.settings.enable_default_print_handler = False

#: The random per-Task ids Marvin generated during the recording run. The
#: recorded responses call ``MarkTaskSuccessful_<id>``, so the replay must
#: reconstruct the same tool name for pydantic-ai to route the call.
_CAST_TASK_ID = "30e5df4a"
_EXTRACT_TASK_ID = "74e5f42a"

_AGENT_NAME = "listing-extraction-agent"

# The freeform MLS write-up the recording run sent. Only the tail matters for the
# replay (the request is not matched), but keeping the real input makes the
# captured ``tool.call.input`` an honest echo of the recorded run.
_LISTING_DESCRIPTION = (
    "Welcome to 1428 Oakridge Lane, a beautifully maintained 1997 craftsman-style "
    "single-family home tucked into the sought-after Oakridge Park neighborhood of "
    "Round Rock. Offered at $749,000, this 2,340 square foot residence gives you four "
    "generous bedrooms and two and a half bathrooms, including a main-floor primary "
    "suite with a spa-inspired walk-in shower and dual vanities."
)


class PropertyListing(BaseModel):
    """The MLS record schema the recording run cast to — the recorded stream's
    tool-call arguments validate against exactly this shape."""

    street_address: str = Field(description="Street address of the property.")
    property_type: str = Field(description="Property type, e.g. 'single-family'.")
    bedrooms: int = Field(description="Number of bedrooms.")
    bathrooms: float = Field(description="Number of bathrooms (half-baths count as 0.5).")
    square_feet: int = Field(description="Interior living area in square feet.")
    list_price_usd: int = Field(description="Asking price in USD.")
    year_built: int = Field(description="Year the home was built.")
    garage_spaces: int = Field(description="Number of garage parking spaces.")
    lot_size_acres: float = Field(description="Lot size in acres.")
    hoa_monthly_usd: float = Field(description="Monthly HOA dues in USD.")


class _FrozenUuidModule:
    """A stand-in for the ``uuid`` module that always mints the same id.

    Installed ONLY as ``marvin.tasks.task.uuid`` so the framework's per-Task
    random id is reproducible; every other ``uuid`` user in the process (notably
    the SDK's own span-id minting) keeps the real module.
    """

    def __init__(self, hex_value: str) -> None:
        self._hex = hex_value

    def uuid4(self):  # noqa: D401 - mirrors uuid.uuid4()
        class _Frozen:
            hex = self._hex

        return _Frozen()

    def __getattr__(self, item):
        return getattr(_real_uuid, item)


@pytest.fixture
def pin_task_id(monkeypatch):
    """Freeze Marvin's random ``Task.id`` so the recorded end-turn tool name matches."""

    def _pin(task_id: str) -> None:
        import marvin.tasks.task as _task_mod

        monkeypatch.setattr(_task_mod, "uuid", _FrozenUuidModule(task_id + "0" * 24))

    return _pin


def _agent(fixture) -> "marvin.Agent":
    """A REAL ``marvin.Agent`` whose pydantic-ai OpenAI model is bound to the
    recorded body through the provider's documented ``http_client=`` seam — the
    same seam ``samples/data/generators/marvin.py`` recorded through."""
    transport, _ = mock_transport(fixture)
    provider = OpenAIProvider(api_key="test-key", http_client=httpx.AsyncClient(transport=transport))
    model = OpenAIChatModel("gpt-4o-mini", provider=provider)
    return marvin.Agent(name=_AGENT_NAME, model=model)


class TestMarvinRecorded:
    def test_cast_over_recorded_openai_stream(self, mock_client, pin_task_id):
        """``marvin.cast`` -> a typed MLS record, driven by the real recorded stream."""
        fixture = load_recorded("marvin", "listing_cast")
        pin_task_id(_CAST_TASK_ID)
        uploaded = capture_framework_trace(mock_client)

        adapter = MarvinAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=marvin)
        try:
            record = marvin.cast(
                _LISTING_DESCRIPTION,
                target=PropertyListing,
                instructions="Extract the structured MLS record for this property.",
                agent=_agent(fixture),
            )
        finally:
            adapter.disconnect()

        # --- Marvin really deserialized the recorded stream into a typed object.
        assert isinstance(record, PropertyListing)
        assert record.street_address == "1428 Oakridge Lane"
        assert record.bedrooms == 4
        assert record.bathrooms == 2.5
        assert record.square_feet == 2340
        assert record.list_price_usd == 749000

        events = uploaded["events"]

        # --- tool.call: the primitive, the developer-declared agent, the resolved
        #     target type, and the REAL parsed output.
        tc = find_event(events, "tool.call")
        assert tc["payload"]["framework"] == "marvin"
        assert tc["payload"]["tool_name"] == "marvin.cast"
        assert tc["payload"]["primitive"] == "cast"
        assert tc["payload"]["agent_name"] == _AGENT_NAME
        assert tc["payload"]["response_model"] == "PropertyListing"
        assert tc["payload"]["success"] is True
        # Only a real deserialization of the recorded body puts these here.
        assert "1428 Oakridge Lane" in tc["payload"]["output"]
        assert "749000" in tc["payload"]["output"]

        # --- model.invoke: the model the DEVELOPER configured on the Agent, not
        #     the dated id the recorded response echoes (the adapter resolves off
        #     Agent.model and never reads a response body).
        mi = find_event(events, "model.invoke")
        assert mi["payload"]["framework"] == "marvin"
        assert mi["payload"]["model"] == "gpt-4o-mini"
        assert mi["payload"]["model_name"] == "gpt-4o-mini"
        assert mi["payload"]["agent_name"] == _AGENT_NAME
        assert mi["payload"]["response_model"] == "PropertyListing"
        assert "1428 Oakridge Lane" in mi["payload"]["response"]
        # ...and the response body demonstrably carries the OTHER model id, so the
        # assertion above is a real boundary, not a coincidence.
        raw = fixture["interactions"][0]["response"]["text"]
        assert "gpt-4o-mini-2024-07-18" in raw

        # --- THE HONEST OMISSION, proven against a body that HAS the tokens:
        #     the recorded stream ends with a real usage block, but Marvin exposes
        #     no usage on its primitives, so the adapter reports no tokens and
        #     emits NO cost.record rather than fabricating one.
        assert '"total_tokens":1366' in raw
        assert '"prompt_tokens":1275' in raw
        for key in ("tokens_prompt", "tokens_completion", "tokens_total", "cost_usd"):
            assert key not in mi["payload"], (
                "marvin surfaces no usage on its primitives — %s must NOT be invented" % key
            )
        assert find_events(events, "cost.record") == []

        # --- the trace still names its agent honestly (the collector's flush-time
        #     resolver picks the developer-declared Agent name).
        ident = find_event(events, "agent.identity")
        assert ident["payload"]["agent_name"] == _AGENT_NAME

    def test_extract_over_recorded_openai_stream(self, mock_client, pin_task_id):
        """``marvin.extract`` -> the real feature list, driven by the recorded stream."""
        fixture = load_recorded("marvin", "features_extract")
        pin_task_id(_EXTRACT_TASK_ID)
        uploaded = capture_framework_trace(mock_client)

        adapter = MarvinAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=marvin)
        try:
            features = marvin.extract(
                _LISTING_DESCRIPTION,
                target=str,
                instructions="Each distinct, marketable feature or amenity of the property.",
                agent=_agent(fixture),
            )
        finally:
            adapter.disconnect()

        # Marvin really parsed the recorded stream's tool-call arguments into a
        # list of real feature strings.
        assert isinstance(features, list)
        assert features and all(isinstance(f, str) for f in features)
        assert any("fireplace" in f.lower() for f in features)

        events = uploaded["events"]

        tc = find_event(events, "tool.call")
        assert tc["payload"]["tool_name"] == "marvin.extract"
        assert tc["payload"]["primitive"] == "extract"
        assert tc["payload"]["agent_name"] == _AGENT_NAME
        # extract()'s target is the ELEMENT type — reported as such, and never as
        # a label set (that key belongs to classify()).
        assert tc["payload"]["response_model"] == "str"
        assert "labels" not in tc["payload"]

        mi = find_event(events, "model.invoke")
        assert mi["payload"]["framework"] == "marvin"
        assert mi["payload"]["model"] == "gpt-4o-mini"
        assert mi["payload"]["primitive"] == "extract"
        assert "fireplace" in mi["payload"]["response"].lower()

        # Same honest omission on the extract path: a real usage block in the
        # body, no invented tokens/cost at marvin's layer.
        raw = fixture["interactions"][0]["response"]["text"]
        assert '"total_tokens":879' in raw
        assert "tokens_total" not in mi["payload"]
        assert find_events(events, "cost.record") == []

    def test_recorded_fixture_is_a_real_provider_body(self):
        """Provenance guard: these fixtures are real recorded OpenAI stream bodies
        (not a hand-written stub), captured by the sample generator's real run."""
        for scenario, task_id in ((("listing_cast"), _CAST_TASK_ID), ("features_extract", _EXTRACT_TASK_ID)):
            fixture = load_recorded("marvin", scenario)
            assert fixture["transport"] == "http"
            prov = fixture["provenance"]
            assert prov["provider"] == "openai"
            assert prov["sdk_version"].startswith("marvin ")
            assert prov["captured_at"] != "pending-creds", "marvin's corpus is a REAL live capture"
            body = fixture["interactions"][0]["response"]
            assert body["status_code"] == 200
            assert body["headers"]["content-type"].startswith("text/event-stream")
            # A real streamed chat.completion: SSE chunks carrying marvin's own
            # end-turn tool call for THIS run's task id.
            assert "chat.completion.chunk" in body["text"]
            assert ("MarkTaskSuccessful_%s" % task_id) in body["text"]
            # ...and a real usage block (the tokens the adapter honestly omits).
            assert '"total_tokens"' in body["text"]
            # Sanity: the stream really parses as SSE JSON chunks.
            chunks = [
                line[len("data: ") :]
                for line in body["text"].splitlines()
                if line.startswith("data: ") and line.strip() != "data: [DONE]"
            ]
            assert chunks, "recorded stream has no data chunks"
            assert json.loads(chunks[0])["object"] == "chat.completion.chunk"
