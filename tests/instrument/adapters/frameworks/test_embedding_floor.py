"""Offline metadata-only-redaction + params + error + attestation floor for the
embedding + vector_store adapters.

These two paired adapters (``EmbeddingAdapter`` -> ``embedding.create``,
``VectorStoreAdapter`` -> ``retrieval.query``) are **metadata-only and
non-agentic**: they emit provider/model/batch/dimension/token and
provider/top-k/filter/score summaries respectively, and *never* the caller's
input text, documents, or filter values. This floor closes the W2 census ◑/gap
cells that the existing unit/recorded suites do not, so a regression fails in
plain CI with no credentials and no network:

* Redaction   — a SENTINEL planted in the embedding **input** and in a Chroma
                **where-filter** is absent from every serialized event. Because
                these adapters have *no content-capture path at all*, the
                standard "capture_content=True carries the content" vacuity
                control is inverted here: the SENTINEL stays absent even under
                ``CaptureConfig.full()`` — that IS the metadata-only guarantee,
                and it is the thing that regresses if someone ever teaches the
                wrapper to capture input text. Non-vacuity is proven instead by a
                known-present control token (the real model id / provider) that
                the same sweep DOES find.
* Params      — the emitted ``embedding.create`` payload carries only its fixed
                allowlist of metadata keys; an extra call kwarg (``user=``,
                carrying a secret) never leaks into the payload or the trace.
* Error-paths — a REAL ``openai.AuthenticationError`` (401 over a mocked
                transport) propagates through the embedding wrapper *verbatim*
                (never swallowed) and never yields a partial/fake success event;
                the failure is now recorded honestly as an ``agent.error``
                carrying the real ``error_type``/``status`` (and, under content
                capture, the error text). The paired ``VectorStoreAdapter`` still
                propagates a REAL ``chromadb`` query error transparently with no
                ``agent.error`` (that separate adapter's gap is out of scope here).
* Cost        — a REAL priced embedding (``text-embedding-3-small``, 2 tokens)
                now carries a non-None ``cost_usd`` on ``embedding.create`` and
                emits a paired priced ``cost.record`` for the platform rollup.
* Attestation — a real embed->retrieve trace (2 real events + the synthesized
                ``trace.root``) flushes an attestation chain that
                ``verify_chain`` reconstructs and accepts, with a tamper control
                proving the check is not vacuous.

The only mock is the network boundary (``httpx.MockTransport`` for the real
openai embeddings SDK); every openai/chroma object, the in-process Chroma engine,
and the adapters' own parsers are real.

CLOSED (were held source-bug PINGs; now fixed + asserted, user-approved):
  * error cell — the embedding wrapper wraps ``original(...)`` in try/except and
    emits an honest ``agent.error`` (error_type/status, plus the error text under
    content capture) before re-raising — see
    ``TestRealErrorShape.test_embedding_real_openai_error_emits_agent_error`` and
    the default-config sibling. (The ``VectorStoreAdapter`` wrappers still have
    the same latent gap; that is a separate adapter, out of scope for this file.)
  * cost cell — ``embedding.create`` now carries a priced ``cost_usd`` and emits a
    paired priced ``cost.record`` (OpenAI embedding rates added to the provider
    PRICING table) — see ``TestCostFloor.test_embedding_create_carries_cost_usd``.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict

import httpx
import pytest

openai = pytest.importorskip("openai")
chromadb = pytest.importorskip("chromadb")

from layerlens.instrument import trace_context  # noqa: E402
from layerlens.attestation._verify import verify_chain  # noqa: E402
from layerlens.attestation._envelope import HashScope, AttestationEnvelope  # noqa: E402
from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.embedding import EmbeddingAdapter  # noqa: E402
from layerlens.instrument.adapters.frameworks.vector_store import VectorStoreAdapter  # noqa: E402

from .conftest import find_event, find_events, capture_framework_trace  # noqa: E402
from ..._recorded import load_recorded, mock_transport  # noqa: E402

SENTINEL = "LL-SENTINEL-7f3a9c2e"

#: The complete metadata vocabulary the embedding adapter is allowed to emit.
#: The floor asserts the payload is a subset of this set: any key outside it is a
#: leak (the bite), while an occasionally-absent optional key is tolerated.
_EMBEDDING_ALLOWED = {
    "framework",
    "provider",
    "model",
    "batch_size",
    "dimensions",
    "total_tokens",
    "latency_ms",
    # A priced embedding call now carries its own ``cost_usd`` (a computed float,
    # never a secret) alongside emitting a paired ``cost.record`` — see
    # ``TestCostFloor``. Added to the allowlist so the metadata-only bite below
    # still fires on any *other* non-allowlisted key.
    "cost_usd",
}


# ---------------------------------------------------------------------------
# Real-object seams (network is the only mock)
# ---------------------------------------------------------------------------
def _recorded_openai_client() -> Any:
    """A real ``openai.OpenAI`` whose transport serves the committed embeddings
    fixture — the proven seam from test_embedding_recorded.py."""
    transport, _ = mock_transport(load_recorded("openai", "embeddings"))
    return openai.OpenAI(api_key="test-key", http_client=httpx.Client(transport=transport), max_retries=0)


def _error_openai_client(status: int = 401) -> Any:
    """A real ``openai.OpenAI`` whose transport returns a real API error body so
    the real SDK raises its real exception class (e.g. AuthenticationError)."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status,
            json={
                "error": {
                    "message": "Incorrect API key provided: sk-****.",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key",
                }
            },
        )

    return openai.OpenAI(
        api_key="test-key", http_client=httpx.Client(transport=httpx.MockTransport(handler)), max_retries=0
    )


def _chroma_collection() -> Any:
    """A real, offline Chroma collection (three explicit 2-D vectors, L2 index).

    ``ids``/``documents``/``metadatas`` all carry the SENTINEL so a redaction
    regression that started capturing any of them would leak it.
    """
    client = chromadb.EphemeralClient()
    col = client.create_collection(name=f"floor_{uuid.uuid4().hex[:8]}", metadata={"hnsw:space": "l2"})
    col.add(
        ids=[f"id-{SENTINEL}", "b", "c"],
        embeddings=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        documents=[f"secret document {SENTINEL}", "other", "third"],
        metadatas=[{"tag": SENTINEL}, {"tag": "x"}, {"tag": "y"}],
    )
    return col


# ---------------------------------------------------------------------------
# Redaction — metadata-only lock (SENTINEL never leaves the boundary)
# ---------------------------------------------------------------------------
class TestRedactionFloor:
    def test_embedding_input_content_never_captured(self, mock_client):
        """capture_content=False: the input text (SENTINEL) is absent, and the
        structural metadata the adapter DOES emit is present."""
        uploaded = capture_framework_trace(mock_client)
        client = _recorded_openai_client()
        adapter = EmbeddingAdapter(mock_client, capture_config=CaptureConfig(capture_content=False))
        adapter.connect(target=client)
        try:
            with trace_context(mock_client, capture_config=CaptureConfig(capture_content=False)):
                client.embeddings.create(model="text-embedding-3-small", input=f"remember {SENTINEL}")
        finally:
            adapter.disconnect()

        events = uploaded["events"]
        evt = find_event(events, "embedding.create")
        # Structural metadata survives (the event is not empty).
        assert evt["payload"]["provider"] == "openai"
        assert evt["payload"]["model"] == "text-embedding-3-small"
        assert evt["payload"]["dimensions"] == 1536
        assert evt["payload"]["total_tokens"] == 2

        blob = json.dumps(events)
        # Non-vacuity guard: the sweep DOES find a known-present control token,
        # so the SENTINEL's absence below is meaningful (not an empty blob).
        assert "text-embedding-3-small" in blob
        # The privacy lock: the caller's input text never reached the trace.
        assert SENTINEL not in blob, "PRIVACY LEAK: embedding input SENTINEL survived into the trace"

    def test_embedding_input_content_never_captured_even_when_full(self, mock_client):
        """Metadata-only INVARIANT (the inverted vacuity control): these adapters
        have no content-capture path, so even ``CaptureConfig.full()`` must NOT
        carry the input text. This is the assertion that regresses the day the
        wrapper is taught to capture input content."""
        uploaded = capture_framework_trace(mock_client)
        client = _recorded_openai_client()
        adapter = EmbeddingAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=client)
        try:
            with trace_context(mock_client, capture_config=CaptureConfig.full()):
                client.embeddings.create(model="text-embedding-3-small", input=f"remember {SENTINEL}")
        finally:
            adapter.disconnect()

        events = uploaded["events"]
        # The event still exists (real metadata captured)...
        assert find_events(events, "embedding.create")
        blob = json.dumps(events)
        assert "text-embedding-3-small" in blob  # non-vacuity control
        # ...but the input content is absent even under full capture.
        assert SENTINEL not in blob, "PRIVACY LEAK: embedding is not metadata-only under CaptureConfig.full()"

    def test_vector_store_filter_content_never_captured(self, mock_client):
        """A Chroma query whose ids/documents/metadata AND ``where`` filter all
        carry the SENTINEL emits only structural metadata — has_filter is True
        (the boolean survives) but no filter value / document text leaks."""
        uploaded = capture_framework_trace(mock_client)
        col = _chroma_collection()
        adapter = VectorStoreAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=col)
        try:
            with trace_context(mock_client, capture_config=CaptureConfig.full()):
                col.query(query_embeddings=[[1.0, 0.0]], n_results=2, where={"tag": SENTINEL})
        finally:
            adapter.disconnect()

        events = uploaded["events"]
        evt = find_event(events, "retrieval.query")
        assert evt["payload"]["provider"] == "chroma"
        # The presence of a filter is recorded structurally, its content is not.
        assert evt["payload"]["has_filter"] is True

        blob = json.dumps(events)
        assert "chroma" in blob  # non-vacuity control
        assert SENTINEL not in blob, "PRIVACY LEAK: vector-store filter/document SENTINEL survived into the trace"


# ---------------------------------------------------------------------------
# Params — the emitted payload is exactly the metadata allowlist
# ---------------------------------------------------------------------------
class TestParamsAllowlist:
    def test_embedding_payload_is_metadata_allowlist_only(self, mock_client):
        """An extra ``user=`` kwarg (carrying a secret) passed to the real
        embeddings call never leaks into the payload — the payload is a subset of
        the fixed metadata allowlist and the secret value is absent from the
        whole trace."""
        uploaded = capture_framework_trace(mock_client)
        client = _recorded_openai_client()
        adapter = EmbeddingAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=client)
        try:
            with trace_context(mock_client, capture_config=CaptureConfig.full()):
                client.embeddings.create(
                    model="text-embedding-3-small",
                    input="hello world",
                    user=f"tenant-{SENTINEL}",  # valid openai kwarg, must NOT be captured
                )
        finally:
            adapter.disconnect()

        events = uploaded["events"]
        payload: Dict[str, Any] = find_event(events, "embedding.create")["payload"]
        # Positive filter: only allowlisted metadata keys — any extra key is a leak.
        extra = set(payload) - _EMBEDDING_ALLOWED
        assert not extra, f"embedding.create leaked non-allowlisted keys: {extra}"
        assert "user" not in payload
        assert "input" not in payload
        # And the secret carried on the unknown kwarg never reaches the trace.
        assert SENTINEL not in json.dumps(events), "params path leaked the SENTINEL kwarg value"


# ---------------------------------------------------------------------------
# Error-paths — real SDK exceptions propagate verbatim, zero telemetry
# ---------------------------------------------------------------------------
class TestRealErrorShape:
    def test_embedding_real_openai_error_propagates_no_partial_event(self, mock_client):
        """A REAL openai 401 error raised by the real SDK propagates through the
        wrapper unchanged, and the failed call emits NO partial/fake
        ``embedding.create`` success event. Bite: a wrapper that swallowed the
        exception or emitted a misleading success event fails here.

        Under the DEFAULT config (``capture_content=False``) the wrapper still
        records the failure as an ``agent.error``, but only the surviving CATEGORY
        (``error_type``/``status``) is present — the free-text ``error`` is stripped
        by the redaction backstop. The full-content variant that asserts the error
        text survives is the sibling ``test_embedding_real_openai_error_emits_agent_error``."""
        uploaded = capture_framework_trace(mock_client)
        client = _error_openai_client(status=401)
        adapter = EmbeddingAdapter(mock_client)
        adapter.connect(target=client)
        try:
            with trace_context(mock_client):
                with pytest.raises(openai.AuthenticationError) as excinfo:
                    client.embeddings.create(model="text-embedding-3-small", input="hi")
        finally:
            adapter.disconnect()

        # The real SDK class + its real message flow through verbatim.
        assert type(excinfo.value).__name__ == "AuthenticationError"
        assert isinstance(excinfo.value, openai.OpenAIError)
        assert "401" in str(excinfo.value)

        events = uploaded["events"]
        assert not find_events(events, "embedding.create"), "a failed embed must not emit a partial event"
        # The failure is recorded honestly as an agent.error carrying the surviving
        # category, but NOT a fake success — and the free-text error is redacted
        # away under capture_content=False.
        err = find_event(events, "agent.error")
        assert err["payload"]["error_type"] == "AuthenticationError"
        assert err["payload"]["status"] == "error"
        assert "error" not in err["payload"], "free-text error must be stripped under capture_content=False"

    def test_embedding_real_openai_error_emits_agent_error(self, mock_client):
        """W1-parity: a REAL openai 401 raised by the real SDK propagates verbatim
        AND the wrapper emits an honest ``agent.error`` carrying the real
        ``error_type``/``status`` and — under content capture — the error text
        (with the 401), so a failed embed is not silently lost. Bite: reverting
        the wrapper's try/except (no emit) drops the agent.error and this fails on
        the empty trace.

        Runs under ``CaptureConfig.full()`` so the free-text ``error`` survives the
        content-redaction backstop (which strips ``error`` under
        ``capture_content=False`` — see the sibling default-config test below,
        where only the category survives)."""
        uploaded = capture_framework_trace(mock_client)
        client = _error_openai_client(status=401)
        adapter = EmbeddingAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect(target=client)
        try:
            with trace_context(mock_client, capture_config=CaptureConfig.full()):
                with pytest.raises(openai.AuthenticationError):
                    client.embeddings.create(model="text-embedding-3-small", input="hi")
        finally:
            adapter.disconnect()

        # No partial success event survives a failure.
        assert not find_events(uploaded["events"], "embedding.create")
        err = find_event(uploaded["events"], "agent.error")
        assert err["payload"]["error_type"] == "AuthenticationError"
        assert "401" in err["payload"]["error"]
        assert err["payload"]["status"] == "error"

    def test_vector_store_real_error_propagates_no_partial_event(self, mock_client):
        """A REAL chromadb query error (dimension mismatch) propagates through the
        wrapper unchanged and emits NO ``retrieval.query`` event."""
        uploaded = capture_framework_trace(mock_client)
        client = chromadb.EphemeralClient()
        col = client.create_collection(name=f"floor_err_{uuid.uuid4().hex[:8]}", metadata={"hnsw:space": "l2"})
        col.add(ids=["a"], embeddings=[[1.0, 0.0]])
        adapter = VectorStoreAdapter(mock_client)
        adapter.connect(target=col)
        try:
            with trace_context(mock_client):
                # 4-D query against a 2-D index -> real chromadb.errors error.
                with pytest.raises(Exception) as excinfo:
                    col.query(query_embeddings=[[1.0, 0.0, 0.0, 0.0]], n_results=1)
        finally:
            adapter.disconnect()

        # A genuine chromadb SDK error, not a hand-rolled stand-in.
        assert type(excinfo.value).__module__.startswith("chromadb")

        events = uploaded["events"]
        assert not find_events(events, "retrieval.query"), "a failed query must not emit a partial event"
        assert not find_events(events, "agent.error")


# ---------------------------------------------------------------------------
# Cost — a real priced embedding carries cost_usd + a paired priced cost.record
# ---------------------------------------------------------------------------
class TestCostFloor:
    def test_embedding_create_carries_cost_usd(self, mock_client):
        """A real ``text-embedding-3-small`` call (recorded usage: 2 prompt
        tokens) must be priced: ``embedding.create`` carries a non-None
        ``cost_usd`` and a paired ``cost.record`` (also priced) is emitted for the
        platform cost rollup. Bite: with embedding pricing removed / the record
        not emitted, cost_usd is absent (None) and this fails."""
        uploaded = capture_framework_trace(mock_client)
        client = _recorded_openai_client()
        adapter = EmbeddingAdapter(mock_client)
        adapter.connect(target=client)
        try:
            with trace_context(mock_client):
                client.embeddings.create(model="text-embedding-3-small", input="hello world")
        finally:
            adapter.disconnect()

        events = uploaded["events"]
        evt = find_event(events, "embedding.create")
        assert evt["payload"]["total_tokens"] == 2
        assert evt["payload"].get("cost_usd") is not None
        assert evt["payload"]["cost_usd"] > 0

        # The paired cost.record the platform rollup sums is present and priced.
        cost = find_event(events, "cost.record")
        assert cost["payload"]["model"] == "text-embedding-3-small"
        assert cost["payload"].get("cost_usd") is not None
        assert cost["payload"]["cost_usd"] == evt["payload"]["cost_usd"]


# ---------------------------------------------------------------------------
# Attestation — offline chain verification over a real embed->retrieve trace
# ---------------------------------------------------------------------------
class TestAttestationOffline:
    def test_attestation_chain_verifies_over_embed_retrieve(self, mock_client):
        uploaded = capture_framework_trace(mock_client)
        oclient = _recorded_openai_client()
        col = _chroma_collection()

        embed_adapter = EmbeddingAdapter(mock_client)
        embed_adapter.connect(target=oclient)
        store_adapter = VectorStoreAdapter(mock_client)
        store_adapter.connect(target=col)
        try:
            with trace_context(mock_client):
                oclient.embeddings.create(model="text-embedding-3-small", input="hello world")
                col.query(query_embeddings=[[1.0, 0.0]], n_results=2)
        finally:
            embed_adapter.disconnect()
            store_adapter.disconnect()

        events = uploaded["events"]
        # The real embed + retrieve both landed (plus the synthesized trace.root).
        assert find_events(events, "embedding.create")
        assert find_events(events, "retrieval.query")

        chain = (uploaded["attestation"] or {}).get("chain") or {}
        raw = chain.get("events") or []
        envelopes = [
            AttestationEnvelope(hash=e["hash"], scope=HashScope(e["scope"]), previous_hash=e.get("previous_hash"))
            for e in raw
        ]
        assert envelopes, "no attestation envelopes captured for the embed->retrieve trace"
        assert len(envelopes) == len(events), (
            f"attestation chain has {len(envelopes)} envelopes for {len(events)} events"
        )
        assert (uploaded["attestation"] or {}).get("root_hash") is not None

        result = verify_chain(envelopes)
        assert result.valid, f"attestation chain invalid: {result.error}"

        # Vacuity control: verify_chain must REJECT a broken interior link.
        assert len(envelopes) >= 2
        tampered = list(envelopes)
        tampered[1] = AttestationEnvelope(
            hash=tampered[1].hash,
            scope=tampered[1].scope,
            previous_hash="sha256:deadbeef-not-the-prior-hash",
        )
        broken = verify_chain(tampered)
        assert not broken.valid and broken.break_index == 1, "verify_chain failed to detect a broken link"
