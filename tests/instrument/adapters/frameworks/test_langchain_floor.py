"""Offline error + attestation + concurrency-isolation floor for the LangChain
framework adapter.

Closes the W1 census cells previously proven only in gated/live lanes (or via
hand-built doubles), so a regression fails in plain CI with no credentials and
no network — every test drives the REAL ``LangChainCallbackHandler`` over real
``langchain_core`` objects (RunnableLambda / FakeListChatModel chains invoked
through the real callback dispatch, or real ``LLMResult`` / ``Generation``
result types fed through the real callback API). No mock of the adapter's own
emitted output is used anywhere.

* Error-path  — a REAL ``langchain_core.exceptions.OutputParserException`` raised
                inside a real ``RunnableLambda`` (invoked with the handler
                attached) drives the real ``on_chain_error`` lifecycle and
                surfaces as ``agent.error`` carrying the real exception message
                and ``status="error"``. A paired success control proves the
                assertion is not vacuous (no ``agent.error`` when the chain
                succeeds). ``error_type == "OutputParserException"`` is asserted
                too — the langchain adapter now records the real exception class
                on its ``agent.error`` payloads (ADP-W1 BUG-7 fix), matching its
                sibling framework adapters.
* Attestation — a real ``RunnableLambda | FakeListChatModel`` chain trace's
                attestation chain reconstructs from the captured
                ``attestation.chain.events`` and verifies offline
                (``verify_chain(...).valid``), mirroring the live harness
                ``_assert_attestation``. A tamper control (one flipped envelope
                hash) proves the verification is real, not vacuously ``valid``.
* Concurrency — two runs with distinct run scopes must not cross-contaminate:
                one trace per run, distinct ``trace_id``s, each holding only its
                own run's content markers and its own agent scope. Proven three
                ways — sequential, genuinely-concurrent worker threads (Barrier
                synchronized), and deterministically-interleaved asyncio tasks
                (ContextVars are copied per Task). If run state ever moved off
                the per-run ``_current_run`` ContextVar onto an instance scalar,
                the two runs would share one collector and these go RED.
"""

from __future__ import annotations

import json
import asyncio
import threading
from uuid import uuid4
from typing import Any, Dict, List

import pytest

from .conftest import (  # noqa: F401
    find_event,
    find_events,
    record_for_schema_lock,
    capture_framework_trace,
)

pytest.importorskip("langchain_core")

from langchain_core.outputs import LLMResult, Generation  # noqa: E402
from langchain_core.runnables import RunnableLambda  # noqa: E402
from langchain_core.exceptions import OutputParserException  # noqa: E402
from langchain_core.language_models.fake_chat_models import FakeListChatModel  # noqa: E402

from layerlens.models import CreateTracesResponse  # noqa: E402
from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.langchain import (  # noqa: E402
    LangChainCallbackHandler,
)

SENTINEL = "LL-SENTINEL-7f3a9c2e"


# ===========================================================================
# Floor 1 — error path: a REAL framework exception through the real callback
# lifecycle surfaces as agent.error.
# ===========================================================================
class TestErrorPathOffline:
    def test_real_chain_error_emits_agent_error(self, mock_client):
        """A real RunnableLambda that raises a real ``OutputParserException``
        drives ``on_chain_error`` and emits ``agent.error`` honestly.

        Bite: if the adapter's chain-error path stopped emitting agent.error,
        or dropped the real message, this fails.
        """
        uploaded = capture_framework_trace(mock_client)
        handler = LangChainCallbackHandler(mock_client, capture_config=CaptureConfig.full())

        def _boom(_: Any) -> Any:
            # REAL langchain framework exception (not a synthetic RuntimeError),
            # carrying the sentinel so we can prove the real message is surfaced.
            raise OutputParserException(f"could not parse LLM output: {SENTINEL}")

        raised: BaseException | None = None
        try:
            RunnableLambda(_boom).invoke({"q": "hi"}, config={"callbacks": [handler]})
        except OutputParserException as exc:  # the real exception propagates out
            raised = exc

        # The REAL framework exception actually drove the callback lifecycle.
        assert isinstance(raised, OutputParserException)

        error = find_event(uploaded["events"], "agent.error")
        assert error["payload"]["status"] == "error"
        assert error["payload"]["framework"] == "langchain"
        # The real exception's message is surfaced honestly — not a placeholder.
        assert SENTINEL in error["payload"]["error"]
        # on_chain_error records the real exception class (ADP-W1 BUG-7 fix);
        # sibling framework adapters all set error_type=type(error).__name__.
        assert error["payload"]["error_type"] == "OutputParserException"

    def test_successful_chain_emits_no_agent_error(self, mock_client):
        """Vacuity control for the error assertion above: the SAME real chain,
        when it does not raise, emits a captured trace but NO agent.error — so
        the presence of agent.error above is a genuine signal of failure, not
        something the handler always emits."""
        uploaded = capture_framework_trace(mock_client)
        handler = LangChainCallbackHandler(mock_client, capture_config=CaptureConfig.full())

        RunnableLambda(lambda x: {"ok": True}).invoke({"q": "hi"}, config={"callbacks": [handler]})

        assert find_events(uploaded["events"], "agent.input"), "handler was not actually wired in"
        assert not find_events(uploaded["events"], "agent.error"), "a successful chain must not emit agent.error"


# ===========================================================================
# Floor 2 — attestation chain verifies offline over a real chain trace.
# ===========================================================================
class TestAttestationOffline:
    def _run_chain(self, mock_client) -> Dict[str, Any]:
        uploaded = capture_framework_trace(mock_client)
        handler = LangChainCallbackHandler(mock_client, capture_config=CaptureConfig.full())
        chain = RunnableLambda(lambda x: x) | FakeListChatModel(responses=["hello world"])
        result = chain.invoke("hi", config={"callbacks": [handler]})
        # The real chain actually executed end-to-end.
        assert result.content == "hello world"
        return uploaded

    def _envelopes(self, uploaded: Dict[str, Any]):
        from layerlens.attestation._envelope import HashScope, AttestationEnvelope

        raw = (uploaded["attestation"].get("chain") or {}).get("events") or []
        return raw, [
            AttestationEnvelope(
                hash=e["hash"],
                scope=HashScope(e["scope"]),
                previous_hash=e.get("previous_hash"),
            )
            for e in raw
        ]

    def test_attestation_chain_verifies_offline(self, mock_client):
        from layerlens.attestation._verify import verify_chain

        uploaded = self._run_chain(mock_client)
        raw, envelopes = self._envelopes(uploaded)

        assert envelopes, "no attestation envelopes captured"
        # One envelope per emitted event — the whole trace is counter-signed.
        assert len(envelopes) == len(uploaded["events"])
        result = verify_chain(envelopes)
        assert result.valid, f"attestation chain invalid: {result.error}"

    def test_attestation_chain_detects_tamper(self, mock_client):
        """Non-vacuity control: flipping one envelope's hash breaks the chain, so
        the ``valid`` assertion above is a real integrity check — not something
        ``verify_chain`` returns unconditionally."""
        from layerlens.attestation._verify import verify_chain
        from layerlens.attestation._envelope import HashScope, AttestationEnvelope

        uploaded = self._run_chain(mock_client)
        raw, _ = self._envelopes(uploaded)
        assert len(raw) >= 2, "need >=2 envelopes to break the linkage"

        tampered_raw = json.loads(json.dumps(raw))  # deep copy
        idx = len(tampered_raw) // 2
        h = tampered_raw[idx]["hash"]
        tampered_raw[idx]["hash"] = ("1" if h[0] != "1" else "0") + h[1:]
        tampered = [
            AttestationEnvelope(hash=e["hash"], scope=HashScope(e["scope"]), previous_hash=e.get("previous_hash"))
            for e in tampered_raw
        ]
        assert not verify_chain(tampered).valid, "verify_chain failed to detect a tampered hash"


# ===========================================================================
# Floor 3 — concurrent-run isolation (langchain-specific): two runs with
# distinct scopes never cross-contaminate.
# ===========================================================================
def _collect_traces(mock_client: Any) -> List[Dict[str, Any]]:
    """Accumulate each uploaded trace payload separately (one entry per flush)."""
    traces: List[Dict[str, Any]] = []
    lock = threading.Lock()

    def _capture(path: str) -> CreateTracesResponse:
        with open(path) as f:
            data = json.load(f)
        payload = data[0]
        with lock:
            traces.append(payload)
            record_for_schema_lock(payload.get("events", []))
        return CreateTracesResponse(trace_ids=[payload.get("trace_id") or "mock-trace-id"])

    mock_client.traces.upload.side_effect = _capture
    return traces


def _build_chain(marker: str):
    """A real 2-step chain whose input, intermediate value, and output all carry
    the run's marker so each event can be attributed to its run."""
    return RunnableLambda(lambda x, mk=marker: f"{mk}-mid") | FakeListChatModel(responses=[f"{marker}-answer"])


def _drive_real(handler: LangChainCallbackHandler, marker: str) -> None:
    """Fire one full real chain run through the real callback dispatch, with a
    developer-declared run_name (the honest per-run agent scope)."""
    _build_chain(marker).invoke(
        f"{marker}-input",
        config={"callbacks": [handler], "run_name": f"{marker}-agent"},
    )


def _assert_isolated(traces: List[Dict[str, Any]], markers: List[str]) -> None:
    assert len(traces) == len(markers), (
        f"expected {len(markers)} uploaded traces (one per run), got {len(traces)} — "
        "concurrent runs merged or lost a trace"
    )
    trace_ids = {t["trace_id"] for t in traces}
    assert len(trace_ids) == len(markers), f"traces must have distinct trace_ids, got {trace_ids}"

    for marker in markers:
        own = [t for t in traces if marker in json.dumps(t["events"])]
        assert len(own) == 1, f"run marker {marker!r} must appear in exactly 1 trace, found in {len(own)}"
        trace = own[0]
        blob = json.dumps(trace["events"])
        for other in markers:
            if other != marker:
                assert other not in blob, f"trace for run {marker!r} is contaminated with run {other!r} events"
        assert all(e["trace_id"] == trace["trace_id"] for e in trace["events"]), (
            "events within a trace must share its trace_id"
        )
        # The agent scope carried by this trace's events belongs ONLY to this run.
        agent_names = {e["payload"].get("agent_name") for e in trace["events"] if e["payload"].get("agent_name")}
        assert agent_names <= {f"{marker}-agent"}, f"foreign agent scope leaked into run {marker!r}: {agent_names}"


class TestConcurrentRunIsolation:
    def test_sequential_real_runs_are_isolated(self, mock_client):
        """GREEN baseline: back-to-back real chain runs through one handler stay
        separate."""
        traces = _collect_traces(mock_client)
        handler = LangChainCallbackHandler(mock_client, capture_config=CaptureConfig.full())

        _drive_real(handler, "alpha")
        _drive_real(handler, "bravo")

        _assert_isolated(traces, ["alpha", "bravo"])

    def test_threaded_concurrent_real_runs_are_isolated(self, mock_client):
        """Two real chain runs on worker threads through one shared handler,
        released simultaneously by a Barrier for genuine wall-clock overlap.
        Each thread starts with a fresh ContextVar context, so each run must get
        its own collector."""
        traces = _collect_traces(mock_client)
        handler = LangChainCallbackHandler(mock_client, capture_config=CaptureConfig.full())
        barrier = threading.Barrier(2)
        errors: List[BaseException] = []

        def run(marker: str) -> None:
            try:
                barrier.wait(timeout=10)
                _drive_real(handler, marker)
            except BaseException as exc:  # noqa: BLE001 — surface, threads swallow
                errors.append(exc)

        ta = threading.Thread(target=run, args=("alpha",))
        tb = threading.Thread(target=run, args=("bravo",))
        ta.start()
        tb.start()
        ta.join(timeout=30)
        tb.join(timeout=30)

        assert not errors, f"worker thread raised: {errors!r}"
        _assert_isolated(traces, ["alpha", "bravo"])

    def test_asyncio_interleaved_runs_are_isolated(self, mock_client):
        """Two runs as concurrent asyncio tasks, deterministically interleaved by
        ``await asyncio.sleep(0)`` between phases, driven through the real
        callback API with real ``LLMResult`` / ``Generation`` result types.

        ContextVars are copied per ``asyncio.Task``, so the two runs' per-run
        ``_current_run`` must not bleed into one another.
        """
        traces = _collect_traces(mock_client)
        handler = LangChainCallbackHandler(mock_client, capture_config=CaptureConfig.full())

        async def run(marker: str) -> None:
            root = uuid4()
            llm = uuid4()
            # ``name`` kwarg = the developer-declared run_name LangChain delivers.
            handler.on_chain_start(
                {"name": f"{marker}-agent"},
                {"q": f"{marker}-input"},
                run_id=root,
                name=f"{marker}-agent",
            )
            await asyncio.sleep(0)
            handler.on_llm_start({"name": "ChatOpenAI"}, [f"{marker}-prompt"], run_id=llm, parent_run_id=root)
            await asyncio.sleep(0)
            result = LLMResult(
                generations=[[Generation(text=f"{marker}-answer")]],
                llm_output={
                    "model_name": "gpt-4",
                    "token_usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
                },
            )
            handler.on_llm_end(result, run_id=llm)
            await asyncio.sleep(0)
            handler.on_chain_end({"result": f"{marker}-result"}, run_id=root)

        async def main() -> None:
            await asyncio.gather(run("alpha"), run("bravo"))

        asyncio.run(main())

        _assert_isolated(traces, ["alpha", "bravo"])


# ===========================================================================
# Floor 4 — streaming depth: a REAL multi-chunk streamed response drives real
# on_llm_new_token callbacks -> streamed_chunks + ttft (streaming cell).
# ===========================================================================
class TestStreamingFloor:
    # A prompt | model chain so on_chain_start opens the outermost run the
    # adapter flushes on (a bare LLM run never begins a root run). GenericFake
    # ChatModel streams the message token-by-token through the real callback
    # dispatch, firing one real on_llm_new_token per token.
    @staticmethod
    def _chain(handler):
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

        model = GenericFakeChatModel(messages=iter(["hello world foo bar"]))
        return ChatPromptTemplate.from_messages([("user", "{q}")]) | model

    def test_streaming_run_emits_streamed_chunk_count_and_ttft(self, mock_client):
        # Real on_llm_new_token callbacks (langchain.py:318-334) -> on_llm_end
        # surfaces streaming=True, streamed_chunks=tokens_accum, ttft_ms
        # (langchain.py:410-417). The existing single-token test can't catch an
        # off-by-one / hardcoded count; a genuine multi-chunk stream can.
        uploaded = capture_framework_trace(mock_client)
        handler = LangChainCallbackHandler(mock_client, capture_config=CaptureConfig.full())

        chunks = list(
            self._chain(handler).stream({"q": "hi"}, config={"callbacks": [handler], "run_name": "stream-agent"})
        )

        # Vacuity: a genuine multi-token stream (bites a count bound to 1).
        assert len(chunks) > 1, "fake model did not stream multiple chunks — count assertion would be vacuous"

        mi = find_event(uploaded["events"], "model.invoke")
        assert mi["payload"]["streaming"] is True
        # Bind to the REAL observed chunk count, not a literal.
        assert mi["payload"]["streamed_chunks"] == len(chunks), (
            f"streamed_chunks {mi['payload'].get('streamed_chunks')} != observed {len(chunks)} on_llm_new_token calls"
        )
        assert "ttft_ms" in mi["payload"] and mi["payload"]["ttft_ms"] >= 0

    def test_non_streaming_run_has_no_streamed_chunks(self, mock_client):
        # Paired control: the SAME chain invoked (not streamed) fires no token
        # callbacks, so streaming/streamed_chunks/ttft are absent — proving they
        # are a genuine signal of the streaming path, not always-emitted.
        uploaded = capture_framework_trace(mock_client)
        handler = LangChainCallbackHandler(mock_client, capture_config=CaptureConfig.full())

        self._chain(handler).invoke({"q": "hi"}, config={"callbacks": [handler], "run_name": "plain-agent"})

        mi = find_event(uploaded["events"], "model.invoke")
        assert "streamed_chunks" not in mi["payload"], "streamed_chunks leaked onto a non-streaming run"
        assert "streaming" not in mi["payload"], "streaming flag leaked onto a non-streaming run"
        assert "ttft_ms" not in mi["payload"], "ttft_ms leaked onto a non-streaming run"


# ===========================================================================
# Floor 5 — cost honesty: LangChain's own token_usage shapes must never yield a
# fabricated $0.00 (LAY-3622 / A4b, found while auditing the openinference
# adapter and traced back to the shared pricing formula).
# ===========================================================================


class TestCostIsNeverFabricated:
    """A real ``on_llm_end`` whose usage carries ONLY a total must not price at 0.0.

    ``token_usage``/``usage_metadata`` are LangChain's own callback vocabulary and
    a total-only dict is a shape real chat models produce. The shared pricing
    formula reads prompt / cached / cache-write / completion and never the total,
    so it summed four zeroes and answered 0.0; ``0.0 is not None``, so the
    price-on-emit chokepoint treated it as a derived cost and a real billed gpt-4
    call shipped as free. langgraph inherits this path, and
    ``test_langgraph.py::TestInheritedBehavior::test_llm_events_inherited`` pinned
    the fabricated zero as expected behaviour.

    Bite proof: revert the unpriceable-shape gate in ``_cost_from_rates`` and the
    total-only cases go RED with ``cost_usd == 0.0``.
    """

    def _run(self, client: Any, llm_output: Dict[str, Any]) -> List[Dict[str, Any]]:
        uploaded = capture_framework_trace(client)
        handler = LangChainCallbackHandler(client, capture_config=CaptureConfig(capture_content=True))
        chain_id, llm_id = uuid4(), uuid4()
        handler.on_chain_start({"name": "Chain"}, {}, run_id=chain_id)
        handler.on_llm_start({"name": "ChatOpenAI"}, ["prompt"], run_id=llm_id, parent_run_id=chain_id)
        handler.on_llm_end(
            LLMResult(generations=[[Generation(text="output")]], llm_output=llm_output),
            run_id=llm_id,
        )
        handler.on_chain_end({}, run_id=chain_id)
        return [e["payload"] for e in uploaded["events"] if e["event_type"] == "cost.record"]

    @pytest.mark.parametrize("usage_key", ["token_usage", "usage_metadata"])
    def test_a_total_only_usage_withholds_the_cost_instead_of_zeroing_it(
        self, mock_client: Any, usage_key: str
    ) -> None:
        costs = self._run(mock_client, {"model_name": "gpt-4", usage_key: {"total_tokens": 10}})
        assert costs, "the cost.record itself must survive — only the unknowable price is withheld"
        cost = costs[0]
        assert cost.get("cost_usd") != 0.0, (
            f"a real billed 10-token gpt-4 call priced at $0.00 from {usage_key}={{'total_tokens': 10}}"
        )
        assert cost.get("cost_usd") is None
        assert cost["tokens_total"] == 10, "the honest token count must survive"
        assert cost.get("cost_status") == "unpriceable_token_shape", (
            "a priced model with no priceable dimension must say WHY it has no cost"
        )

    def test_a_split_usage_still_prices(self, mock_client: Any) -> None:
        # VACUITY CONTROL: the assertions above must not pass merely because this
        # adapter stopped pricing altogether.
        costs = self._run(
            mock_client,
            {"model_name": "gpt-4", "token_usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}},
        )
        assert costs and costs[0]["cost_usd"] > 0
        assert "cost_status" not in costs[0]
