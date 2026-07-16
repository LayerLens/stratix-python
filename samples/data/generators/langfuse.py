"""ADP-W2 Family-B recorder for the ``langfuse`` OBSERVABILITY adapter
(record-real-once, LIVE round-trip).

Records TWO real Langfuse -> LayerLens trace migrations and writes each as a
sealed real-trace fixture under ``samples/data/traces/industry/``:

* ``generate_langfuse_single`` -> ``media_langfuse_moderation.jsonl``: a
  SINGLE-observation migration. A media platform ran a content-moderation
  decision through a real ``gpt-4o-mini`` call; that call's REAL model / token
  usage / decision text is logged to a live Langfuse instance as one
  ``generation`` observation, then imported back through the REAL
  ``LangfuseAdapter``. The migrated trace renders an honest single node (the
  Langfuse trace name ``content-moderation``, no fabricated agent) with a real
  ``model.invoke`` (framework=langfuse) + a real Langfuse-calculated
  ``cost.record``.

* ``generate_langfuse_multi`` -> ``media_langfuse_moderation_pipeline.jsonl``: a
  MULTI-observation migration exercising the langfuse-distinctive SCORE path. A
  content-moderation review pipeline is logged to Langfuse as a ``generation``
  (the moderation LLM call), a ``span`` (the ``policy_lookup`` step), an
  ``event`` (an auto-escalation state change), AND a real LLM-as-judge
  ``score`` (a genuine ``gpt-4o-mini`` judge's numeric ``policy_adherence``
  rating). Imported back, the trace carries the full observation tree:
  ``model.invoke`` + ``cost.record`` (from the generation), ``tool.call`` (from
  the span), ``agent.state.change`` (from the event), and ``evaluation.result``
  (from the score) — the migration path that preserves all grading signal.

WHY LIVE (not mocked): the langfuse adapter is a batch REST-sync pipeline, so a
faithful fixture must exercise the real ``/api/public/ingestion`` ->
``/api/public/traces/{id}`` contract. Both fixtures are produced by POSTing a
real Langfuse ingestion batch to a live Langfuse instance (creds from the live
``.env``), polling until Langfuse has materialized the trace + its observations
(and, for multi, the score), then importing that trace through the REAL
``LangfuseAdapter._import_single_trace`` under the ``_generate_fixtures`` capture
seam (``set_trace_observer`` + a no-op ``enqueue_upload``) so the sealed payload
— real per-observation events + an intact attestation chain — is captured but
never uploaded during generation. The samples upload the captured fixtures
themselves at run time.

NOTHING is fabricated. The moderation ``generation`` carries a REAL
``gpt-4o-mini`` call's model/tokens/output; the ``score`` is a REAL LLM-as-judge
numeric rating; the cost is Langfuse's own calculated figure (or the adapter's
real pricing fallback); the Framework column shows ``langfuse`` (the tool the
trace was migrated from). No ``agent.identity`` is emitted (a migrated
observability trace is single-node / non-agentic), so it renders an honest
single node — never a fabricated multi-agent DAG. HEALTHY (non-ERROR) trace:
the census-flagged Langfuse ``level=ERROR`` fidelity-loss path is a HELD source
bug and is deliberately NOT exercised here.

Recording prerequisites (a missing one is a skip, not a crash, in the
best-effort ``_generate_fixtures.main`` loop):
    LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / (LANGFUSE_HOST or
    LANGFUSE_BASE_URL) for a live Langfuse instance, plus OPENAI_API_KEY for the
    real moderation + judge calls whose genuine output is migrated.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone

# Reuse the record-real-once seam from the sibling ``_generate_fixtures`` module
# (single source of truth for the fixture writer + capture config + model name).
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.dirname(_HERE)        # samples/data
_SAMPLES = os.path.dirname(_DATA)     # samples
_REPO = os.path.dirname(_SAMPLES)
for _p in (os.path.join(_REPO, "src"), _SAMPLES, _DATA):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# This module is named ``langfuse.py`` (to match the adapter). When the file is
# run directly, Python inserts its own directory at ``sys.path[0]``. Drop it so
# any function-local framework import always resolves to the installed package.
sys.path[:] = [_p for _p in sys.path if os.path.abspath(_p or ".") != _HERE]

from layerlens import Stratix  # noqa: E402
import layerlens.instrument._collector as _collector_mod  # noqa: E402
from layerlens.instrument._collector import set_trace_observer  # noqa: E402

import _generate_fixtures as _gf  # noqa: E402

_write = _gf._write
_CAPTURE = _gf._CAPTURE
OPENAI_MODEL = _gf.OPENAI_MODEL


# --------------------------------------------------------------------------
# Live Langfuse connection + real OpenAI helpers
# --------------------------------------------------------------------------
def _connect_adapter(client: Stratix):
    """Construct + connect a REAL ``LangfuseAdapter`` to the live instance."""
    from layerlens.instrument.adapters.frameworks.langfuse import LangfuseAdapter

    host = os.environ.get("LANGFUSE_HOST") or os.environ.get("LANGFUSE_BASE_URL")
    adapter = LangfuseAdapter(client, capture_config=_CAPTURE)
    adapter.connect(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        host=host,
    )
    return adapter


def _iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _openai_client():
    """A fresh, UN-instrumented OpenAI client (the real moderation/judge calls
    are the source data we log to Langfuse — they must not leak into any
    LayerLens collector; this fixture's events come only from the langfuse
    IMPORT path)."""
    from openai import OpenAI

    return OpenAI()


def _real_moderation(oai, post: str) -> dict:
    """Run a REAL ``gpt-4o-mini`` content-moderation decision. Returns the
    genuine model, token usage, and decision text to log into Langfuse."""
    system = (
        "You are a media platform content-moderation assistant. Review the user "
        "post and return a decision of ALLOW, FLAG, or REMOVE with a one-sentence "
        "justification grounded in platform policy. Be concise."
    )
    resp = oai.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": post}],
    )
    u = resp.usage
    return {
        "model": resp.model,
        "system": system,
        "output": resp.choices[0].message.content or "",
        "prompt_tokens": u.prompt_tokens,
        "completion_tokens": u.completion_tokens,
        "total_tokens": u.total_tokens,
    }


def _real_judge(oai, post: str, decision: str) -> dict:
    """Run a REAL ``gpt-4o-mini`` LLM-as-judge scoring the moderation decision's
    policy adherence. Returns a genuine numeric score in [0, 1] + rationale."""
    resp = oai.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an evaluation judge. Rate how well a content-moderation "
                    "decision applies sound platform policy to the post. Reply with a "
                    'JSON object {"score": <float 0..1>, "rationale": "<one sentence>"}.'
                ),
            },
            {
                "role": "user",
                "content": f"POST:\n{post}\n\nMODERATION DECISION:\n{decision}",
            },
        ],
    )
    raw = resp.choices[0].message.content or "{}"
    data = json.loads(raw)
    score = float(data["score"])
    if not 0.0 <= score <= 1.0:  # clamp defensively; never fabricate outside range
        score = max(0.0, min(1.0, score))
    rationale = str(data.get("rationale") or "").strip()
    return {"score": round(score, 4), "rationale": rationale}


# --------------------------------------------------------------------------
# Live ingestion + poll + capture-via-import
# --------------------------------------------------------------------------
def _ingest(adapter, batch: list) -> None:
    """POST a Langfuse ingestion batch and surface any per-item rejections
    (207 with a non-empty ``errors`` array must not silently pass)."""
    resp = adapter._http.post("/api/public/ingestion", json={"batch": batch})
    resp.raise_for_status()
    errors = (resp.json() or {}).get("errors") or []
    if errors:
        raise RuntimeError(f"Langfuse ingestion rejected {len(errors)} item(s): {str(errors)[:400]}")


def _poll_ready(adapter, trace_id: str, since: str, *, min_obs: int, want_score: bool,
                timeout: float = 90.0) -> None:
    """Poll until Langfuse has materialized the trace with >= ``min_obs``
    observations (and, if ``want_score``, its score) so the import round-trips
    the full batch (Langfuse ingestion is async + item-by-item)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        lr = adapter._http.get("/api/public/traces", params={"fromTimestamp": since, "limit": 20})
        data = lr.json().get("data", []) if lr.status_code == 200 else []
        if any(d.get("id") == trace_id for d in data):
            detail = adapter._http.get(f"/api/public/traces/{trace_id}").json()
            obs = detail.get("observations", []) or []
            scores = detail.get("scores", []) or []
            if len(obs) >= min_obs and (not want_score or len(scores) >= 1):
                return
        time.sleep(3)
    raise RuntimeError(
        f"Langfuse trace {trace_id} never became visible with {min_obs} observations"
        + (" + score" if want_score else "")
    )


def _capture_import(client: Stratix, adapter, trace_id: str) -> dict:
    """Import the one materialized Langfuse trace through the REAL adapter code
    path (``_import_single_trace``, the loop body of ``import_traces``) under the
    capture seam, and return the sealed payload the collector flushed."""
    captured: dict = {}
    set_trace_observer(lambda p: captured.setdefault("payload", p))
    orig = _collector_mod.enqueue_upload
    _collector_mod.enqueue_upload = lambda *a, **k: None
    try:
        adapter._import_single_trace({"id": trace_id})
    finally:
        set_trace_observer(None)
        _collector_mod.enqueue_upload = orig
    payload = captured.get("payload")
    if not payload:
        raise RuntimeError("no payload captured for langfuse import")
    return payload


def _stamp(payload: dict, *, tags: list, note: str) -> dict:
    payload["tags"] = list(tags)
    payload["metadata"] = {
        "migrated_from": "langfuse",
        "source": "real Langfuse round-trip import (live self-hosted Langfuse instance)",
        "backend_model": OPENAI_MODEL,
        "note": note,
    }
    return payload


# --------------------------------------------------------------------------
# Scenario content (Media content-moderation; synthetic, non-sensitive posts)
# --------------------------------------------------------------------------
_POST_SINGLE = (
    "Just DM me the word CRYPTO and I'll 10x your money in 48 hours, guaranteed! "
    "Limited spots — first 50 people only, don't miss out. #CryptoKing #GetRichQuick"
)
_POST_MULTI = (
    "URGENT: every batch of NutriStart baby formula is being recalled for containing "
    "toxic heavy metals — throw yours out NOW and share to save lives! The authorities "
    "are covering it up so spread the word before this post gets deleted."
)
_POLICY_LOOKUP_OUT = (
    "POL-HM-004 Dangerous Health / Safety Misinformation: unverified recall or "
    "contamination claims that can cause panic are FLAGGED for review and REMOVED if "
    "no credible source is cited. Encouraging mass-sharing amplifies harm."
)


def generate_langfuse_single(client: Stratix) -> dict:
    """Single-observation Langfuse migration: one moderation ``generation``."""
    oai = _openai_client()
    mod = _real_moderation(oai, _POST_SINGLE)

    adapter = _connect_adapter(client)
    try:
        since = _iso()
        trace_id = uuid.uuid4().hex
        gen_id = uuid.uuid4().hex
        batch = [
            {"id": uuid.uuid4().hex, "type": "trace-create", "timestamp": _iso(), "body": {
                "id": trace_id, "name": "content-moderation",
                "input": _POST_SINGLE, "output": mod["output"],
                "metadata": {"surface": "social_feed", "category": "spam_scam"}}},
            {"id": uuid.uuid4().hex, "type": "generation-create", "timestamp": _iso(), "body": {
                "id": gen_id, "traceId": trace_id, "name": "moderation-llm",
                "model": mod["model"], "input": mod["system"] + "\n\n" + _POST_SINGLE,
                "output": mod["output"],
                "usage": {"promptTokens": mod["prompt_tokens"],
                          "completionTokens": mod["completion_tokens"],
                          "totalTokens": mod["total_tokens"]}}},
        ]
        _ingest(adapter, batch)
        _poll_ready(adapter, trace_id, since, min_obs=1, want_score=False)
        payload = _capture_import(client, adapter, trace_id)
    finally:
        try:
            adapter.disconnect()
        except Exception:
            pass

    _stamp(payload, tags=["layerlens-sample", "industry", "media", "content-moderation",
                          "observability-migration"],
           note="single generation migrated; model/tokens/output are a REAL gpt-4o-mini "
                "moderation call, cost is Langfuse-calculated. Nothing fabricated.")
    events = payload.get("events", [])
    ev = lambda t: [e for e in events if e.get("event_type") == t]
    cr = ev("cost.record")
    cost = (cr[0].get("payload") or {}).get("cost_usd") if cr else None
    print("  langfuse single (content-moderation, single-observation migration)  "
          "events=%d model.invoke=%d cost.record=%d cost_usd=%s"
          % (len(events), len(ev("model.invoke")), len(cr), cost))
    print("  ->", _write([payload], "industry", "media_langfuse_moderation"), "\n")
    return payload


def generate_langfuse_multi(client: Stratix) -> dict:
    """Multi-observation Langfuse migration: generation + span + event + a real
    LLM-as-judge score (exercises the ``evaluation.result`` import path)."""
    oai = _openai_client()
    mod = _real_moderation(oai, _POST_MULTI)
    judged = _real_judge(oai, _POST_MULTI, mod["output"])

    adapter = _connect_adapter(client)
    try:
        since = _iso()
        trace_id = uuid.uuid4().hex
        gen_id = uuid.uuid4().hex
        span_id = uuid.uuid4().hex
        event_id = uuid.uuid4().hex
        batch = [
            {"id": uuid.uuid4().hex, "type": "trace-create", "timestamp": _iso(), "body": {
                "id": trace_id, "name": "content-moderation-review",
                "input": _POST_MULTI, "output": mod["output"],
                "metadata": {"surface": "social_feed", "category": "health_safety_misinformation"}}},
            {"id": uuid.uuid4().hex, "type": "generation-create", "timestamp": _iso(), "body": {
                "id": gen_id, "traceId": trace_id, "name": "moderation-llm",
                "model": mod["model"], "input": mod["system"] + "\n\n" + _POST_MULTI,
                "output": mod["output"],
                "usage": {"promptTokens": mod["prompt_tokens"],
                          "completionTokens": mod["completion_tokens"],
                          "totalTokens": mod["total_tokens"]}}},
            {"id": uuid.uuid4().hex, "type": "span-create", "timestamp": _iso(), "body": {
                "id": span_id, "traceId": trace_id, "name": "policy_lookup",
                "input": "health_safety_misinformation", "output": _POLICY_LOOKUP_OUT}},
            {"id": uuid.uuid4().hex, "type": "event-create", "timestamp": _iso(), "body": {
                "id": event_id, "traceId": trace_id, "name": "escalation",
                "statusMessage": "auto-escalated to human review (misinformation, no credible source)"}},
            {"id": uuid.uuid4().hex, "type": "score-create", "timestamp": _iso(), "body": {
                "id": uuid.uuid4().hex, "traceId": trace_id, "observationId": gen_id,
                "name": "policy_adherence", "value": judged["score"], "dataType": "NUMERIC",
                "source": "EVAL", "comment": judged["rationale"]}},
        ]
        _ingest(adapter, batch)
        _poll_ready(adapter, trace_id, since, min_obs=3, want_score=True)
        payload = _capture_import(client, adapter, trace_id)
    finally:
        try:
            adapter.disconnect()
        except Exception:
            pass

    _stamp(payload, tags=["layerlens-sample", "industry", "media", "content-moderation",
                          "observability-migration", "multi-observation"],
           note="generation + span + event + a REAL gpt-4o-mini LLM-as-judge score migrated; "
                "generation model/tokens/output are a real moderation call, the score is a real "
                "judge rating, cost is Langfuse-calculated. Nothing fabricated.")
    events = payload.get("events", [])
    ev = lambda t: [e for e in events if e.get("event_type") == t]
    evals = ev("evaluation.result")
    score = (evals[0].get("payload") or {}).get("value") if evals else None
    print("  langfuse multi (content-moderation pipeline, multi-observation migration)  "
          "events=%d model.invoke=%d cost.record=%d tool.call=%d state.change=%d evaluation.result=%d score=%s"
          % (len(events), len(ev("model.invoke")), len(ev("cost.record")), len(ev("tool.call")),
             len(ev("agent.state.change")), len(evals), score))
    print("  ->", _write([payload], "industry", "media_langfuse_moderation_pipeline"), "\n")
    return payload


if __name__ == "__main__":  # pragma: no cover - manual regeneration entrypoint
    _client = Stratix()
    generate_langfuse_single(_client)
    generate_langfuse_multi(_client)
