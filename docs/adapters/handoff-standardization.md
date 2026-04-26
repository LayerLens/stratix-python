# Handoff Event Standardisation

Every framework adapter that emits the cross-cutting `agent.handoff`
event must populate a small, stable set of metadata fields. This
contract makes handoffs from any adapter (LangGraph, CrewAI, AutoGen,
agno, openai_agents, llama_index, google_adk, ms_agent_framework,
Semantic Kernel, …) consumable by the same downstream replay,
attestation, and analytics pipelines.

This document describes the contract and the shared helper that
implements it.

## Why

Handoff events without a sequence number are unorderable when
wall-clock timestamps collide (sub-microsecond collisions are common
when handoffs fire back-to-back in async loops). Without context
hashes, the replay engine cannot verify that re-execution received
the same payload that the original run did. Without bounded previews,
event sinks risk receiving multi-megabyte handoff payloads.

The mature adapters (LangChain, LangGraph, CrewAI, AutoGen,
Agentforce, Semantic Kernel) already converged on the same
three-field shape, but the convergence was implementation-by-
implementation rather than via a shared helper. This module
consolidates the pattern.

## The contract

Every `agent.handoff` event emitted by a framework adapter MUST
include these fields in addition to the framework-required
`from_agent` / `to_agent`:

| Field             | Type    | Description                                                                          |
|-------------------|---------|--------------------------------------------------------------------------------------|
| `handoff_seq`     | `int`   | Monotonically increasing seq number, scoped to the adapter instance. Starts at 1.    |
| `context_hash`    | `str`   | `"sha256:" + hex_digest` of canonical-JSON-encoded handoff context.                  |
| `context_preview` | `str`   | Human-readable summary of the context, length-bounded (default 256 chars).           |
| `timestamp`       | `str`   | ISO 8601 UTC timestamp emitted at handoff time.                                      |

Optional but recommended:

| Field        | Type   | Description                                                                |
|--------------|--------|----------------------------------------------------------------------------|
| `framework`  | `str`  | The originating adapter's `FRAMEWORK` identifier (e.g. `"google_adk"`).    |
| `reason`     | `str`  | Framework-specific cause (`"team_delegation"`, `"transfer_to_agent"`, …).  |

## The shared helper

All helpers live in
`src/layerlens/instrument/adapters/_base/handoff.py` and are re-
exported from `layerlens.instrument.adapters._base`.

### `HandoffSequencer`

Thread-safe monotonic counter. **One instance per adapter** —
typically constructed in the adapter's `__init__` and held as
`self._handoff_sequencer`. Concurrent agent invocations (asyncio
gathers, threadpool workers, framework callbacks firing from multiple
OS threads) all draw from the same instance, so the lock is mandatory.

```python
from layerlens.instrument.adapters._base import HandoffSequencer

class MyAdapter(BaseAdapter):
    def __init__(self) -> None:
        super().__init__()
        self._handoff_sequencer = HandoffSequencer()
```

The counter is **1-indexed**: `next()` returns 1 on first call so
downstream consumers can use `handoff_seq > 0` as a "have we
observed any handoffs yet?" predicate without an extra null check.

### `compute_context_hash(state)`

Returns a deterministic SHA-256 digest of the handoff context.
Canonicalises via `json.dumps(state, sort_keys=True, default=str)`
so two semantically-equal contexts always hash identically across
machines and Python versions.

```python
from layerlens.instrument.adapters._base import compute_context_hash

assert compute_context_hash({"a": 1, "b": 2}) == compute_context_hash({"b": 2, "a": 1})
```

`None` and empty dicts both hash to the digest of `"{}"` so the
returned string is never empty.

### `make_preview(content, max_chars=256)`

Returns a length-bounded preview string. Truncates with a single
U+2026 ellipsis so the total length never exceeds `max_chars`.
Coerces non-string values via `str()` and falls back to
`"<unrepresentable>"` if `__str__` raises.

### `build_handoff_payload(...)`

Convenience wrapper that allocates the seq, computes the hash, builds
the preview, and assembles a payload dict ready to pass to
`adapter.emit_dict_event("agent.handoff", payload)`. Use this in
preference to wiring the three primitives by hand.

```python
from layerlens.instrument.adapters._base import build_handoff_payload

def on_handoff(self, src: str, dst: str, ctx: dict) -> None:
    payload = build_handoff_payload(
        sequencer=self._handoff_sequencer,
        from_agent=src,
        to_agent=dst,
        context=ctx,
        extra={"reason": "delegation", "framework": self.FRAMEWORK},
    )
    self.emit_dict_event("agent.handoff", payload)
```

`extra` is merged AFTER the standard fields, but **standard fields
win** — passing `extra={"handoff_seq": 999}` will not override the
real seq. This prevents accidental shadowing of the contract.

## Adapter authoring guide

When you add a new framework adapter that emits `agent.handoff`:

1. **Construct one `HandoffSequencer` per adapter instance** in
   `__init__`. Do NOT use a module-level singleton — concurrent
   adapter instances must have independent sequence numbers.
2. **Route every handoff emit site through `build_handoff_payload`.**
   If your adapter has multiple detection paths (e.g. SDK-callback
   AND a manual `on_handoff` hook), all paths must draw seqs from the
   same sequencer instance.
3. **Include `framework` and `reason` in `extra`.** These are not
   strictly required by the contract but they let dashboards filter
   handoffs by source.
4. **Add tests** that verify the standardised metadata appears. See
   `tests/instrument/adapters/_base/test_handoff.py` for a complete
   helper test suite, and any of the five adapter test files (e.g.
   `test_agno_adapter.py::test_team_delegation_emits_standardized_metadata`)
   for per-adapter integration patterns.

## Adapters following this contract

As of 2026-04-25:

| Adapter            | File:Line                                              | Notes                                          |
|--------------------|--------------------------------------------------------|------------------------------------------------|
| agno               | `frameworks/agno/lifecycle.py`                         | Two emit sites (team delegation + on_handoff). |
| openai_agents      | `frameworks/openai_agents/lifecycle.py`                | Two emit sites (HandoffSpanData + on_handoff). |
| llama_index        | `frameworks/llama_index/lifecycle.py`                  | One emit site (on_handoff).                    |
| google_adk         | `frameworks/google_adk/lifecycle.py`                   | One emit site (transfer_to_agent).             |
| ms_agent_framework | `frameworks/ms_agent_framework/lifecycle.py`           | Two emit sites (group-chat-turn + on_handoff). |

The mature adapters (LangChain, LangGraph, CrewAI, AutoGen, Agentforce,
Semantic Kernel) currently use bespoke implementations of the same
contract — they will migrate to the shared helper in a follow-up PR
once their independent test suites are stable.
