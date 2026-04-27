# Memory persistence contract for adapters

LayerLens adapters carry per-conversation, per-agent recall — episodic,
procedural, and semantic memory — alongside the trace events they emit.
This module ports the ad-hoc memory plumbing that the four mature
framework adapters (LangChain, AutoGen, CrewAI, Semantic Kernel) carry
on the `ateam` monorepo into a shared, replay-safe primitive that any
adapter on the `stratix-python` SDK can plug into. Without this
plumbing, lighter adapters behave as "goldfish agents" — every run
starts from a blank slate, which is the difference between a usable
production agent and a demo.

This document defines the binding contract every adapter that integrates
the recorder must satisfy. It is enforced at runtime by
`MemoryRecorder.__init__` (fail-fast on missing tenant), by
`MemoryRecorder.restore` (cross-tenant rejection + content-hash
integrity check), and at CI time by
`tests/instrument/adapters/_base/test_memory.py`.

Cross-pollination audit reference:
[`A:/tmp/adapter-cross-pollination-audit.md`](../../../tmp/adapter-cross-pollination-audit.md)
§2 item #1.

## The three buckets

The memory model is the canonical agent-memory split that appears
across the literature (LangChain memory module; CrewAI procedural
memory; AutoGen episodic/semantic split):

| Bucket         | Lifetime              | Bounded by            | Eviction              |
|----------------|-----------------------|------------------------|------------------------|
| **Episodic**   | per-turn              | `max_episodic` (200)   | FIFO (oldest dropped)  |
| **Procedural** | recurring patterns    | `max_procedural` (16)  | least-frequent + ties broken by oldest `last_seen_turn` |
| **Semantic**   | long-lived facts      | `max_semantic` (64)    | least-recently-set     |

* **Episodic** — per-turn `(input, output, error?, tools?, extra?)`
  records, ordered by `turn_index`. The detector for procedural
  patterns reads this stream.
* **Procedural** — derived: each entry is
  `{"pattern": [[prev_turn_tools], [cur_turn_tools]], "count": int,
  "last_seen_turn": int}`. Detected automatically from the recent
  episodic window every time `record_turn` is called.
* **Semantic** — caller-controlled key/value store
  (`set_semantic(key, value)`). Both keys and values are stringified.
  Callers wanting structured semantic data should JSON-encode their
  value.

All three buckets are **bounded** (CLAUDE.md "every cache must be
bounded"). The defaults are conservative; callers wanting a different
size construct the recorder with explicit `max_*` kwargs.

## The contract

1. **Every adapter owns exactly one `MemoryRecorder`.** Constructed in
   `BaseAdapter.__init__` and exposed via `adapter.memory_recorder`
   (read-only property). The recorder is bound to the same `org_id` as
   the adapter — multi-tenancy is propagated.

2. **Construction without a tenant raises.** `MemoryRecorder(org_id="")`
   raises `ValueError`. There is no "default" tenant, no blank
   fallback. `BaseAdapter.__init__` already fails fast on missing
   `org_id` (see [multi-tenancy.md](multi-tenancy.md)) so the recorder
   inherits that guarantee.

3. **Every recorded turn is bounded.** A single oversized turn cannot
   blow past the bucket caps: per-field strings longer than 8 KB are
   truncated to a deterministic suffix (`<...truncated:orig_len=N>`).
   The truncation is defence-in-depth, not a substitute for the
   adapter-level truncation policy (cross-poll #3). Detection of
   recurring tool patterns runs in O(window) per turn.

4. **Cross-tenant restore is prohibited.** `restore(snapshot)` raises
   `ValueError` if `snapshot.org_id != recorder.org_id`. This mirrors
   the `BaseAdapter.org_id` contract — a tenant-A snapshot cannot land
   in a tenant-B recorder, even if both happen to share a process.

5. **Snapshots are tamper-evident.** `restore(snapshot)` recomputes the
   SHA-256 content hash and rejects the snapshot if the recorded hash
   does not match. Guards against accidentally-mutated dicts in
   transit and against forged snapshots reconstructed without the
   `MemorySnapshot` factory.

6. **Snapshots are replay-safe.** The round-trip
   `snapshot() → restore() → snapshot()` produces a snapshot with the
   identical `content_hash` (deterministic reconstruction). This is the
   foundation of replay-safe memory: the replay engine restores the
   recorder, then the adapter re-runs the agent and produces the same
   next-turn snapshot. The `record_turn` method stamps a wall-clock
   `timestamp_ns` into the new turn — replay engines suppress this
   drift by capturing the original `timestamp_ns` from the source
   trace and seeding the recorder's clock at restore time.

7. **Snapshot serialisation is dict-shaped.** `snapshot.to_dict()`
   returns a JSON-serialisable dict; `MemorySnapshot.from_dict(data)`
   round-trips. Adapters embed the dict under
   `ReplayableTrace.metadata["memory_snapshot"]` so the replay engine
   can reconstruct via `MemoryRecorder.restore(MemorySnapshot.from_dict(...))`.

8. **Recording is best-effort.** `BaseAdapter.record_memory_turn`
   catches and logs all exceptions at DEBUG. A failure inside the
   recorder MUST NOT propagate into the host framework's call stack —
   tracing never breaks user code (CLAUDE.md). The trade-off is that a
   recorder bug shows up as missing memory in the replay rather than a
   crash in production.

9. **Thread-safe.** `record_turn`, `set_semantic`, `clear`, and
   `restore` are all guarded by an internal lock. Many concurrent
   `record_turn` calls produce a snapshot whose `episodic` indices
   form an unbroken `1..N` sequence.

## Wiring at the lifecycle hook

Every adapter wires `record_memory_turn(...)` into its **agent-output
boundary** — the point at which the framework reports a completed
agent step / chat turn / invocation. The exact hook varies by
framework:

| Adapter             | Hook                                                                | Episodic input            | Tool list                                                   |
|---------------------|---------------------------------------------------------------------|---------------------------|--------------------------------------------------------------|
| `agno`              | `Agent.run` / `arun` finally-block                                   | `args[0] / kwargs["input"]`| `_collect_tool_names(agent, result)` from `result.messages`  |
| `ms_agent_framework`| `Chat.invoke` / `invoke_stream` finally-block                        | `kwargs["input"]/["message"]`| `_collect_tool_names_from_messages(seen)` from streamed items|
| `openai_agents`     | `_on_agent_span_end` (TraceProcessor) + `on_run_end` (Runner wrap)    | cached at `_on_agent_span_start` per `span_id` | rolled up from `_on_function_span_end` per `parent_id` |
| `llama_index`       | `_on_agent_step_end`                                                  | cached at `_on_agent_step_start` per thread id| rolled up from `_on_tool_call` per thread id |
| `google_adk`        | `after_agent_callback` + `on_agent_end`                                | cached at `before_agent_callback` per thread id| rolled up from `after_tool_callback` per thread id |
| `bedrock_agents`    | `_after_invoke_agent` (boto3 hook)                                    | cached at `_before_invoke_agent` per thread id| rolled up from `_process_trace` action-group / KB step names |

Each adapter also embeds its memory snapshot in `serialize_for_replay`
output via `ReplayableTrace.metadata["memory_snapshot"] =
self.memory_snapshot_dict()` — so a downstream replay engine can
reconstruct the full episodic + procedural + semantic state before
re-execution.

## Honest scope disclosure (target adapter coverage)

The cross-pollination audit §2 item #1 enumerates **seven** target
adapters: `agno`, `ms_agent_framework`, `openai_agents`, `llama_index`,
`google_adk`, `bedrock_agents`, **`browser_use`**.

Six are wired in this PR. The seventh — `browser_use` — does not exist
on this branch's base (`feat/instrument-multitenancy-org-id-propagation`);
it lives on the parallel `feat/instrument-frameworks-browser-use-full`
history. It will be wired when that adapter is ported to this base or
when the histories merge. This follows the same honest-disclosure
pattern as PR #120 (state filters, which omitted `ms_agent_framework`
for the same reason — adapter not on its base).

For `browser_use`, the eventual wiring (per the cross-pollination
audit) will be:

* **Episodic** — page navigation events (`url`, `action`, `selector`)
  per turn.
* **Procedural** — recurring `(prev_action, current_action)` patterns
  (e.g. `"click[search]"` → `"type[query]"` → `"click[submit]"`).
* **Semantic** — long-lived page-content cache keyed by URL or DOM
  hash, so a re-visit can short-circuit page reload during replay.

## Audit hooks

* **Construction failures** — `MemoryRecorder.__init__` raises with a
  message naming the missing field (`"non-empty org_id"`,
  `"bounded buffer sizes"`).
* **Cross-tenant restore** — raises with the explicit
  `"Cross-tenant restore is prohibited (CLAUDE.md multi-tenancy)"`
  message.
* **Tampered snapshots** — raises with
  `"snapshot content_hash mismatch"` and includes the recorded vs
  recomputed hashes for triage.
* **Best-effort recording failures** — logged at DEBUG via
  `BaseAdapter.record_memory_turn` with `exc_info=True` so the failing
  call site is preserved without escalating.

## Replay engine integration

A replay flow looks like:

```python
# Original run captures both events and memory.
adapter = AgnoAdapter(stratix=client, org_id="tenant-A")
# ... agent runs, on_run_end fires record_memory_turn() ...
trace = adapter.serialize_for_replay()
trace.metadata["memory_snapshot"]  # serialised MemorySnapshot dict.

# Replay reconstructs the recorder before re-execution.
replay_adapter = AgnoAdapter(stratix=client, org_id="tenant-A")
snapshot = MemorySnapshot.from_dict(trace.metadata["memory_snapshot"])
replay_adapter.memory_recorder.restore(snapshot)
# Re-run the agent — it sees the original recall state.
```

The next-turn snapshot taken from `replay_adapter` will match the
original (modulo the wall-clock `timestamp_ns` of the new turn — see
contract item 6). This is what makes memory persistence "replay-safe":
the replay engine can drive an adapter through the same agent state
the original run reached.
