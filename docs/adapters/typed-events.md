# Typed Events Migration Guide

The LayerLens instrument layer adopts a single canonical event payload
schema for every framework, protocol, and provider adapter. This guide
explains the dual-path emission contract introduced by the
`feat/instrument-typed-events-foundation` PR and the rules each
adapter author must follow.

## TL;DR

- Use `BaseAdapter.emit_event(typed_payload)` with a Pydantic model
  from `layerlens.instrument._compat.events` (e.g. `ToolCallEvent`,
  `ModelInvokeEvent`).
- The legacy `BaseAdapter.emit_dict_event(event_type, dict)` path
  emits a `DeprecationWarning` on every call. Adapter authors must
  migrate every emit site before the warning is promoted to an error
  in a future release.
- The typed path validates payloads through `validate_typed_event`
  and **rejects** malformed inputs by raising
  `TypedEventValidationError`. Schema validation is non-negotiable —
  there is no `errors="ignore"` mode.

## Why typed events

The instrument layer is the wire boundary between customer agent code
and the LayerLens platform. Every adapter emission becomes a row in
the trace store, a span in OpenTelemetry, and a record in the
attestation chain. The canonical schema (vendored from
`ateam/stratix/core/events`) is the contract those downstream systems
rely on.

The previous `emit_dict_event` path let each adapter ship whatever
dict shape it found convenient. That was workable for the first wave
of framework ports but produced four problems:

1. **Schema drift.** No two adapters serialised `tool.call` the same
   way. Atlas-app trace search had to special-case every framework's
   field names.
2. **Silent corruption.** A typo in a payload key shipped fine — the
   bad event landed in production unnoticed until somebody tried to
   query it.
3. **No validation surface.** Pydantic models attached to incoming
   dict events would have caught most field-shape bugs at the
   adapter boundary instead of three systems downstream.
4. **No discoverability.** Adapter authors had to read other
   adapters' source to figure out what to put in a payload. Typed
   models surface the contract in IDE autocomplete.

## The dual-path emission contract

`BaseAdapter` exposes two emission methods:

| Method | Use when | Validation | Warning |
|---|---|---|---|
| `emit_event(payload)` | **Always.** | Strict — rejects malformed | None |
| `emit_dict_event(event_type, payload)` | Legacy callers being migrated | None (forwards as-is) | `DeprecationWarning` |

### `emit_event` — preferred path

```python
from layerlens.instrument._compat.events import (
    ToolCallEvent,
    IntegrationType,
)

self.emit_event(
    ToolCallEvent.create(
        name="search",
        version="1.0",
        integration=IntegrationType.LIBRARY,
        input_data={"query": "what's the weather"},
        output_data={"result": "sunny, 72F"},
        latency_ms=412.0,
    )
)
```

The base adapter's emission pipeline:

1. Runs the circuit-breaker check (drops events while open).
2. Runs the `CaptureConfig` filter (drops disabled layers).
3. Runs `validate_typed_event(event_type, payload)`. Invalid payloads
   raise `TypedEventValidationError` and increment the adapter's
   error counter.
4. Stamps the bound `org_id` onto the payload (multi-tenancy).
5. Calls `self._stratix.emit(payload, privacy_level)`.
6. On success, records the emission in the replay buffer and
   dispatches to every attached `EventSink`.

### `emit_dict_event` — legacy path

```python
import warnings

# Existing call site that has not yet been migrated:
self.emit_dict_event("tool.call", {
    "framework": "my_framework",
    "tool_name": "search",
    "tool_input": {"query": "hi"},
})
```

This path is **not** removed — until every adapter migrates, the dict
shape is what their customer apps and tests assert against. The path
emits a `DeprecationWarning` on every call so the gap stays visible
in CI logs and adapter test output. It does NOT run schema validation
because the existing dict shapes (`framework`, `tool_name`,
`tool_input`) intentionally diverge from the canonical
`tool: {name, version, integration}` shape — running the validator
would reject 100% of unmigrated emissions.

The `org_id` stamp still runs on this path, so multi-tenant scoping
is preserved during the transition.

## Per-adapter `extra="allow"` decision

Each adapter declares one boolean class attribute that controls how
the validator treats unknown event types:

```python
class MyAdapter(BaseAdapter):
    # Adapter targets the canonical 13-event taxonomy. Unknown event
    # types are rejected at emission time.
    ALLOW_UNREGISTERED_EVENTS: bool = False
```

```python
class LangfuseImporter(BaseAdapter):
    # Adapter ingests third-party trace shapes whose taxonomy
    # diverges from the canonical schema. Unknown event types are
    # wrapped in an open Pydantic model and forwarded.
    ALLOW_UNREGISTERED_EVENTS: bool = True
```

The default is `False` (strict). Setting `True` is the documented
escape hatch for adapters whose source data genuinely cannot be
mapped onto the canonical 13 event types — typically importer
adapters (langfuse, benchmark_import) and custom-event-emitting
runtimes.

## Migrating an adapter

### Checklist

- [ ] Replace every `self.emit_dict_event(event_type, payload_dict)`
      call with `self.emit_event(TypedModel.create(...))`.
- [ ] Move adapter-specific provenance fields (e.g. `framework`,
      `agent_name`, `timestamp_ns`) into the typed model's
      `metadata` / `attributes` / `parameters` slot — whichever the
      canonical model exposes.
- [ ] If the adapter emits adapter-specific event types (e.g.
      `langfuse.observation`), set
      `ALLOW_UNREGISTERED_EVENTS = True` on the adapter class and
      document why in the class docstring.
- [ ] Update the adapter's test file to assert against the canonical
      payload shape (e.g. `payload["tool"]["name"]` instead of
      `payload["tool_name"]`).
- [ ] Add a `test_<adapter>_emits_typed_payloads_only` test that
      asserts every emit site uses `emit_event` (no
      `emit_dict_event` call sites remaining).
- [ ] Add a `test_<adapter>_emit_does_not_warn_after_migration`
      test that fails if any call site triggers the
      `DeprecationWarning`.
- [ ] Verify with `grep emit_dict_event src/.../<adapter>/` that
      zero call sites remain.

### Worked example: `agno`

The `agno` adapter is the reference migration shipped in PR
`feat/instrument-typed-events-foundation`. Before:

```python
self.emit_dict_event("tool.call", {
    "framework": "agno",
    "tool_name": tool_name,
    "tool_input": self._safe_serialize(tool_input),
    "tool_output": self._safe_serialize(tool_output),
})
```

After:

```python
self.emit_event(
    ToolCallEvent.create(
        name=tool_name,
        version="unavailable",  # agno does not expose tool versions
        integration=IntegrationType.LIBRARY,
        input_data=input_data,
        output_data=output_data,
        latency_ms=latency_ms,
    )
)
```

Test assertions move from
`evt["payload"]["tool_name"]` to
`evt["payload"]["tool"]["name"]`. Adapter-specific fields like
`framework="agno"` move from the top-level dict to the typed model's
`metadata` slot or are dropped if the canonical schema does not
expose an equivalent.

## Cross-cutting requirements

### sha256 hashes are non-optional

`AgentHandoffEvent` and `AgentStateChangeEvent` carry sha256 hashes.
The previous adapter code emitted `None` or partial hex strings; the
canonical models reject both. Use the `_sha256_of(value)` helper
pattern (see `agno/lifecycle.py`) — it produces a `sha256:<hex64>`
string from any string input, including the empty string.

### Cross-cutting events have no `layer`

The canonical `AgentHandoffEvent`, `CostRecordEvent`, and
`PolicyViolationEvent` payloads do not carry a `layer` field — they
are not bound to a single layer. Tests asserting on `payload["layer"]`
must skip cross-cutting types.

### `org_id` lives on the envelope, not the payload

The canonical event models do not declare `org_id` as a field. The
base adapter re-injects `org_id` into the dict view returned by
`model_dump` so downstream sinks always see the tenant binding, and
the replay buffer carries `org_id` at the envelope level.

## Backlog

See `docs/adapters/typed-events-followups.md` for the per-adapter
migration backlog and the running site count for each unmigrated
adapter.
