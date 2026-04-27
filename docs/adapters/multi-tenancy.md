# Multi-tenancy contract for adapters

LayerLens is a multi-tenant SaaS platform. Every event emitted by an
adapter MUST be tagged with the originating tenant's `org_id`. Cache
keys, queue topics, ingest streams, RLS policies, and downstream
attestation chains all read this field to scope data to a single
tenant.

This document defines the binding contract that every framework /
protocol / provider adapter must satisfy. It is enforced at runtime by
`BaseAdapter.__init__` (fail-fast) and at CI time by the test suite at
`tests/instrument/adapters/_base/test_org_id_propagation.py` plus the
parametrized `tests/instrument/adapters/frameworks/test_per_adapter_org_id.py`.

## The contract

1. **Every adapter is bound to exactly one tenant at construction.**
   The tenant binding (`org_id`) is stored as `self._org_id` and
   exposed as the read-only property `adapter.org_id`. The bound value
   is a non-empty string — there is no null sentinel, no empty
   fallback, no `"default"` placeholder.

2. **Construction without a resolvable `org_id` raises.** Resolution
   order at `__init__`:

   1. Explicit `org_id=...` keyword to the adapter constructor.
   2. `stratix.org_id` attribute on the attached client (if not blank).
   3. `stratix.organization_id` attribute on the attached client — the
      public `layerlens.Stratix` client uses this name (if not blank).

   If none of the three resolve to a non-empty string,
   `BaseAdapter.__init__` raises `ValueError`. This is a fail-fast.
   Callers cannot opt out, suppress, or work around it. There is no
   silent fallback. A blank `org_id` is rejected with the same error
   as an absent one.

3. **Every emission is stamped.** Both `emit_event` (typed payload)
   and `emit_dict_event` (dict payload) call `BaseAdapter._stamp_org_id`
   before forwarding to the client. The bound `self._org_id` is
   written to the payload's `org_id` field unconditionally — any
   caller-supplied value (including a wrong tenant's id) is
   overwritten. The adapter binding is the source of truth.

4. **Every trace record carries `org_id`.** The replay event records
   stored in `self._trace_events` include `org_id` at the envelope
   level *and* inside the payload dict, so replay round-trips and
   downstream re-ingest preserve the binding.

5. **Every sink dispatch carries `org_id`.** The `EventSink.send`
   ABC requires the keyword: `send(event_type, payload, timestamp_ns,
   *, org_id: str)`. Sinks that omit it are flagged at the type-check
   layer (mypy `--strict`). The `IngestionPipelineSink` uses the
   per-event `org_id` as the `tenant_id` for downstream ingest.

## Wiring a new adapter

Subclasses of `BaseAdapter` (and `BaseProtocolAdapter` /
`LLMProviderAdapter`) get the contract for free **as long as their
`__init__` forwards `org_id` to `super().__init__`**. The canonical
shape:

```python
class MyAdapter(BaseAdapter):
    FRAMEWORK = "my_framework"
    VERSION = "0.1.0"

    def __init__(
        self,
        stratix: Any | None = None,
        capture_config: CaptureConfig | None = None,
        # framework-specific args here ...
        *,
        org_id: str | None = None,
    ) -> None:
        super().__init__(
            stratix=stratix,
            capture_config=capture_config,
            org_id=org_id,
        )
        # adapter-specific state ...
```

Note the keyword-only `*` separator for `org_id`. The rest of
`__init__` is unchanged from the pre-multi-tenancy era.

Adapter helper functions (the `instrument_*` convenience exports in
each adapter's `__init__.py`) should also accept and forward `org_id`:

```python
def instrument_agent(
    agent: Any,
    stratix: Any = None,
    capture_config: dict[str, Any] | None = None,
    org_id: str | None = None,
) -> MyAdapter:
    adapter = MyAdapter(
        stratix=stratix,
        capture_config=capture_config,
        org_id=org_id,
    )
    adapter.connect()
    adapter.instrument_agent(agent)
    return adapter
```

## Test obligations

Every new framework adapter must:

1. Have its class added to `_all_adapter_classes()` in
   `tests/instrument/adapters/frameworks/test_per_adapter_org_id.py`.
   The two parametrized tests there assert (a) the adapter accepts
   `org_id` and exposes the bound value via the property, and (b) the
   adapter raises without an `org_id`.
2. If the adapter ships its own dedicated test file, every test that
   constructs the adapter must pass `org_id` (typically via the
   shared `_RecordingStratix` test stand-in, which carries
   `org_id = "test-org"` as a class attribute).
3. The cross-tenant isolation guarantee is covered centrally in
   `tests/instrument/adapters/_base/test_org_id_propagation.py`. New
   adapters do not need to re-prove cross-tenant isolation if they
   route emissions through the standard `BaseAdapter` path; they MUST
   add a per-adapter cross-tenant test if they bypass the base path.

## What changed (April 2026)

Prior to this change, all adapter emissions in the stratix-python SDK
shipped without `org_id` propagation. The 2026-04-25 audit
(`A:/tmp/adapter-depth-audit.md`, cross-cutting finding #3) flagged
this as a CLAUDE.md violation. The fix:

- `BaseAdapter.__init__` now requires a resolvable `org_id` and
  stores it on the instance.
- `emit_event` and `emit_dict_event` stamp `org_id` into every
  payload before forwarding to the client.
- `EventSink.send` now requires the `org_id` keyword.
- Every shipped adapter (17 framework + protocol + provider) was
  updated to thread `org_id` through to `super().__init__`.

## References

- CLAUDE.md, "Multi-Tenancy" section — the platform-wide mandate.
- `A:/tmp/adapter-depth-audit.md` — the audit that surfaced the gap.
- `src/layerlens/instrument/adapters/_base/adapter.py` — `_resolve_org_id`,
  `BaseAdapter.__init__`, `_stamp_org_id`, `emit_event`,
  `emit_dict_event`, `_post_emit_success`.
- `src/layerlens/instrument/adapters/_base/sinks.py` — `EventSink`
  ABC, `TraceStoreSink.send`, `IngestionPipelineSink.send`.
