# A2UI (Agent-to-User Interface) instrumentation sample

End-to-end demo of `A2UIAdapter` — simulates an A2UI surface lifecycle
in-memory: a checkout surface is created, then the user performs two
actions on it (confirm purchase + select payment method). The
PII-sensitive `context` dict on each user action is hashed with sha256
*inside the adapter* before emission — the cleartext cart total never
leaves the calling process.

## Prerequisites

```bash
pip install 'layerlens[protocols-a2ui]'
```

(The `protocols-a2ui` extra is empty — no peer library is required.)

Optional environment, only needed if you want the events to actually
ship to a LayerLens deployment:

| Variable                       | Purpose                            |
|--------------------------------|------------------------------------|
| `LAYERLENS_STRATIX_API_KEY`    | Bearer token for atlas-app ingest. |
| `LAYERLENS_STRATIX_BASE_URL`   | atlas-app base URL.                |

## Run

```bash
uv run python -m samples.instrument.a2ui.main
```

## Expected output

The sample drives three adapter hook calls. Event-type strings come
straight from `_commerce.py`:

| Event                            | Source hook                            |
|----------------------------------|----------------------------------------|
| `commerce.ui.surface_created`    | `adapter.on_surface_created(...)`      |
| `commerce.ui.user_action` (×2)   | `adapter.on_user_action(...)`          |

The sample prints a one-line summary and the sink's `batches_sent`
counter on the way out.

## What this demonstrates

- `A2UIAdapter.connect()` brings the adapter to `HEALTHY` with no peer
  library installed.
- `on_surface_created` records the top-level surface
  (`surface_id="surf-checkout-1"`, 12 components rooted at
  `cmp-checkout-root`).
- Two `on_user_action` calls — `confirm_purchase` carrying
  `{"cart_total": 49.99, "currency": "USD"}` and
  `select_payment_method` carrying `{"method": "card"}`. The adapter
  hashes both context dicts with sha256 before emission, so the
  emitted event carries only the digest.
- `HttpEventSink` batching with `max_batch=10`, `flush_interval_s=1.0`.
- Clean shutdown via `adapter.disconnect()` + `sink.close()` in a
  `finally` block.

`commerce.*` events bypass `CaptureConfig` gating via
`ALWAYS_ENABLED_EVENT_TYPES`. The full hook-to-event mapping is in
[`docs/adapters/protocols-a2ui.md`](../../../docs/adapters/protocols-a2ui.md).
