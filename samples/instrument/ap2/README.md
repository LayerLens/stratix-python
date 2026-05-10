# AP2 (Agent Payments Protocol) instrumentation sample

End-to-end demo of `AP2Adapter` — walks the three-stage AP2
authorization chain in-memory: **intent mandate** (with org-policy
guardrail evaluation) → **payment mandate signing** → **payment
receipt issuance**. The sample uses a fake JWT signature, a fake
merchant, and a fake payment ID — no real payment processor is
contacted.

## Prerequisites

```bash
pip install 'layerlens[protocols-ap2]'
```

(The `protocols-ap2` extra is empty — no peer library is required.)

Optional environment, only needed if you want the events to actually
ship to a LayerLens deployment:

| Variable                       | Purpose                            |
|--------------------------------|------------------------------------|
| `LAYERLENS_STRATIX_API_KEY`    | Bearer token for atlas-app ingest. |
| `LAYERLENS_STRATIX_BASE_URL`   | atlas-app base URL.                |

## Run

```bash
uv run python -m samples.instrument.ap2.main
```

## Expected output

The sample configures an org policy, then walks one full mandate chain.
Event-type strings come straight from `_commerce.py`:

| Event                                 | Source hook                              |
|---------------------------------------|------------------------------------------|
| `commerce.payment.intent_created`     | `adapter.on_intent_mandate_created(...)` |
| `commerce.payment.intent_validated`   | (emitted after guardrail check passes)   |
| `commerce.payment.mandate_signed`     | `adapter.on_payment_mandate_signed(...)` |
| `commerce.payment.receipt_issued`     | `adapter.on_payment_receipt_issued(...)` |

If the synthetic intent had violated the configured policy
(`max_single_tx=500.0`, `daily_limit=2000.0`,
`allowed_merchants=["merchant-shopify"]`), the sample would print the
violations to stderr and exit with status 1 *before* signing — the
sample is wired to surface guardrail failures.

The sample prints a one-line summary and the sink's `batches_sent`
counter on the way out.

## What this demonstrates

- `AP2Adapter.connect()` brings the adapter to `HEALTHY`.
- `configure_policy` registers a per-org spending guardrail
  (`org_id="org-sample"`, allowed merchant `merchant-shopify`).
- `on_intent_mandate_created` emits both `commerce.payment.intent_created`
  and `commerce.payment.intent_validated`. It returns the list of
  guardrail violations; the sample exits early if non-empty.
- `on_payment_mandate_signed` records the signed mandate
  (`mandate_id="im-1"`, `payment_method="CARD"`).
- `on_payment_receipt_issued` records the terminal receipt
  (`payment_id="pay-1"`, `status="success"`).
- `HttpEventSink` batching with `max_batch=10`, `flush_interval_s=1.0`.
- Clean shutdown via `adapter.disconnect()` + `sink.close()` in a
  `finally` block.

`commerce.*` events bypass `CaptureConfig` gating via
`ALWAYS_ENABLED_EVENT_TYPES` — they are audit-critical and cannot be
suppressed. The full mapping from each hook to the canonical event name
is in [`docs/adapters/protocols-ap2.md`](../../../docs/adapters/protocols-ap2.md).
