# AP2 (Agent Payments Protocol) adapter

> **Canonical source:** [`protocols/ap2.py`](../../src/layerlens/instrument/adapters/protocols/ap2.py)
> and [`protocols/_commerce.py`](../../src/layerlens/instrument/adapters/protocols/_commerce.py).
> Every event-name string in this doc matches the literal `event_type`
> default at source.

`layerlens.instrument.adapters.protocols.ap2.AP2Adapter` instruments the
[Agent Payments Protocol](https://github.com/google/agent-payments-protocol)
— the three-stage authorization chain for autonomous-agent commerce:

1. **Intent Mandate** — spending guardrails + merchant constraints.
2. **Payment Mandate** — cryptographic authorization to pay.
3. **Payment Receipt** — settlement confirmation.

## Install

```bash
pip install 'layerlens[protocols-ap2]'
```

The `protocols-ap2` extra has no required dependencies; the adapter
operates on protocol payloads, not on a specific SDK.

## Quick start

```python
from layerlens.instrument.adapters.protocols.ap2 import AP2Adapter
from layerlens.instrument.transport.sink_http import HttpEventSink

sink = HttpEventSink(adapter_name="ap2")
adapter = AP2Adapter()
adapter.add_sink(sink)
adapter.connect()

# Configure a per-org policy
adapter.configure_policy(
    org_id="org-123",
    max_single_tx=500.0,
    daily_limit=2000.0,
    allowed_merchants=["merchant-shopify"],
)

# Record the three-stage chain
violations = adapter.on_intent_mandate_created(
    mandate_id="im-1",
    description="Buy 1 shirt under $50",
    org_id="org-123",
    merchants=["merchant-shopify"],
    max_amount=50.0,
    intent_expiry="2026-04-26T00:00:00Z",
    agent_id="agent-shopper",
)
if not violations:
    adapter.on_payment_mandate_signed(
        mandate_id="im-1",
        payment_details_id="pd-1",
        total_amount=49.99,
        merchant_agent="merchant-shopify",
        org_id="org-123",
        signature="<jwt>",
    )
    adapter.on_payment_receipt_issued(
        mandate_id="im-1",
        payment_id="pay-1",
        amount=49.99,
        org_id="org-123",
        merchant_confirmation_id="conf-1",
    )

adapter.disconnect()
sink.close()
```

## What's wrapped

`AP2Adapter` exposes hook methods that the host (an agent or marketplace)
calls at each commerce step:

- `on_intent_mandate_created(...)` — emits intent + validation events,
  evaluates org-level guardrails, returns violation messages.
- `on_payment_mandate_signed(...)` — emits the signed payment mandate.
  Updates cumulative spending and emits spending-threshold events when
  configured limits are exceeded. The raw signature is sha256-hashed
  before storage.
- `on_payment_receipt_issued(...)` — emits the settlement receipt and
  closes out the mandate from the in-memory registry.
- `configure_policy(org_id, ...)` — installs guardrail config for an org.

## Events emitted

| Event | Layer | When |
|---|---|---|
| `commerce.payment.intent_created` | L7a | Per `on_intent_mandate_created` (`_commerce.py:99-101`). |
| `commerce.payment.intent_validated` | L7a | Per intent, after guardrail evaluation (`_commerce.py:130-132`). |
| `commerce.payment.guardrail_violation` | L7a | Per failed guardrail (`_commerce.py:225-227`). |
| `commerce.payment.mandate_signed` | L7a | Per `on_payment_mandate_signed` (`_commerce.py:167-169`). |
| `commerce.payment.threshold_exceeded` | L7a | Per cumulative-spend threshold breach (`_commerce.py:270-272`). |
| `commerce.payment.receipt_issued` | L7a | Per `on_payment_receipt_issued` (`_commerce.py:198-200`). |

All `commerce.*` events bypass the `CaptureConfig` gate via the
`ALWAYS_ENABLED_EVENT_TYPES` rule — these are audit-critical events that
must never be suppressed by capture configuration.

## AP2 specifics

- **Guardrail evaluation**: configured via `configure_policy(...)`. Supported
  rules: `max_single_tx`, `daily_limit`, `weekly_limit`, `monthly_limit`,
  `allowed_merchants` (whitelist), `require_refundability`, plus per-mandate
  `intent_expiry` enforcement at sign time.
- **Signature handling**: payment-mandate signatures are sha256-hashed
  before being stored on the event. The raw signature value is never
  written to the event stream — this preserves auditability without
  retaining sensitive key material.
- **Cumulative spending**: tracked per-org in-process. For long-running
  workers, persist the cumulative state externally (see
  `memory_service=` constructor arg).
- **Expiry enforcement**: if an intent mandate has expired by the time
  `on_payment_mandate_signed` is called, a
  `commerce.payment.guardrail_violation` is emitted before the signed
  event.

## Capture config

`commerce.*` events are always captured regardless of the
`CaptureConfig` flags — this is a deliberate platform invariant.

## BYOK

AP2 cryptographic keys are managed by the host application. The adapter
does not own them; only the sha256 of the signature is captured.
