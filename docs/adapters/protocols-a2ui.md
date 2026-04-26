# A2UI (Agent-to-User Interface) protocol adapter

`layerlens.instrument.adapters.protocols.a2ui.A2UIAdapter` instruments
the A2UI protocol — surface lifecycle and user-action events for
agent-driven UI experiences (checkout widgets, confirmation dialogs,
inline agent panels).

## Install

```bash
pip install 'layerlens[protocols-a2ui]'
```

The `protocols-a2ui` extra has no required dependencies; the adapter
operates on protocol payloads, not on a specific SDK.

## Quick start

```python
from layerlens.instrument.adapters.protocols.a2ui import A2UIAdapter
from layerlens.instrument.transport.sink_http import HttpEventSink

sink = HttpEventSink(adapter_name="a2ui")
adapter = A2UIAdapter()
adapter.add_sink(sink)
adapter.connect()

adapter.on_surface_created(
    surface_id="surf-checkout-1",
    org_id="org-123",
    root_component_id="cmp-checkout-root",
    component_count=12,
)

adapter.on_user_action(
    surface_id="surf-checkout-1",
    action_name="confirm_purchase",
    org_id="org-123",
    component_id="cmp-confirm-btn",
    context={"cart_total": 49.99, "currency": "USD"},
)

adapter.disconnect()
sink.close()
```

## What's wrapped

`A2UIAdapter` exposes two primary hooks the host UI runtime calls:

- `on_surface_created(surface_id, org_id, root_component_id, component_count)`
  — emits `commerce.ui.surface_created` and registers the surface
  in-process for action correlation.
- `on_user_action(surface_id, action_name, org_id, component_id, context)`
  — emits `commerce.ui.user_action`. The `context` dict is **always**
  sha256-hashed before emission; cleartext context never leaves the host.

## Events emitted

| Event | Layer | When |
|---|---|---|
| `commerce.ui.surface_created` | L7c | Per `on_surface_created`. |
| `commerce.ui.user_action` | L7c | Per `on_user_action`. |

Like AP2 events, `commerce.ui.*` events bypass `CaptureConfig` gating
via `ALWAYS_ENABLED_EVENT_TYPES` — these are audit-critical UI events.

## A2UI specifics

- **PII safety by construction**: `context` is hashed before emission.
  This is a hard guarantee — the cleartext value is never stored on the
  event, never logged at INFO level, and never written to disk. Only the
  hash is suitable for cross-event correlation; for content inspection,
  use the host's own logging at debug level.
- **Component-level granularity**: `component_id` is optional but
  recommended for analytics — surfaces can drill into per-component
  conversion funnels.
- **Surface tree summary**: only `component_count` and `root_component_id`
  are captured at surface creation. Full tree introspection is left to
  the host application.

## Capture config

`commerce.*` events are always captured regardless of `CaptureConfig`
flags.

## BYOK

Not applicable — A2UI is transport-only at the protocol layer.
