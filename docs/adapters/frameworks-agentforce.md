# Salesforce Agentforce framework adapter

`layerlens.instrument.adapters.frameworks.agentforce.AgentForceAdapter`
imports Salesforce Agentforce session traces from Data Cloud DMOs and emits
them as LayerLens events.

This adapter is **import-mode** rather than runtime-instrumentation: it
authenticates against a Salesforce org via OAuth 2.0 JWT Bearer and runs
SOQL queries against the AgentForce DMO objects to backfill trace data.

## Install

```bash
pip install 'layerlens[agentforce]'
```

Pulls `requests>=2.28`. The Salesforce credentials must be provisioned
out-of-band (Connected App + private key + permitted user).

## Quick start

```python
from layerlens.instrument.adapters.frameworks.agentforce import (
    AgentForceAdapter,
    SalesforceCredentials,
)
from layerlens.instrument.transport.sink_http import HttpEventSink

credentials = SalesforceCredentials(
    client_id="3MVG9...",
    username="agent-importer@example.com",
    private_key="env:SALESFORCE_PRIVATE_KEY",       # or file path or raw PEM
    instance_url="https://example.my.salesforce.com",
)

sink = HttpEventSink(adapter_name="salesforce_agentforce")
adapter = AgentForceAdapter(credentials=credentials)
adapter.add_sink(sink)
adapter.connect()  # JWT flow runs here

result = adapter.import_sessions(
    start_date="2026-04-01",
    end_date="2026-04-25",
    limit=100,
)
print(f"Imported {result.events_generated} events from {result.sessions_imported} sessions")

adapter.disconnect()
sink.close()
```

## What's wrapped

This adapter does not monkey-patch anything. It calls SOQL against:

- `AIAgentSession` — top-level session record
- `AIAgentSessionParticipant` — agents + users in the session
- `AIAgentInteraction` — turns within the session
- `AIAgentInteractionStep` — individual steps inside an interaction
- `AIAgentInteractionMessage` — raw input/output messages

Each row is normalized via `AgentForceNormalizer` and emitted through the
adapter's `emit_dict_event` pipeline.

Companion modules:

- `AgentApiClient` — direct REST client for Agent API (real-time capture)
- `PlatformEventSubscriber` — gRPC Pub/Sub subscriber for near-real-time
- `TrustLayerImporter` — imports Einstein Trust Layer policies
- `EinsteinEvaluator` — runs LLM evaluation scenarios

## Events emitted

| Event | Layer | When |
|---|---|---|
| `agent.input` | L1 | Per `AIAgentInteractionMessage` with role=user. |
| `agent.output` | L1 | Per `AIAgentInteractionMessage` with role=agent. |
| `agent.action` | L4a | Per `AIAgentInteractionStep`. |
| `tool.call` | L5a | Per step where `StepType` is a tool/action invocation. |
| `model.invoke` | L3 | Per LLM call captured in step metadata. |
| `policy.violation` | cross-cutting | Per Einstein Trust Layer policy hit. |
| `agent.handoff` | L4a | Per `AIAgentSessionParticipant` change. |

Each emitted event includes `_identity` (the Salesforce record `Id`) and
`_timestamp` (record `LastModifiedDate`) for re-import idempotency.

## Salesforce specifics

- **Authentication**: JWT Bearer (OAuth 2.0). `SalesforceCredentials` accepts
  the private key as a raw PEM, an `env:NAME` reference, or a file path.
- **Token lifetime**: ~2 hours. The adapter re-authenticates automatically
  when the cached token expires before the next operation.
- **Rate limits**: a warning is logged when the API daily limit consumption
  passes 80%. Salesforce returns the consumption in response headers.
- **Incremental sync**: pass `last_import_timestamp` to
  `import_sessions(...)` to fetch only records modified since a watermark.
- **Batch size**: configurable via the `batch_size` constructor arg
  (default 200, the SOQL maximum is 2000).

## Capture config

```python
from layerlens.instrument.adapters._base import CaptureConfig

# Recommended for compliance backfills.
adapter = AgentForceAdapter(
    credentials=credentials,
    capture_config=CaptureConfig.standard(),
)

# Strip raw message bodies, keep only structural events.
adapter = AgentForceAdapter(
    credentials=credentials,
    capture_config=CaptureConfig(
        l1_agent_io=True,
        l4a_environment_config=True,
        capture_content=False,
    ),
)
```

## BYOK

Salesforce manages its own model keys (Einstein Trust Layer abstracts the
provider). The adapter does not own model API keys. The Salesforce
credentials themselves are intended to live in atlas-app's `byok_credentials`
table once M1.B ships — see `docs/adapters/byok.md`.

## Replay

`adapter.serialize_for_replay()` returns a `ReplayableTrace` with all events
captured during the current `import_sessions` call. Replay is a re-emit
operation: the adapter does not re-query Salesforce.
