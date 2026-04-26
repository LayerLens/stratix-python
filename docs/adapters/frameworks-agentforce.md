# Salesforce Agentforce framework adapter

`layerlens.instrument.adapters.frameworks.agentforce.AgentForceAdapter`
imports Salesforce Agentforce session traces from Data Cloud DMOs and
emits them as LayerLens canonical events. The adapter package also ships
companion modules for the Agent API REST surface, the Pub/Sub Platform
Events stream, the Einstein Trust Layer policy importer, and an
LLM evaluator that runs LayerLens graders against captured sessions.

This adapter is **import-mode** rather than runtime monkey-patching: it
authenticates against a Salesforce org via OAuth 2.0 JWT Bearer and runs
SOQL queries against the AgentForce DMO objects to backfill trace data.
Salesforce Agentforce itself is a remote multi-tenant service, not a
Python library, so there is no framework SDK to instrument in-process.

## Install

```bash
pip install 'layerlens[agentforce]'
```

The `[agentforce]` extra pulls `requests>=2.28` (used by the JWT Bearer
flow, the SOQL HTTP transport, the Agent API REST client, and the CometD
Pub/Sub fallback). The Salesforce credentials must be provisioned
out-of-band (Connected App + private key + permitted user — see
[OAuth setup](#oauth-setup) below).

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
print(
    f"Imported {result.events_generated} events "
    f"from {result.sessions_imported} sessions"
)

adapter.disconnect()
sink.close()
```

A fully runnable, mocked end-to-end sample lives in
[`samples/instrument/agentforce/`](../../samples/instrument/agentforce/).

## What's wrapped

This adapter does not monkey-patch anything in process. It calls SOQL
against the following Data Cloud DMO objects:

| DMO object                       | Purpose                                  |
|----------------------------------|------------------------------------------|
| `AIAgentSession`                 | Top-level session record                 |
| `AIAgentSessionParticipant`      | Agents + users in the session            |
| `AIAgentInteraction`             | Turns within a session                   |
| `AIAgentInteractionStep`         | Individual steps inside an interaction   |
| `AIAgentInteractionMessage`      | Raw input / output messages              |

Each row is normalized via `AgentForceNormalizer` and emitted through
the adapter's `emit_dict_event` pipeline (which honors the
`CaptureConfig` filter and circuit-breaker state).

Companion modules in the same package:

| Module                    | What it does                                          |
|---------------------------|-------------------------------------------------------|
| `auth.py`                 | OAuth 2.0 JWT Bearer flow + SOQL HTTP client          |
| `client.py`               | Agent API REST client (real-time session capture)     |
| `events.py`               | Platform Events subscriber (gRPC + CometD fallback)   |
| `mapper.py`               | Agent API session → LayerLens event mapper            |
| `trust_layer.py`          | Einstein Trust Layer policy import / YAML emission    |
| `llm_eval.py`             | `EinsteinEvaluator` — A/B prompt + model comparison   |

## Events emitted

| Event                | Layer  | When                                                   |
|----------------------|--------|--------------------------------------------------------|
| `agent.lifecycle`    | L1     | Per `AIAgentSession` start / end.                      |
| `agent.identity`     | L1     | Per `AIAgentSessionParticipant`.                       |
| `agent.interaction`  | L1     | Per `AIAgentInteraction`.                              |
| `agent.input`        | L1     | Per `AIAgentInteractionMessage` with role=user.        |
| `agent.output`       | L1     | Per `AIAgentInteractionMessage` with role=agent.       |
| `model.invoke`       | L3     | Per `LLMExecutionStep` from `AIAgentInteractionStep`.  |
| `tool.call`          | L5a    | Per `ActionInvocationStep` / `FunctionStep`.           |
| `environment.config` | L4a    | Per topic classification (Agent API path).             |
| `agent.state.change` | L1     | Per Agent API session start / end (live mapper).       |
| `policy.violation`   | cross  | Per Einstein Trust Layer policy hit.                   |
| `agent.handoff`      | L4a    | Per escalation (Agent API mapper).                     |

Each emitted event from the importer path includes `_identity` (the
Salesforce record `Id`) and `_timestamp` (record `LastModifiedDate`) for
re-import idempotency.

## OAuth setup

The adapter authenticates with Salesforce via the
[OAuth 2.0 JWT Bearer flow][oauth-jwt]. This is the supported
server-to-server flow for backfill agents — no interactive user login
or refresh-token rotation is needed.

[oauth-jwt]: https://help.salesforce.com/s/articleView?id=sf.remoteaccess_oauth_jwt_flow.htm&type=5

### 1. Create a Connected App in Salesforce

In your Salesforce org: **Setup → App Manager → New Connected App**.
Configure:

- **Connected App Name**: `LayerLens AgentForce Importer`
- **API (Enable OAuth Settings)**: ✅
- **Use digital signatures**: ✅ — upload your public-key X.509 certificate
- **Selected OAuth Scopes**:
  - `Manage user data via APIs (api)`
  - `Perform requests at any time (refresh_token, offline_access)`
  - `Access Agentforce Service APIs (agentforce_api)` (if available in
    your edition; otherwise `api` is sufficient for SOQL DMO reads)
- **Require Secret for Web Server Flow**: ✅
- **Callback URL**: any placeholder (e.g. `https://login.salesforce.com/`)
  — JWT Bearer flow does not actually use this.

Save and copy the **Consumer Key** — that's your `client_id`.

### 2. Generate a key pair

```bash
openssl req -x509 -nodes -newkey rsa:2048 \
    -keyout layerlens-agentforce.key \
    -out layerlens-agentforce.crt \
    -days 365 -subj "/CN=layerlens-agentforce"
```

Upload the `.crt` to the Connected App. Keep the `.key` secret.

### 3. Pre-authorize the integration user

**Setup → Connected Apps → Manage → Edit Policies**:

- **Permitted Users**: `Admin approved users are pre-authorized`
- Add a profile or permission set that includes the integration user.
  The integration user must have read access to the AgentForce DMOs
  (`AIAgentSession*`).

### 4. Configure the SDK

Pass the credentials via `SalesforceCredentials`. The `private_key`
field accepts three forms:

| Form                  | Example                              |
|-----------------------|--------------------------------------|
| `env:NAME` reference  | `env:SF_PRIVATE_KEY_PEM`             |
| Filesystem path       | `/etc/secrets/layerlens-agentforce.key` |
| Inline PEM string     | `-----BEGIN PRIVATE KEY-----\n...\n` |

```python
from layerlens.instrument.adapters.frameworks.agentforce import (
    SalesforceCredentials,
)

credentials = SalesforceCredentials(
    client_id="3MVG9...",                     # Connected App Consumer Key
    username="layerlens-agentforce@example.com",
    private_key="env:SF_PRIVATE_KEY_PEM",
    instance_url="https://example.my.salesforce.com",
)
```

The `SalesforceConnection.authenticate()` call constructs and signs the
JWT with `RS256` and exchanges it at
`https://${instance_url}/services/oauth2/token` for an access token.
Tokens are cached in-memory for ~1 hour and refreshed automatically.

## Salesforce specifics

- **Token lifetime**: ~2 hours, treated as 1 hour to leave room for
  clock drift. The adapter re-authenticates automatically when the
  cached token expires before the next operation.
- **Rate limits**: a warning is logged when the API daily limit
  consumption passes 80%. Salesforce returns the consumption in the
  `Sforce-Limit-Info` response header.
- **Incremental sync**: pass `last_import_timestamp` to
  `import_sessions(...)` to fetch only records modified since a
  watermark.
- **Batch size**: configurable via the `batch_size` constructor arg
  (default 200; the SOQL `IN` clause maximum is 2000).
- **SOQL injection**: every parent ID interpolated into the `WHERE … IN
  (…)` clause is validated against the `^[a-zA-Z0-9]{15}(?:[a-zA-Z0-9]{3})?$`
  Salesforce ID regex before splicing. Date / timestamp parameters are
  validated against ISO 8601 regexes.

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

Salesforce manages its own model keys (Einstein Trust Layer abstracts
the provider). The adapter does not own model API keys. The Salesforce
credentials themselves are intended to live in atlas-app's
`byok_credentials` table once M1.B ships — see `docs/adapters/byok.md`.

## Trust Layer round-trip

`TrustLayerImporter` exports the org's Einstein Trust Layer policy as
LayerLens YAML so the same guardrails can be re-evaluated outside the
Salesforce control plane:

```python
from layerlens.instrument.adapters.frameworks.agentforce import (
    SalesforceConnection,
    TrustLayerImporter,
)

connection = SalesforceConnection(credentials=credentials)
connection.authenticate()
config, yaml_str = TrustLayerImporter(connection).import_and_convert(
    policy_name="agentforce_trust_layer",
)
print(yaml_str)
```

The legacy alias `to_stratix_policy(...)` is retained for compatibility
with the original `stratix.*` adapter package and emits a
`DeprecationWarning`; new code should call `to_layerlens_policy(...)`
directly.

## Replay

`adapter.serialize_for_replay()` returns a `ReplayableTrace` with all
events captured during the current `import_sessions` call. Replay is a
re-emit operation: the adapter does not re-query Salesforce.
