# Salesforce AgentForce instrumentation sample

> ⚠ **Smoke-only sample.** Salesforce AgentForce is a remote REST API,
> not a Python library. A real `import_sessions` call requires:
>
> 1. A Salesforce Connected App with the JWT Bearer flow configured.
> 2. A private key (PEM) authorized for the Connected App.
> 3. A user with read access to the `AIAgentSession` DMOs.
>
> Most CI environments don't have those, so this sample only exercises
> the adapter's **local wiring** (instantiate, attach sink, report
> `get_adapter_info`) and exits cleanly. If the required `SALESFORCE_*`
> env vars are present, it will additionally attempt one `connect()`
> call to verify JWT auth — but no session import is performed.

## Install

```bash
pip install 'layerlens[agentforce]'
```

The `agentforce` extra installs `requests>=2.28` only (the adapter talks
to Salesforce over plain HTTP — there is no Salesforce SDK dependency).
The default `pip install layerlens` does NOT pull `requests` for the
instrument layer — that's the lazy-import guarantee tested by
`tests/instrument/test_lazy_imports.py`.

## Run

```bash
python -m samples.instrument.agentforce.main
```

Optional environment variables for the auth check path:

| Variable | Purpose |
|---|---|
| `SALESFORCE_CLIENT_ID` | Connected App consumer key. |
| `SALESFORCE_USERNAME` | Salesforce user the JWT is issued for. |
| `SALESFORCE_PRIVATE_KEY` | PEM-encoded private key. |
| `SALESFORCE_INSTANCE_URL` | Your org's My Domain URL (defaults to `https://login.salesforce.com`). |
| `LAYERLENS_STRATIX_API_KEY` | LayerLens API key (optional). |
| `LAYERLENS_STRATIX_BASE_URL` | atlas-app base URL (optional). |

The smoke path requires **no** environment variables.

## Expected output

### Smoke path (no `SALESFORCE_*` env vars)

```text
SALESFORCE_* env vars are not set; running smoke check only.
Adapter: salesforce_agentforce v0.1.0 (framework=salesforce_agentforce)
```

The smoke path prints the result of `adapter.get_adapter_info()` (see
`main.py:74-78`) and exits 0 with no network I/O.

### Auth check path (all three required `SALESFORCE_*` vars set)

```text
AgentForce adapter authenticated against Salesforce.
Telemetry shipped (smoke). Check the LayerLens dashboard.
```

If JWT auth fails, the sample prints `Salesforce auth failed: <error>` to
stderr and exits 1 (see `main.py:102-104`).

## What this sample demonstrates

| Component | Where in `main.py` | What it proves |
|---|---|---|
| `HttpEventSink(adapter_name="salesforce_agentforce", path="/telemetry/spans")` | L60-65 | Telemetry transport is configured against the atlas-app spans endpoint with a 10-event batch and 1 s flush interval. |
| `_have_salesforce_env()` | L51-56 | The sample gates network I/O behind explicit env-var presence; CI runs the smoke path by default. |
| `AgentForceAdapter(capture_config=CaptureConfig.standard())` (smoke path) | L74 | Adapter can be constructed and inspected (`get_adapter_info()`) without any Salesforce credentials. |
| `SalesforceCredentials(...)` + `AgentForceAdapter(credentials=...)` (auth path) | L81-93 | The adapter accepts JWT Bearer flow credentials through a strongly-typed `SalesforceCredentials` model. |
| `adapter.connect()` (auth path) | L97 | Authenticates against Salesforce; raises `SalesforceAuthError` on failure (caught at L102). |
| `import_sessions(...)` (documented, not executed) | L99-101 (commented) | The full path — `import_sessions(start_date=..., limit=...)` — is shown in a comment but not run, because a real run requires DMO read access. |
| `sink.close()` + `adapter.disconnect()` | L78 / L106-107 | Adapter is torn down cleanly in both paths. |

## Production note (multi-tenancy)

This sample does **not** pass `org_id` to `AgentForceAdapter`. In a
multi-tenant deployment every adapter must be scoped to a tenant — pass
`org_id=...` to the adapter (or via the `STRATIX` instance) so imported
AgentForce sessions are correctly attributed to the owning org. The
omission here is acceptable for a local single-tenant smoke test only.
