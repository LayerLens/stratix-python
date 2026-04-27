# Salesforce Agentforce sample

Runnable end-to-end sample for the
`layerlens.instrument.adapters.frameworks.agentforce` adapter.

The sample is **fully mocked** — it makes no network calls to either
Salesforce or LayerLens. It exists to demonstrate the API surface and act
as a smoke test that the `[agentforce]` extra installs cleanly.

## Install

```bash
pip install 'layerlens[agentforce]'
```

The `[agentforce]` extra pulls in `requests>=2.28` (the JWT Bearer flow
and SOQL HTTP transport).

## Run

```bash
python -m samples.instrument.agentforce.main
```

You should see four labeled flows print to stdout:

* `[backfill]` — SOQL session backfill via the Data Cloud DMO importer.
* `[live]` — Synchronous Agent API request / response capture.
* `[trust-layer]` — Einstein Trust Layer export to LayerLens YAML policy.
* `[evaluator]` — Einstein evaluator offline behavior (logs the
  zero-score fallback when no LayerLens API key is configured).

The sample exits 0 on success.

## Live Salesforce auth (optional)

If you have a Salesforce Connected App with the JWT Bearer flow
configured, set these environment variables before running and the
sample will additionally exercise a live `connect()` against the org:

```bash
export SALESFORCE_CLIENT_ID="3MVG9..."
export SALESFORCE_USERNAME="agent-importer@example.com"
export SALESFORCE_PRIVATE_KEY="env:SF_PRIVATE_KEY_PEM"   # or a file path / raw PEM
export SALESFORCE_INSTANCE_URL="https://example.my.salesforce.com"
```

`SALESFORCE_PRIVATE_KEY` accepts three forms:

| Form | Example |
|------|---------|
| `env:NAME` reference | `env:SF_PRIVATE_KEY_PEM` |
| Filesystem path | `/etc/secrets/sf-jwt.pem` |
| Inline PEM string | `-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----` |

See `docs/adapters/frameworks-agentforce.md` for the OAuth Connected
App setup, the Trust Layer policy round-trip, and the full event taxonomy
the adapter emits.
