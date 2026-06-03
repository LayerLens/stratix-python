# Agentforce adapter

Batch-imports [Salesforce Agentforce](https://www.salesforce.com/agentforce/)
sessions and interactions from Data Cloud Data Model Objects (DMOs). Unlike
the in-process framework adapters, Agentforce is observed post-hoc by
querying Salesforce, so the integration is OAuth-authenticated HTTP rather
than a callback or filter API.

## Install

```bash
pip install layerlens[agentforce]
```

Pulls `httpx>=0.27.0` for the Salesforce REST client.

## OAuth setup

The adapter authenticates with Salesforce via the **OAuth 2.0 Client
Credentials** flow. You'll need:

1. **A Connected App** in your Salesforce org with:
   - OAuth scopes: `api`, `refresh_token`, `cdp_query_api`
   - "Enable Client Credentials Flow" turned on
   - A "Run As" user with permission to read `AIAgentSession__dlm`,
     `AIAgentInteraction__dlm`, and `AIAgentConfiguration__dlm` on Data Cloud
2. **The Consumer Key** (client ID) and **Consumer Secret** (client secret)
   from the Connected App
3. **Your org's My Domain URL** (e.g. `https://myorg.my.salesforce.com`)

In Setup → App Manager, create a new Connected App, enable OAuth settings,
add the scopes above, then under "Client Credentials Flow" assign a user to
run as. After saving, copy the consumer key/secret from "Manage Consumer
Details."

Pass the credentials to `connect()`:

```python
import os
from layerlens.instrument.adapters.frameworks import AgentforceAdapter

adapter = AgentforceAdapter(client=layerlens_client)
adapter.connect(
    credentials={
        "client_id":     os.environ["SF_CLIENT_ID"],
        "client_secret": os.environ["SF_CLIENT_SECRET"],
        "instance_url":  os.environ["SF_INSTANCE_URL"],
    },
)
```

`connect()` performs the client-credentials token exchange against
`{instance_url}/services/oauth2/token` and caches the access token on the
adapter for subsequent queries.

## Usage

```python
adapter.connect(credentials={...})

# Incremental import. Pass the previous run's next_cursor for exactly-once.
summary = adapter.import_sessions(limit=50, since_cursor=previous_cursor)

print(summary["sessions_imported"], summary["events_emitted"])
next_cursor = summary["next_cursor"]      # persist for the next run

adapter.disconnect()
```

`import_sessions` accepts `start_date`, `end_date`, `limit`, and
`since_cursor`. The returned `next_cursor` is the max `StartTime` seen, so a
caller can persist it and pass it back to incrementally sync without
re-importing.

## Event surface

Each Agentforce session becomes its own trace via `_begin_run` /
`_end_run`. Inside a session:

- `environment.config` — one event per session with the agent configuration
  (model name, instructions, topic/action counts) pulled from
  `AIAgentConfiguration__dlm`.
- `model.invoke` for LLM/generative steps (`StepType` ∈ {llm, model,
  generative}), with prompt/completion token counts.
- `tool.call` for action/function/tool/flow steps, with tool name, input,
  and output.
- `agent.handoff` for escalation/handoff/transfer steps, with the escalation
  target.
- `agent.error` for steps with a non-empty `ErrorMessage`.

Step types are detected from the `StepType` field on
`AIAgentInteraction__dlm` and dispatched through `_STEP_DISPATCH`.

## Sample

[`samples/instrument/agentforce/example.py`](../../../samples/instrument/agentforce/example.py)

## Compat

- Salesforce REST API v62.0
- Python 3.9+
