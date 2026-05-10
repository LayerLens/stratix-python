# AWS Bedrock Agents instrumentation sample

End-to-end demo of `BedrockAgentsAdapter` — wraps a
`bedrock-agent-runtime` boto3 client, runs a single `invoke_agent`
call, drains the streamed response, and ships telemetry events to
atlas-app via `HttpEventSink`.

> **Live-only.** This sample requires a real Bedrock Agent ID; there
> is no mock mode. If `BEDROCK_AGENT_ID` is unset the sample exits
> cleanly with code `2`.

## Prerequisites

```bash
pip install 'layerlens[bedrock-agents]'
```

The `bedrock-agents` extra installs `boto3>=1.34`.

Required environment:

- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` (or any standard
  boto3 credential source — IAM role, profile, etc.).
- `AWS_REGION` — region your agent lives in (defaults to
  `us-east-1`).
- `BEDROCK_AGENT_ID` — your Bedrock Agent ID (e.g. `ABCDEFGHIJ`).
- `BEDROCK_AGENT_ALIAS_ID` — alias to invoke (defaults to
  `TSTALIASID`).
- `LAYERLENS_STRATIX_API_KEY` — your LayerLens API key (optional).
- `LAYERLENS_STRATIX_BASE_URL` — atlas-app base URL (optional).

## Run

```bash
uv run python -m samples.instrument.bedrock_agents.main
```

## What this demonstrates

| Component | What it proves (source: `main.py`) |
|---|---|
| `BedrockAgentsAdapter(capture_config=CaptureConfig.standard())` | Standard capture profile. |
| `adapter.add_sink(HttpEventSink(...))` | Batched HTTP transport. |
| `adapter.connect()` / `adapter.disconnect()` | Full lifecycle. |
| `adapter.instrument_client(client)` | Registers event hooks on the boto3 `bedrock-agent-runtime` client; the module docstring states this emits `agent.input` + `model.invoke` + `tool.call` + `agent.output` events. |
| `client.invoke_agent(agentId=..., agentAliasId=..., sessionId=uuid4(), inputText="What is 2 + 2?")` | Live invocation against your configured agent. |
| Streamed response drain (`for event in response["completion"]: ...`) | Trace events fire as each chunk is iterated; concatenated bytes are decoded and printed. |

## Expected output

When `BEDROCK_AGENT_ID` is unset:

```text
BEDROCK_AGENT_ID is not set; cannot run sample.
```

Exit code: `2`.

When `boto3` is not installed:

```text
boto3 not installed. Install with:
    pip install 'layerlens[bedrock-agents]'
```

Exit code: `2`.

When the call succeeds:

```text
Response: <agent reply>
Telemetry shipped. Check the LayerLens dashboard adapter health page.
```

## Multi-tenancy note

This sample does not pass `org_id` to `BedrockAgentsAdapter`. The
constructor does not yet accept `org_id` — production multi-tenant
wiring lands with the PR #118 adapter-side contract (currently
DRAFT). Once PR #118 merges, pass `org_id` to the adapter so every
emitted event carries it.
