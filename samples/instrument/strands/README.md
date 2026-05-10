# AWS Strands instrumentation sample

End-to-end demo of `StrandsAdapter` — builds a one-shot Strands
`Agent` backed by a Bedrock model, wraps it via
`StrandsAdapter.instrument_agent`, and runs a single call.

## Prerequisites

```bash
pip install 'layerlens[strands]'
```

The `strands` extra installs `strands-agents>=0.1,<1.0` (Python 3.10+).

Required environment:

- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` **or** `AWS_PROFILE`
  (or any standard boto3 credential source). If neither
  `AWS_ACCESS_KEY_ID` nor `AWS_PROFILE` is set the sample exits with
  code `2`.
- `AWS_REGION` — region where your Bedrock model access is enabled
  (Strands defaults to `us-west-2`).
- `BEDROCK_MODEL_ID` — Bedrock model ID for Strands to use; defaults
  to `us.anthropic.claude-3-5-sonnet-20241022-v2:0` if unset.
- `LAYERLENS_STRATIX_API_KEY` — your LayerLens API key (optional).
- `LAYERLENS_STRATIX_BASE_URL` — atlas-app base URL (optional).

## Run

```bash
uv run python -m samples.instrument.strands.main
```

## What this demonstrates

| Component | What it proves (source: `main.py`) |
|---|---|
| `StrandsAdapter(capture_config=CaptureConfig.standard())` | Standard capture profile. |
| `adapter.add_sink(HttpEventSink(...))` | Batched HTTP transport. |
| `adapter.connect()` / `adapter.disconnect()` | Full lifecycle. |
| `adapter.instrument_agent(agent)` | Wraps the Strands `Agent`; the module docstring states each call emits `agent.input` + `model.invoke` + `agent.output`. |
| `agent("What is 2 + 2?")` | One synchronous call against the configured Bedrock model. |
| `BEDROCK_MODEL_ID` env override | Demonstrates pulling the model id from environment with a sensible Claude 3.5 Sonnet default. |

## Expected output

When AWS credentials are not configured:

```text
AWS credentials are not set (need AWS_ACCESS_KEY_ID or AWS_PROFILE); cannot run sample.
```

Exit code: `2`.

When `strands-agents` is not installed:

```text
strands-agents not installed. Install with:
    pip install 'layerlens[strands]'
```

Exit code: `2`.

When the call succeeds:

```text
Response: <model reply>
Telemetry shipped. Check the LayerLens dashboard adapter health page.
```

## Multi-tenancy note

This sample does not pass `org_id` to `StrandsAdapter`. The
constructor does not yet accept `org_id` — production multi-tenant
wiring lands with the PR #118 adapter-side contract (currently
DRAFT). Once PR #118 merges, pass `org_id` to the adapter so every
emitted event carries it.
