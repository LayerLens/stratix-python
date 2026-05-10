# Google ADK instrumentation sample

End-to-end demo of `GoogleADKAdapter` — builds a one-shot `LlmAgent`
backed by `gemini-2.0-flash`, attaches LayerLens callbacks, and runs
a single turn through the ADK `InMemoryRunner`.

## Prerequisites

```bash
pip install 'layerlens[google-adk]'
```

Required environment (one of):

- `GOOGLE_API_KEY` — used by the Gemini model against Google AI
  Studio. **OR**
- `GOOGLE_GENAI_USE_VERTEXAI=true` — route through Vertex AI; supply
  Application Default Credentials.

If neither is set the sample exits with code `2`.

Optional:

- `LAYERLENS_STRATIX_API_KEY` — your LayerLens API key.
- `LAYERLENS_STRATIX_BASE_URL` — atlas-app base URL.

## Run

```bash
uv run python -m samples.instrument.google_adk.main
```

## What this demonstrates

| Component | What it proves (source: `main.py`) |
|---|---|
| `GoogleADKAdapter(capture_config=CaptureConfig.standard())` | Standard capture profile. |
| `adapter.add_sink(HttpEventSink(...))` | Batched HTTP transport. |
| `adapter.connect()` / `adapter.disconnect()` | Full lifecycle. |
| `adapter.instrument_agent(agent)` | Attaches LayerLens callbacks to the `LlmAgent`; each callback emits a LayerLens event (per `main.py` module docstring). |
| `InMemoryRunner(agent=agent, app_name="layerlens-sample")` + `run_async` | Async iteration over the ADK event stream; text parts are concatenated and printed. |
| Pre-created session via `runner.session_service.create_session(...)` | Demonstrates the required ADK pattern for giving `run_async` a session to write into. |

## Expected output

When neither `GOOGLE_API_KEY` nor `GOOGLE_GENAI_USE_VERTEXAI=true` is
set:

```text
Neither GOOGLE_API_KEY nor GOOGLE_GENAI_USE_VERTEXAI is set; cannot run sample.
```

Exit code: `2`.

When `google-adk` is not installed:

```text
google-adk not installed. Install with:
    pip install 'layerlens[google-adk]'
```

Exit code: `2`.

When the call succeeds:

```text
Response: <model reply>
Telemetry shipped. Check the LayerLens dashboard adapter health page.
```

## Multi-tenancy note

This sample does not pass `org_id` to `GoogleADKAdapter`. The
constructor does not yet accept `org_id` — production multi-tenant
wiring lands with the PR #118 adapter-side contract (currently
DRAFT). Once PR #118 merges, pass `org_id` to the adapter so every
emitted event carries it.
