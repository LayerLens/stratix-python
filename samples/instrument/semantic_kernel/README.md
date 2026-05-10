# Semantic Kernel instrumentation sample

End-to-end demo of `SemanticKernelAdapter` — builds a `Kernel` with
an `OpenAIChatCompletion("gpt-4o-mini")` service, registers
LayerLens filters via `SemanticKernelAdapter.instrument_kernel`, and
runs a single `invoke_prompt` call.

## Prerequisites

```bash
pip install 'layerlens[semantic-kernel,providers-openai]'
```

The `semantic-kernel` extra installs `semantic-kernel>=1.0,<2.0`
(Python 3.10+).

Required environment:

- `OPENAI_API_KEY` — used by `OpenAIChatCompletion`. If unset the
  sample exits with code `2`.
- `LAYERLENS_STRATIX_API_KEY` — your LayerLens API key (optional).
- `LAYERLENS_STRATIX_BASE_URL` — atlas-app base URL (optional).

## Run

```bash
uv run python -m samples.instrument.semantic_kernel.main
```

## What this demonstrates

| Component | What it proves (source: `main.py`) |
|---|---|
| `SemanticKernelAdapter(capture_config=CaptureConfig.standard())` | Standard capture profile. |
| `adapter.add_sink(HttpEventSink(...))` | Batched HTTP transport (`max_batch=10`, `flush_interval_s=1.0`). |
| `adapter.connect()` / `adapter.disconnect()` | Full lifecycle. |
| `adapter.instrument_kernel(kernel)` | Registers LayerLens filters on the kernel; module docstring states filter callbacks emit `agent.input` / `agent.output` / `model.invoke` events. |
| `kernel.invoke_prompt(prompt=..., arguments=KernelArguments())` | One async prompt invocation through the kernel. |
| `asyncio.run(_run(kernel))` | Sync-driver-around-async-kernel pattern. |
| Lazy import of `KernelArguments` inside `_run` | Keeps the module importable even when `semantic-kernel` is absent. |

## Expected output

When `OPENAI_API_KEY` is unset:

```text
OPENAI_API_KEY is not set; cannot run sample.
```

Exit code: `2`.

When `semantic-kernel` is not installed:

```text
semantic-kernel not installed. Install with:
    pip install 'layerlens[semantic-kernel,providers-openai]'
```

Exit code: `2`.

When the call succeeds:

```text
Response: <prompt result>
Telemetry shipped. Check the LayerLens dashboard adapter health page.
```

## Multi-tenancy note

This sample does not pass `org_id` to `SemanticKernelAdapter`. The
constructor does not yet accept `org_id` — production multi-tenant
wiring lands with the PR #118 adapter-side contract (currently
DRAFT). Once PR #118 merges, pass `org_id` to the adapter so every
emitted event carries it.
