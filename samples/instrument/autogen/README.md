# AutoGen instrumentation sample

End-to-end demo of `AutoGenAdapter` wiring two AutoGen agents together
through a single-turn `initiate_chat` exchange. The adapter monkey-patches
`ConversableAgent.send` / `receive` / `generate_reply` (per
`lifecycle.py`'s adapter docstring) so each message hop becomes a
LayerLens event shipped to atlas-app via `HttpEventSink`.

## Install

```bash
pip install 'layerlens[autogen,providers-openai]'
```

The `autogen` extra pins `pyautogen>=0.2,<0.5`. The `providers-openai`
extra adds `openai>=1.30,<2`. The default `pip install layerlens` does NOT
pull either — that's the lazy-import guarantee tested by
`tests/instrument/test_lazy_imports.py`.

## Run

```bash
export OPENAI_API_KEY=sk-...
export LAYERLENS_STRATIX_API_KEY=ll-...   # optional
export LAYERLENS_STRATIX_BASE_URL=...     # optional
python -m samples.instrument.autogen.main
```

If `OPENAI_API_KEY` is unset the sample exits with code 2 (see
`main.py:30-32`). If `pyautogen` is not installed the sample exits with
code 2 and points at the right install command (see `main.py:37-43`).

## Expected output

```text
Response: <one-sentence model answer to "What is 2 + 2?">
Telemetry shipped. Check the LayerLens dashboard adapter health page.
```

The response is whatever the assistant returns via
`assistant.last_message(user)` (see `main.py:80`). The conversation is
capped at a single turn — `max_consecutive_auto_reply=0` on the
`UserProxyAgent` and `is_termination_msg=lambda _msg: True` ensure the
exchange ends after the assistant's first reply.

## What this sample demonstrates

| Component | Where in `main.py` | What it proves |
|---|---|---|
| `HttpEventSink(adapter_name="autogen", path="/telemetry/spans")` | L45-50 | Telemetry transport is configured against the atlas-app spans endpoint with a 10-event batch and 1 s flush interval. |
| `AutoGenAdapter(capture_config=CaptureConfig.standard())` | L52 | Adapter is constructed using the standard capture profile. |
| `adapter.add_sink(sink)` + `adapter.connect()` | L53-54 | The sink is wired and the adapter lifecycle reaches `HEALTHY`. |
| `AssistantAgent` + `UserProxyAgent` with single-turn config | L64-75 | Demonstrates a deterministic two-agent topology: a concise assistant and a non-interactive proxy that terminates after one reply. |
| `adapter.connect_agents(assistant, user)` | L78 | The adapter monkey-patches both agents' `send` / `receive` / `generate_reply` methods so message hops are intercepted. |
| `user.initiate_chat(assistant, message="What is 2 + 2?")` | L79 | A single-turn conversation runs through the instrumented hooks, emitting telemetry on every message hop. |
| `sink.close()` + `adapter.disconnect()` | L83-84 | Adapter is torn down cleanly inside a `try/finally`; monkey-patches are reverted by `disconnect()`. |

## Production note (multi-tenancy)

This sample does **not** pass `org_id` to `AutoGenAdapter`. In a
multi-tenant deployment every adapter must be scoped to a tenant — pass
`org_id=...` to the adapter (or via the `STRATIX` instance) so emitted
telemetry is correctly attributed. The omission here is acceptable for a
local single-tenant smoke test only.
