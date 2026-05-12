# browser_use framework adapter

`layerlens.instrument.adapters.frameworks.browser_use.BrowserUseAdapter`
instruments [browser-use](https://github.com/browser-use/browser-use) —
the LLM-driven Playwright agent that performs autonomous web
navigation, form filling, and content extraction. The adapter wraps
`Agent.run()` (and `Agent.run_sync()` when present), threads per-step
browser / action / screenshot / DOM / model events through the
LayerLens pipeline, and applies the field-specific truncation policy
so multi-megabyte screenshot / DOM payloads cannot blow past the
ingestion sink limits.

## Install

```bash
pip install 'layerlens[browser-use]'
```

Pulls `browser-use>=0.1.0,<2`. Requires Python 3.11+ (browser_use's
own constraint) and Playwright (the runtime SDK pulls it transitively
and runs `playwright install chromium` on first use).

## Quick start

```python
import asyncio

from browser_use import Agent
from langchain_openai import ChatOpenAI

from layerlens.instrument.adapters.frameworks.browser_use import (
    BrowserUseAdapter,
    instrument_agent,
)
from layerlens.instrument.transport.sink_http import HttpEventSink

sink = HttpEventSink(adapter_name="browser_use")

agent = Agent(
    task="find the price of a Logitech MX Master 3S on a demo store",
    llm=ChatOpenAI(model="gpt-4o-mini"),
)

# One-liner: construct adapter, connect, wrap agent, return adapter.
adapter = instrument_agent(agent, org_id="org_acme")
adapter.add_sink(sink)

result = asyncio.run(agent.run())

adapter.disconnect()
sink.close()
```

For an offline reproduction (no `browser-use` install required) see
`samples/instrument/browser_use/`.

## What's wrapped

`adapter.instrument_agent(agent)` patches the following on each Agent:

- `run` — async entry point. Emits the full session lifecycle plus
  per-step browser / action / screenshot / model events.
- `run_sync` — sync entry point (when present in the browser_use
  build). Same semantics.

`disconnect()` restores all originals and clears wrapping state.

## Capabilities

| Capability                         | Declared |
|------------------------------------|----------|
| `AdapterCapability.TRACE_TOOLS`    | Yes      |
| `AdapterCapability.TRACE_MODELS`   | Yes      |
| `AdapterCapability.TRACE_STATE`    | Yes      |
| `AdapterCapability.STREAMING`      | Yes      |
| `AdapterCapability.REPLAY`         | Yes      |
| `AdapterCapability.TRACE_HANDOFFS` | No (browser_use is single-agent) |

## Events emitted

| Event                    | Layer         | When                                                              |
|--------------------------|---------------|-------------------------------------------------------------------|
| `environment.config`     | L4a           | First time an agent is registered. Captures model, browser, task. |
| `browser.session.start`  | L1            | Beginning of every `run`. Includes a generated `session_id`.      |
| `agent.input`            | L1            | Same boundary as `browser.session.start`. Carries the task.       |
| `browser.navigate`       | L5a           | Per page-load (URL change).                                       |
| `browser.action`         | L5a           | Per click / type / select / scroll. Mirrored as `tool.call`.      |
| `tool.call`              | L5a           | Mirror of `browser.action` for unified analytics.                 |
| `browser.screenshot`     | L5c           | Per screenshot. **Bytes DROPPED** to a SHA-256 reference.         |
| `browser.dom.extract`    | L5c           | Per DOM snapshot. HTML capped at 16 KiB.                          |
| `model.invoke`           | L3            | Per LLM call inside the reasoning loop.                           |
| `cost.record`            | cross-cutting | Per LLM call (when token counts are available).                   |
| `agent.output`           | L1            | End of every `run`. Includes `duration_ns` and any `error`.       |
| `agent.state.change`     | cross-cutting | After `agent.output` — `session_complete` or `session_failed`.    |
| `agent.error`            | L1            | When `run` raises. Emitted BEFORE the exception propagates.       |
| `tool.error`             | L5a           | When a browser action raises. Paired with the failed `tool.call`. |
| `model.error`            | L3            | When the LLM call raises. Paired with the failed `model.invoke`.  |

## Truncation policy (CRITICAL)

browser_use payloads are uniquely susceptible to unbounded data — a
single navigation step can produce multi-megabyte base64 PNG
screenshots, DOM HTML over 100 KB, and verbose page content. The
adapter wires `DEFAULT_POLICY` from
`layerlens.instrument.adapters._base.truncation` from day one with
the following per-field caps:

| Field                                                   | Cap                  |
|---------------------------------------------------------|----------------------|
| `screenshot`, `image_data`, `image_b64`, `binary_data`  | DROPPED → SHA-256 ref |
| `html`, `dom`, `page_content`                           | 16 KiB               |
| `prompt`, `completion`, `messages`, `output`, `input`   | 4 KiB                |
| `tool_input`, `tool_output`, `arguments`                | 2 KiB                |
| `state`, `context`, `history`                           | 8 KiB                |
| `traceback`, `stacktrace`                               | 8 frames             |

Truncations are NEVER silent — every clipped field appears in the
`_truncated_fields` audit list attached to the emitted payload.
Customers who need full-fidelity screenshots should ship them through
a separate object store (S3 / R2) and embed only the storage
reference in events.

## Multi-tenancy

The adapter binds an `org_id` at construction (`org_id` kwarg or
resolved from `stratix.org_id` / `stratix.organization_id`) and
stamps it onto every emitted payload. Caller-supplied `org_id` values
are overwritten defensively to prevent cross-tenant leaks via misuse.

```python
adapter = BrowserUseAdapter(stratix=client, org_id="org_acme")
# Every event payload carries org_id="org_acme".
```

## Resilience

Every public hook is wrapped in `try / except` so an exception in our
observability code can NEVER crash the customer's browser_use agent.
Failures bump the per-callback resilience counter:

```python
adapter.resilience_snapshot()
# {
#   "resilience_failures_total": 0,
#   "resilience_failures_by_callback": {},
#   "resilience_last_error": None,
# }
```

Operators surface this through the adapter health endpoint to detect
silent observability degradation early.

## Error-aware emission

When the wrapped agent raises (rate limit, page-load timeout, LLM
outage, malformed prompt), the adapter emits a structured `agent.error`
event BEFORE re-raising the exception. Dashboards always see a complete
`agent.input` → `agent.error` → `agent.output` triple — never a hung
"start" with no matching "end".

The same contract applies to `tool.error` (action failures) and
`model.error` (LLM call failures).

## Capture config

```python
from layerlens.instrument.adapters._base import CaptureConfig

# Recommended.
adapter = BrowserUseAdapter(capture_config=CaptureConfig.standard())

# Heavy: include screenshot + DOM extract events. They still respect
# the truncation policy — DROP for screenshots, 16 KiB for HTML.
adapter = BrowserUseAdapter(
    capture_config=CaptureConfig(
        l1_agent_io=True,
        l3_model_metadata=True,
        l4a_environment_config=True,
        l5a_tool_calls=True,
        l5c_tool_environment=True,  # screenshot + DOM events
    ),
)
```

## browser_use specifics

- **Async-only by default.** browser_use's `Agent.run` is async. The
  adapter exposes both sync (`_create_traced_run_sync`) and async
  (`_create_traced_run_async`) wrappers; instrumentation auto-detects
  which methods are present on the agent.
- **History walk fallback.** browser_use returns an
  `AgentHistoryList` from `Agent.run`. The adapter walks the history
  at the end of every run to backfill per-step events in case the
  customer constructed an agent before the per-step hooks existed.
- **Pydantic v2.** browser_use uses Pydantic v2 internally. The
  adapter declares `requires_pydantic = PydanticCompat.V2_ONLY` so
  the catalog UI warns customers pinning v1.
- **No native callback bus.** browser_use does not expose a callback
  registration API today — the adapter uses the wrapper pattern
  (preserve-then-restore on `disconnect`). When upstream adds a
  callback bus the adapter will switch to it without breaking the
  public surface.

## BYOK

browser_use's LLM client (LangChain `ChatOpenAI`, `ChatAnthropic`,
etc.) reads its own credentials. The adapter does not own them.
For platform-managed BYOK see `docs/adapters/byok.md` (atlas-app M1.B).

## Replay

The adapter implements `serialize_for_replay()` and declares
`AdapterCapability.REPLAY`. The serialized trace contains every
emitted event (with truncation already applied — replays do not pay
the bytes cost twice) plus the bound `org_id` and framework version.

```python
trace = adapter.serialize_for_replay()
# trace.events  -> list of {"event_type", "payload", "timestamp_ns"}
# trace.config  -> {"capture_config", "org_id", "framework_version"}
# trace.metadata -> {"resilience_failures": {...}}
```
