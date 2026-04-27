# browser_use instrumentation sample

End-to-end demo of `BrowserUseAdapter` — runs **offline** with no
`browser-use` install, no Playwright, no OpenAI key, no network calls.
It uses a duck-typed `_FakeAgent` so the wrapper, lifecycle hooks,
truncation policy, and event emission can be exercised on any
developer laptop.

## Run

```bash
# Happy path — three-step navigation.
python -m samples.instrument.browser_use.main

# Failure path — exercises agent.error emission before re-raise.
python -m samples.instrument.browser_use.main --fail
```

Expected output for the happy path (event count and order are
deterministic):

```text
Agent finished. 3 step(s) executed.

Emitted 14 event(s):
  -        environment.config  org=org_demo  agent=demo-bot model=gpt-4o-mini
  -    browser.session.start  org=org_demo  session=...
  -              agent.input  org=org_demo  task=find the price of a Logitech mouse...
  -          browser.navigate  org=org_demo  url=https://store.example.com/
  -            browser.action  org=org_demo  action=navigate
  ...
  -             agent.output  org=org_demo  duration_ns=...
  -      agent.state.change  org=org_demo
```

Notice the screenshot lines render as
`<dropped:screenshot:sha256:...>` rather than the multi-megabyte
PNG bytes — the truncation policy refuses to embed binary blobs in
events.

## What the sample exercises

| Component | What it proves |
|---|---|
| `BrowserUseAdapter.connect()` | Adapter reaches `HEALTHY` even when `browser-use` is not installed. |
| `BrowserUseAdapter.instrument_agent(agent)` | `agent.run` is wrapped with the traced async shim. |
| Lifecycle hooks | `browser.session.start`, `agent.input`, `browser.navigate`, `browser.action`, `browser.screenshot`, `model.invoke`, `cost.record`, `agent.output`, `agent.state.change` all emitted in order. |
| Truncation policy | The 50 KB screenshot blob is replaced by a SHA-256 reference; the same blob across steps produces the same hash (replay correlation). |
| Multi-tenant org_id | Every emit carries `org_id="org_demo"`, demonstrating the PR #118 contract. |
| PR #115 error path (`--fail`) | When the agent raises, `agent.error` is emitted BEFORE the exception propagates so the dashboard sees a complete pair. |
| `BrowserUseAdapter.disconnect()` | `agent.run` is restored to the original. |
| `resilience_snapshot()` | Per-callback failure counters surface for operators. |

## Going to a real run

Swap `_FakeAgent` for a real browser_use Agent and route events to
the LayerLens dashboard via `HttpEventSink`:

```python
from browser_use import Agent
from langchain_openai import ChatOpenAI

from layerlens.instrument.transport.sink_http import HttpEventSink
from layerlens.instrument.adapters.frameworks.browser_use import (
    BrowserUseAdapter,
    instrument_agent,
)

sink = HttpEventSink(adapter_name="browser_use")

agent = Agent(
    task="find the price of a Logitech MX Master 3S on a demo store",
    llm=ChatOpenAI(model="gpt-4o-mini"),
)

adapter = instrument_agent(agent, org_id="org_acme")
adapter.add_sink(sink)

result = await agent.run()

adapter.disconnect()
sink.close()
```

Required env for the live path: `LAYERLENS_STRATIX_API_KEY`,
`LAYERLENS_STRATIX_BASE_URL`, plus whatever credentials your LLM
provider needs (e.g. `OPENAI_API_KEY`).

Install with: `pip install 'layerlens[browser-use]'`.
