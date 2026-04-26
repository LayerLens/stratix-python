# CopilotKit Samples

Production-shaped reference for building CopilotKit-powered UIs on top of
LayerLens. Two agents ship here:

| Agent | Purpose |
|---|---|
| `agents/evaluator_agent.py` | Multi-step evaluation with human-in-the-loop judge selection. |
| `agents/investigator_agent.py` | Trace investigation (errors, latency, cost hot spots). No HITL. |

The evaluator is the interesting one — it uses CopilotKit's current idiom for
HITL: an **LLM-driven agent** with **backend tools** for data access and a
**frontend tool** for human confirmation, rendered as a rich widget in the
chat.

## Prerequisites

```bash
pip install layerlens --index-url https://sdk.layerlens.ai/package \
    "copilotkit==0.1.87" "langchain==1.2.15" "langchain-openai>=1.1.0,<2.0.0" \
    "langgraph==1.1.9" "ag-ui-langgraph==0.0.34" "pydantic>=2,<3" fastapi uvicorn
npm install @copilotkit/react-core@1.56.3 @copilotkit/react-ui@1.56.3 @copilotkit/runtime@1.56.3
export LAYERLENS_STRATIX_API_KEY=your-layerlens-key
export OPENAI_API_KEY=your-openai-key   # the evaluator's LLM
```

For byte-identical transitive deps, install from the committed lockfile:

```bash
pip install -r samples/copilotkit/app/backend/requirements.lock
```

### Version matrix

| Package | Pin |
|---|---|
| `layerlens` | `>=1.6.0` |
| `copilotkit` (Python SDK) | `==0.1.87` |
| `langchain` | `==1.2.15` |
| `langchain-openai` | `>=1.1.0,<2.0.0` |
| `langgraph` | `==1.1.9` |
| `ag-ui-langgraph` | `==0.0.34` |
| `pydantic` | `>=2,<3` |
| `@copilotkit/react-core` | `==1.56.3` (pulls in `@ag-ui/client==0.0.52`) |
| `@copilotkit/react-ui` | `==1.56.3` |
| `@copilotkit/runtime` | `==1.56.3` |
| Python | `>=3.10,<3.13` |
| Next.js | `>=15` |

## Architecture

```
+-- Browser (Next.js + CopilotKit UI) ------------------------+
|                                                             |
|   <CopilotChat>  +-- useCopilotAction("confirm_judge", ---+ |
|                  |   renderAndWaitForResponse: picker)   | |
|                  +----------------------------------------+ |
|                                |                             |
|                                v  respond({id, name})        |
|                                                             |
+-------------------------- /api/copilotkit ------------------+
                             |
                             v
+-- FastAPI backend ------------------------------------------+
|                                                             |
|   ag_ui_langgraph.add_langgraph_fastapi_endpoint            |
|             (wraps the graph in LangGraphAGUIAgent)         |
|                             |                                |
|                             v                                |
|   langchain.agents.create_agent                             |
|     - model: ChatOpenAI(gpt-4o-mini)                        |
|     - middleware: [CopilotKitMiddleware()]                  |
|     - tools:                                                |
|         list_judges          (backend)                      |
|         list_recent_traces   (backend)                      |
|         run_trace_evaluation (backend)                      |
|         get_evaluation_result(backend)                      |
|         confirm_judge        (frontend, bridged by middleware)
|                             |                                |
|                             v                                |
|   LayerLens Python SDK (Stratix client)                     |
+-------------------------------------------------------------+
```

The LLM drives the conversation. When its system prompt says to confirm a
judge with the user, it emits a tool call for `confirm_judge` with the
list of candidates as arguments. `CopilotKitMiddleware` routes that call
to the frontend, which renders the picker via `useCopilotAction`. When
the user selects a judge, the frontend's `respond({id, name})` returns
the selection as the tool result, and the LLM continues.

## Why this pattern (and not `interrupt()`)

An earlier revision of this sample used a custom `StateGraph` with
`langgraph.types.interrupt()` for HITL. That path hit two protocol-level
bugs in `ag-ui-langgraph` (tracked upstream at
[`ag-ui-protocol/ag-ui#1582`](https://github.com/ag-ui-protocol/ag-ui/issues/1582)
and [`#1584`](https://github.com/ag-ui-protocol/ag-ui/issues/1584)) and
required a local workaround subclass that reached into private internals
of the CopilotKit Python SDK. It worked, but it wasn't the pattern
CopilotKit themselves ship in their active showcases.

The current pattern — **LLM + frontend tool via `useCopilotAction`** —
matches CopilotKit's `hitl_in_chat_agent.py` and `interrupt_agent.py`
starters. It sidesteps the `interrupt()` pipeline entirely:

- No backend pause/resume boundary, so the two upstream bugs don't
  apply to this sample.
- No checkpointer needed (the `create_agent` driver owns state).
- The HITL prompt is **structured data**, not free-form text — so the
  frontend can render a real UI widget (a card list, a picker, a form)
  instead of a text input.
- The LLM naturally handles follow-ups ("actually, use the other judge")
  without any extra glue.

## Agents

### `evaluator_agent.py`

`langchain.agents.create_agent` with `CopilotKitMiddleware()`, four
backend `@tool` functions, and a system prompt that guides the LLM
through the evaluation flow. ~160 lines total, most of it tool
docstrings and the prompt itself. Requires `OPENAI_API_KEY` (or inject
your own `ChatModel` via `build_graph(model=...)`).

Tools:

- `list_judges() -> [{id, name, goal}]`
- `list_recent_traces(limit=20) -> [{id, filename, created_at}]`
- `run_trace_evaluation(trace_id, judge_id) -> {evaluation_id, status}`
- `get_evaluation_result(evaluation_id) -> {status, passed, score, reasoning}`
- `confirm_judge({candidates})` — **frontend tool**; declared in
  `app/frontend/app/page.tsx` via `useCopilotAction` and
  bridged into the LLM's toolbelt by `CopilotKitMiddleware`. Returns
  `{id, name}` once the user picks.

### `investigator_agent.py`

A procedural `StateGraph` that fetches a trace, extracts events,
and flags errors / slow spans / high token usage. No HITL, no LLM.
Good reference for a non-conversational agent that still plugs into
the CopilotKit runtime.

## Frontend wiring

```tsx
// app/page.tsx
"use client";

import { CopilotChat } from "@copilotkit/react-ui";
import { useCopilotAction, useCopilotChat } from "@copilotkit/react-core";
import { TextMessage, Role } from "@copilotkit/runtime-client-gql";

export default function Page() {
  const { appendMessage, isLoading } = useCopilotChat();

  useCopilotAction({
    name: "confirm_judge",
    description:
      "Ask the user to choose which judge to apply. Returns {id, name}.",
    parameters: [
      {
        name: "candidates",
        type: "object[]",
        required: true,
        attributes: [
          { name: "id", type: "string", required: true },
          { name: "name", type: "string", required: true },
          { name: "goal", type: "string", required: true },
        ],
      },
    ],
    renderAndWaitForResponse: ({ args, respond, status }) => {
      if (status === "complete") return <div>Judge selected.</div>;
      const candidates = args?.candidates ?? [];
      return (
        <ul>
          {candidates.map((j) => (
            <li key={j.id}>
              <strong>{j.name}</strong> — {j.goal}
              <button onClick={() => respond?.({ id: j.id, name: j.name })}>
                Select
              </button>
            </li>
          ))}
        </ul>
      );
    },
  });

  return <CopilotChat />;
}
```

```tsx
// app/layout.tsx
import { CopilotKit } from "@copilotkit/react-core";

export default function RootLayout({ children }) {
  return (
    <CopilotKit runtimeUrl="/api/copilotkit" agent="evaluator">
      {children}
    </CopilotKit>
  );
}
```

```ts
// app/api/copilotkit/route.ts
import {
  CopilotRuntime,
  ExperimentalEmptyAdapter,
  copilotRuntimeNextJSAppRouterEndpoint,
} from "@copilotkit/runtime";
import { LangGraphHttpAgent } from "@copilotkit/runtime/langgraph";

const runtime = new CopilotRuntime({
  agents: {
    evaluator: new LangGraphHttpAgent({
      url: process.env.EVALUATOR_BACKEND_URL ?? "http://127.0.0.1:8123/evaluator",
    }),
  },
});

export const POST = async (req) => {
  const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
    runtime,
    serviceAdapter: new ExperimentalEmptyAdapter(),
    endpoint: "/api/copilotkit",
  });
  return handleRequest(req);
};
```

## Backend wiring

```python
# server.py
from fastapi import FastAPI
from ag_ui_langgraph import add_langgraph_fastapi_endpoint
from copilotkit import LangGraphAGUIAgent

from agents.evaluator_agent import evaluator_graph
from agents.investigator_agent import investigator_graph

app = FastAPI()

add_langgraph_fastapi_endpoint(
    app,
    agent=LangGraphAGUIAgent(name="evaluator", graph=evaluator_graph),
    path="/evaluator",
)
add_langgraph_fastapi_endpoint(
    app,
    agent=LangGraphAGUIAgent(name="investigator", graph=investigator_graph),
    path="/investigator",
)
```

Run with `uvicorn server:app --reload --port 8123`.

## Running the sample end-to-end

```bash
# 1. Backend
cd samples/copilotkit/app/backend
pip install -r requirements.lock
export LAYERLENS_STRATIX_API_KEY=...
export OPENAI_API_KEY=...
python server.py         # listens on http://127.0.0.1:8123

# 2. Frontend
cd samples/copilotkit/app/frontend
npm install
npm run dev              # listens on http://127.0.0.1:3000
```

Open `http://127.0.0.1:3000`, click **Evaluate my traces** (or type your
own request into the chat). The agent fetches your traces, fetches your
judges, renders the judge picker inline in the chat, waits for you to
choose, then runs the evaluations and summarises the results.

## Tests

```bash
# Mocked tests (no API keys needed)
pytest tests/test_samples_e2e.py -k copilotkit

# Live LLM test (real OpenAI-compatible endpoint exercises the agent
# end-to-end through the AG-UI FastAPI endpoint and asserts the
# documented tool-call sequence)
cp tests/.env.example tests/.env   # then fill in OPENAI_API_KEY (or
                                   # OPENROUTER_API_KEY for OpenRouter)
pytest tests/test_samples_e2e.py -k copilotkit -m live
```

The mocked tests exercise each backend tool with a patched Stratix
client and verify the graph imports cleanly with mocked heavy deps.

The live test (`test_copilotkit_evaluator_live_llm`) drives the real
LLM through the FastAPI endpoint and asserts:

- The agent calls `list_recent_traces` -> `list_judges` ->
  `confirm_judge` in order.
- It pauses at `confirm_judge` (doesn't proceed to
  `run_trace_evaluation` until the frontend resolves it).
- Run lifecycle is clean: one `RUN_STARTED`, one `RUN_FINISHED` with
  matching `runId`, no `RUN_ERROR`.

The test loads credentials from a gitignored `.env` so local devs and
CI both work without leaking keys. Skipped when no key is available.

Playwright tests under `samples/copilotkit/app/frontend/tests/` are
for manual verification in a real Chrome (Playwright cannot reliably
drive CopilotChat's textarea in headless mode — separate testability
issue tracked at
[`CopilotKit/CopilotKit#4215`](https://github.com/CopilotKit/CopilotKit/issues/4215)).

## Upstream issues (informational)

Two bugs in `ag-ui-langgraph`'s `interrupt()` code path — not exercised
by this sample but still worth being aware of if you build a custom
`StateGraph` that uses `langgraph.types.interrupt()`:

- [`ag-ui-protocol/ag-ui#1582`](https://github.com/ag-ui-protocol/ag-ui/issues/1582):
  `RUN_FINISHED` emitted with LangGraph's internal `run_id` instead of
  the client-supplied one.
- [`ag-ui-protocol/ag-ui#1584`](https://github.com/ag-ui-protocol/ag-ui/issues/1584):
  Duplicate `RUN_STARTED` on the `has_active_interrupts` re-entry path.

Both manifest in the browser as `RUN_ERROR / INCOMPLETE_STREAM`. The
pattern in this sample (LLM + frontend tool, no `interrupt()`) avoids
them by construction.
