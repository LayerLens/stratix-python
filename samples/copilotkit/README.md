# CopilotKit Samples — LayerLens Stratix

Production-shaped reference for building CopilotKit-powered UIs on top of
LayerLens Stratix. Two agents ship here:

| Agent | Purpose |
|---|---|
| `agents/evaluator_agent.py` | Multi-step evaluation with human-in-the-loop judge selection. Drives the demo app at `app/`. |
| `agents/investigator_agent.py` | Trace investigation (errors, latency, cost hot spots). No HITL. |

The evaluator is the showcase. It uses CopilotKit's canonical
[`coagents-research-canvas`](https://github.com/CopilotKit/CopilotKit/tree/main/examples/v1/research-canvas)
pattern: an LLM-driven agent with backend tools, a state-driven canvas
on the host page, a chat sidebar with HITL widgets, and out-of-band
polling for long-running async work.

## Layout

```
samples/copilotkit/
├── agents/
│   ├── evaluator_agent.py       # the agent (LangChain v1 create_agent)
│   └── investigator_agent.py
├── components/                  # SDK card library — 5 cards + MarkdownLite
│   ├── EvaluationCard.tsx
│   ├── TraceCard.tsx
│   ├── JudgeVerdictCard.tsx
│   ├── MetricCard.tsx
│   ├── ComplianceCard.tsx
│   └── markdown-lite.tsx
├── app/                         # the customer-facing sample
│   ├── backend/                 # FastAPI server + AG-UI endpoint
│   │   └── server.py
│   └── frontend/                # Next.js 16 + shadcn/ui
│       ├── app/                 # page.tsx, layout.tsx, theme-toggle.tsx, route.ts
│       ├── components/ui/       # shadcn primitives (npx shadcn add ...)
│       └── lib/utils.ts         # cn()
└── README.md                    # this file
```

## Prerequisites

```bash
# Python deps
pip install layerlens --index-url https://sdk.layerlens.ai/package
pip install -r samples/copilotkit/app/backend/requirements.lock

# Frontend deps (already shrinkwrapped via package-lock.json)
cd samples/copilotkit/app/frontend && npm install

# Credentials — real LayerLens Stratix only; no fake fixture path
export LAYERLENS_STRATIX_API_KEY=your-stratix-key
export OPENAI_API_KEY=your-openai-key
# optional — point at OpenRouter, Ollama, LM Studio, etc.
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
export OPENAI_MODEL=openai/gpt-4o-mini
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
| `@copilotkit/react-core` | `==1.56.3` |
| `@copilotkit/react-ui` | `==1.56.3` |
| `@copilotkit/runtime` | `==1.56.3` |
| `next` | `^16` (Tailwind 4) |
| `react` | `^19` |
| `radix-ui`, `class-variance-authority`, `clsx`, `tailwind-merge`, `tw-animate-css` | shadcn/ui foundation |
| Python | `>=3.10,<3.13` |

## Architecture

```
+-- Browser (Next.js + shadcn/ui + CopilotKit) --------------------+
|                                                                  |
|  +-- Canvas (host page) -----+   +-- Chat sidebar -------------+ |
|  |  useCoAgent({name})       |   |  <CopilotChat>              | |
|  |  state.{traces,judges,    |   |    JudgePicker              | |
|  |        evaluations,results}|   |      useCopilotAction       | |
|  |  Live cards from state:   |   |        renderAndWaitFor...  | |
|  |    MetricCard × 4         |   |    Theme toggle, send btn   | |
|  |    TraceCard × N          |   +-----------------------------+ |
|  |    EvaluationCard summary |                                  |
|  |    JudgeVerdictCard × N   |   Polls /evaluations/{id}        |
|  |    PendingVerdictCard × M |   every 5s for in-flight evals.  |
|  +---------------------------+                                  |
|                                                                  |
+--- /api/copilotkit (CopilotRuntime + LangGraphHttpAgent) --------+
                |
                v
+-- FastAPI backend (samples/copilotkit/app/backend/server.py) ---+
|                                                                 |
|  POST /evaluator                                                |
|    ag_ui_langgraph.add_langgraph_fastapi_endpoint               |
|        wraps the graph in build_agui_agent (recursion_limit=200)|
|                                                                 |
|  GET  /evaluations/{id}                                         |
|    Out-of-band polling endpoint — returns the current verdict   |
|    (status / passed / score / reasoning / trace_id / judge_id). |
|                                                                 |
|  langchain.agents.create_agent (state_schema=EvaluatorState)    |
|    - model: ChatOpenAI (OpenAI-compatible)                      |
|    - tools (async, return Command(update={...})):               |
|        list_recent_traces                                       |
|        list_judges                                              |
|        run_trace_evaluation                                     |
|        get_evaluation_result                                    |
|    - middleware: [CopilotKitMiddleware]                         |
|    - confirm_judge bridged in from the frontend (no-arg HITL)   |
|                                                                 |
|  Tools call copilotkit_emit_state to push state mid-run so the  |
|  canvas updates while the agent works.                          |
|                                                                 |
|  layerlens.Stratix client                                       |
+-----------------------------------------------------------------+
```

### How the demo flows

1. User clicks **Evaluate my traces** in the header. The page sends
   *"Please evaluate my recent traces."* into the chat.
2. Agent calls `list_recent_traces` → state.traces populates →
   `MetricCard` and `TraceCard`s appear on the canvas.
3. Agent calls `list_judges` → state.judges populates → "Available
   judges" section appears with markdown-rendered goals.
4. Agent calls `confirm_judge` (no args). The `JudgePicker` in the
   chat reads candidates from `useCoAgent` state and renders an
   inline picker. User selects a judge.
5. Agent calls `run_trace_evaluation` for the first 5 traces.
   `EvaluationCard` summary appears on the canvas; one
   `PendingVerdictCard` per evaluation in the verdicts grid.
6. Agent ends the run with one short summary line in the chat
   ("Started 5 evaluations against \<judge\>. They will appear in the
   canvas as they finish.").
7. Frontend polls `/evaluations/{id}` every 5 s for each pending
   evaluation. As verdicts land, `PendingVerdictCard`s flip to real
   `JudgeVerdictCard`s with markdown-rendered reasoning.

## Why this pattern

We deliberately avoid `langgraph.types.interrupt()` for HITL. That
path hits two upstream bugs in `ag-ui-langgraph` — see the
[upstream issues](#upstream-issues-informational) at the bottom — and
isn't the pattern CopilotKit's own samples ship today.

The pattern here matches `coagents-research-canvas`:

- **Stateful canvas + thin chat.** Cards live on the host page, driven
  by `useCoAgent({name: "evaluator"})`. The chat is reserved for
  conversation and HITL widgets. Same split research-canvas uses.
- **Frontend-defined HITL via `useCopilotAction({ available: "remote",
  renderAndWaitForResponse })`.** The picker is a real React widget,
  not a text prompt.
- **Out-of-band polling for slow async work.** Real LayerLens
  evaluations take 30+ seconds — way too long to block a chat turn.
  The agent kicks them off and exits; the frontend polls the
  backend's `/evaluations/{id}` endpoint and folds results into the
  canvas as each completes. No LLM hallucination about state it can't
  observe.
- **`copilotkit_emit_state` for live updates.** Tools are `async` and
  call `copilotkit_emit_state(config, ...)` after each LayerLens API
  call so STATE_SNAPSHOT events stream to the browser mid-run.

## Agents

### `evaluator_agent.py`

`langchain.agents.create_agent` with `CopilotKitMiddleware`, four
backend `@tool` functions, and an extended `EvaluatorState` schema.

State schema (`EvaluatorState(AgentState)`):

```python
traces: list[dict[str, Any]]                                  # from list_recent_traces
judges: list[dict[str, Any]]                                  # from list_judges
evaluations: Annotated[list[dict[str, Any]], operator.add]    # one per run_trace_evaluation
results: Annotated[list[dict[str, Any]], operator.add]        # one per get_evaluation_result success
```

Tools (all `async`, all return `langgraph.types.Command`):

- `list_recent_traces(limit=20) -> Command(update={"traces", "messages"})`
- `list_judges() -> Command(update={"judges", "messages"})`
- `run_trace_evaluation(trace_id, judge_id) -> Command(update={"evaluations", "messages"})`
- `get_evaluation_result(evaluation_id) -> Command(update={"results"?, "messages"})`
- `confirm_judge()` — **frontend tool, zero arguments**. The picker reads
  candidates from `state.judges` directly so we don't have to stream a
  large candidate list through tool args (which used to fail with
  `tool_argument_parse_failed: Unterminated string in JSON` when the
  LLM truncated mid-stream).

Each tool calls `copilotkit_emit_state(config, ...)` after its
LayerLens API call so the canvas updates immediately, not at run-end.

### `investigator_agent.py`

A procedural `StateGraph` that fetches a trace, extracts events, and
flags errors / slow spans / high token usage. No HITL, no LLM. Good
reference for a non-conversational agent that still plugs into the
CopilotKit runtime.

## Frontend

### Foundation

The sample is a Next.js 16 + Tailwind 4 + shadcn/ui app. Primitives
were generated via:

```bash
cd samples/copilotkit/app/frontend
npx shadcn@latest add card button badge progress separator
```

`globals.css` ships the stock shadcn neutral OKLCH tokens
(`--background`, `--foreground`, `--card`, `--muted`, `--destructive`,
…) plus a `--copilot-kit-*` bridge that maps the chat widget's
palette onto the same shadcn tokens — so `<CopilotChat>` feels native
to the host page in both light and dark mode.

A Tailwind 4 `@source` directive scans
`samples/copilotkit/components/**/*.{ts,tsx}` so utility classes used
inside the SDK card library make it into the generated CSS.

### Theme

Light by default (matches every official CopilotKit sample). A
segmented `ThemeToggle` (Light / System / Dark) lives in the header
and persists to `localStorage`. System mode tracks the OS via
`matchMedia("(prefers-color-scheme: dark)")`.

### SDK card library (`samples/copilotkit/components/`)

| Card | Use |
|---|---|
| `MetricCard` | KPI tile with optional unit + trend chip. Used in the canvas's top strip. |
| `TraceCard` | Per-trace summary (framework / duration / tokens / cost / events). `dashboardBaseUrl` opt-in for "Trace Explorer" / "Agent Graph" links. |
| `EvaluationCard` | Run summary with pass-rate hero, case bar, score distribution, sparkline. |
| `JudgeVerdictCard` | Per-result verdict with severity pill + filled Pass/Fail/Error badge + markdown-rendered reasoning. |
| `ComplianceCard` | Compliance framework status with attestations + violations. (Showcase; not in the evaluator demo.) |
| `MarkdownLite` | Tiny safe inline-markdown renderer — paragraph breaks, line breaks, `**bold**`, `*italic*`. Used by `JudgeVerdictCard.reasoning`, the `JudgePicker` goal text, and the canvas's "Available judges" card. No `react-markdown` dep — the SDK card library has to stay resolvable without the host app's `node_modules` in scope. |

Brand accent (`#6766FC`) is applied via Tailwind class strings on
CTAs only — never by editing `--primary`. Same approach
`research-canvas` takes for its `#6766FC` indigo.

### `app/page.tsx` highlights

```tsx
import { useCoAgent, useCopilotAction, useCopilotChat } from "@copilotkit/react-core";
import {
  EvaluationCard, JudgeVerdictCard, MarkdownLite,
  MetricCard, TraceCard,
} from "@layerlens/copilotkit-cards";
import { ThemeToggle } from "./theme-toggle";

export default function Page() {
  const { state } = useCoAgent<EvaluatorState>({
    name: "evaluator",
    initialState: { traces: [], judges: [], evaluations: [], results: [] },
  });

  // Out-of-band polling: any evaluation in state.evaluations not yet
  // in state.results gets polled every 5s; results merge into the
  // canvas as each completes.
  const [polledResults, setPolledResults] = useState<ResultRecord[]>([]);
  useEffect(() => { /* fetch /evaluations/{id} for pending IDs */ }, [...]);

  const results = useMemo(() => mergeResults(state.results, polledResults), ...);

  // HITL: no-arg confirm_judge tool, picker reads candidates from state.judges
  useCopilotAction({
    name: "confirm_judge",
    parameters: [],
    renderAndWaitForResponse: ({ respond, status }) => (
      <JudgePicker respond={respond} status={status} />
    ),
  });

  return (
    <main className="flex h-screen flex-col bg-background text-foreground">
      <header>{/* title + ThemeToggle + Evaluate button */}</header>
      <div className="flex flex-1">
        <section className="flex-1 overflow-y-auto p-6">{/* canvas: cards from state */}</section>
        <aside className="w-[420px] border-l">
          <CopilotChat />
        </aside>
      </div>
    </main>
  );
}
```

### `app/layout.tsx`

Wraps `<CopilotKit runtimeUrl="/api/copilotkit" agent="evaluator"
showDevConsole={false} enableInspector={false}>`. Theme is hydrated
from `localStorage` on mount via `ThemeToggle`.

### `app/api/copilotkit/route.ts`

```ts
new CopilotRuntime({
  agents: {
    evaluator: new LangGraphHttpAgent({
      url: process.env.EVALUATOR_BACKEND_URL ?? "http://127.0.0.1:8123/evaluator",
    }),
  },
});
```

## Backend

```python
# app/backend/server.py
from fastapi import FastAPI
from ag_ui_langgraph import add_langgraph_fastapi_endpoint
from agents.evaluator_agent import build_agui_agent, evaluator_graph

app = FastAPI()

@app.get("/evaluations/{evaluation_id}")
def get_evaluation(evaluation_id: str) -> dict:
    """Out-of-band polling for the frontend."""
    ...

add_langgraph_fastapi_endpoint(
    app,
    agent=build_agui_agent(name="evaluator", graph=evaluator_graph),
    path="/evaluator",
)
```

`build_agui_agent` constructs a `LangGraphAGUIAgent` with
`config={"recursion_limit": 200}` (the default 25 isn't enough for a
5-trace fan-out plus polls) and a small `_RunIdPreserving` subclass
that workarounds [`ag-ui-protocol/ag-ui#1582`](https://github.com/ag-ui-protocol/ag-ui/issues/1582).

### Real LayerLens only

Missing or empty `LAYERLENS_STRATIX_API_KEY` is a hard `RuntimeError`
at startup. There is no fake/fixture mode. This sample exists to
exercise real LayerLens; a fake mode would defeat the demo.

## Running the sample end-to-end

```bash
# Terminal 1 — backend
cd samples/copilotkit/app/backend
pip install -r requirements.lock
export LAYERLENS_STRATIX_API_KEY=...
export OPENAI_API_KEY=...
python server.py         # http://127.0.0.1:8123

# Terminal 2 — frontend
cd samples/copilotkit/app/frontend
npm install
npm run dev              # http://127.0.0.1:3000
```

Open `http://127.0.0.1:3000`, click **Evaluate my traces** (or type
*"Please evaluate my recent traces."* in chat). Watch the canvas
populate progressively as each tool runs and as verdicts complete.

## Tests

```bash
pytest tests/test_samples_e2e.py -k copilotkit
```

The mocked tests exercise each backend tool with a patched Stratix
client and verify the graph imports cleanly with mocked heavy deps.

Playwright tests under `samples/copilotkit/app/frontend/tests/` are
for manual verification in real Chrome (Playwright cannot reliably
drive `<CopilotChat>`'s textarea in headless mode — separate
testability issue tracked at
[`CopilotKit/CopilotKit#4215`](https://github.com/CopilotKit/CopilotKit/issues/4215)).

## Upstream issues (informational)

- [`ag-ui-protocol/ag-ui#1582`](https://github.com/ag-ui-protocol/ag-ui/issues/1582):
  `RUN_FINISHED` emitted with LangGraph's internal `run_id` instead of
  the client-supplied one. Worked around by `_RunIdPreserving` in
  `evaluator_agent.py`.
- [`ag-ui-protocol/ag-ui#1584`](https://github.com/ag-ui-protocol/ag-ui/issues/1584):
  Duplicate `RUN_STARTED` on the `has_active_interrupts` re-entry path.
  Avoided by construction in this sample (no `interrupt()` call).

Both manifest in the browser as `RUN_ERROR` / `INCOMPLETE_STREAM` if
you take the `interrupt()` path. The pattern in this sample sidesteps
them.
