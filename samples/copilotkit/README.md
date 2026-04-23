# CopilotKit Samples

Building AI-powered user interfaces requires more than backend evaluation logic -- it requires
real-time feedback loops between the AI backend and the frontend. These samples provide
CopilotKit CoAgents (LangGraph-based) and React components that connect LayerLens evaluation
capabilities to interactive UIs, enabling human-in-the-loop evaluation workflows where users
can review, confirm, and act on AI quality assessments in real time.

## Prerequisites

```bash
pip install layerlens --index-url https://sdk.layerlens.ai/package \
    copilotkit langgraph pydantic mcp ag-ui-langgraph fastapi uvicorn
npm install @copilotkit/react-core @copilotkit/runtime
export LAYERLENS_STRATIX_API_KEY=your-api-key
```

### Version compatibility

The samples are pinned against the matrix below. Later versions often work but are not tested.

| Package | Version |
|---|---|
| `layerlens` | `>=1.6.0` |
| `langgraph` | `>=1.1.0` |
| `ag-ui-langgraph` | `>=0.0.34` |
| `copilotkit` (python) | `>=0.1.87` |
| `@copilotkit/react-core` | `>=1.56.3` |
| `@copilotkit/runtime` | `>=1.56.3` |
| Python | `>=3.12` |
| Next.js | `>=15` |

## Quick Start

Start with the Evaluator Agent to see the core human-in-the-loop pattern:

```bash
python agents/evaluator_agent.py
```

Expected output: the agent parses an evaluation intent, selects appropriate judges, pauses
for human confirmation, executes the evaluation, and emits AG-UI protocol events at each step.

## Agents (LangGraph CoAgents)

| File | Scenario | Description |
|------|----------|-------------|
| `agents/evaluator_agent.py` | Product teams building evaluation dashboards | A multi-step evaluation workflow: parses user intent, selects judges, requests human confirmation before execution, runs evaluations, and summarizes results. Uses LangGraph `interrupt()` for the confirmation step (**requires a checkpointer -- see below**). |
| `agents/investigator_agent.py` | Operations teams building trace debugging UIs | Fetches a trace by ID, analyzes spans for errors, latency anomalies, and cost outliers, then generates actionable fix suggestions. No `interrupt()`, no checkpointer required. |

## Human-in-the-loop: checkpointers

### Why a checkpointer is mandatory

`evaluator_agent.py` uses LangGraph's `interrupt()` to pause execution while the user
confirms which judge to run. LangGraph implements `interrupt()` by persisting the graph's
state to a **checkpointer** at the pause boundary and resuming from that checkpoint when
`Command(resume=...)` is sent. **Without a checkpointer there is nothing to resume from.**

Concretely, if you compile with `build_graph().compile()` instead of
`build_graph().compile(checkpointer=...)`:

1. The first run emits `__interrupt__` and stops. Fine so far.
2. The client sends the user's confirmation, which `ag-ui-langgraph` translates into
   `graph.astream(Command(resume="..."), config={...})`.
3. Because no checkpoint exists for this `thread_id`, the resumed stream produces **zero
   events**. LangGraph has no record of the paused state.
4. `ag-ui-langgraph` never observes a terminal state on the resume stream, so it never
   emits `RUN_FINISHED` to the AG-UI wire protocol.
5. The CopilotKit frontend's protocol state machine is stuck. The next user message fails:

   ```
   Cannot send 'RUN_STARTED' while a run is still active.
   The previous run must be finished with 'RUN_FINISHED' before starting a new run.
   ```

Any LangGraph graph that uses `interrupt()` has this requirement. The sample code would
behave identically without CopilotKit in front of it -- the resume would just silently
return nothing.

### Checkpointer options matrix

| Checkpointer | Durable? | Install | Use case |
|---|---|---|---|
| `InMemorySaver` | No -- lost on restart | built into `langgraph` | Local demos, tests, this sample's default |
| `SqliteSaver` | Single-node file | `pip install langgraph-checkpoint-sqlite` | Single-process apps, dev deployments |
| `PostgresSaver` | Yes, horizontally scalable | `pip install langgraph-checkpoint-postgres` | Production deployments |
| `AsyncRedisSaver` | Yes, with TTL | `pip install langgraph-checkpoint-redis` | Ephemeral sessions, existing Redis infra |
| LangGraph Platform | Managed | hosted by LangChain | No-ops checkpointer management |

### Migrating from `InMemorySaver` to production

The sample ships with:

```python
# samples/copilotkit/agents/evaluator_agent.py
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
evaluator_graph = build_graph().compile(checkpointer=checkpointer)
```

For production, replace that block with one of the following:

**Postgres** (recommended for multi-instance deployments):

```python
# pip install langgraph-checkpoint-postgres
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = os.environ["LANGGRAPH_CHECKPOINT_DB_URI"]
# e.g. "postgresql://user:pass@host:5432/langgraph?sslmode=require"
checkpointer = PostgresSaver.from_conn_string(DB_URI)
checkpointer.setup()  # one-time: creates the checkpoint tables
evaluator_graph = build_graph().compile(checkpointer=checkpointer)
```

For fully async code paths use `AsyncPostgresSaver` from
`langgraph.checkpoint.postgres.aio` instead; its API is identical except `setup()` and
`from_conn_string()` are awaitable.

**SQLite** (single-process, file-backed):

```python
# pip install langgraph-checkpoint-sqlite
from langgraph.checkpoint.sqlite import SqliteSaver

checkpointer = SqliteSaver.from_conn_string("checkpoints.sqlite")
evaluator_graph = build_graph().compile(checkpointer=checkpointer)
```

**Redis** (TTL-friendly):

```python
# pip install langgraph-checkpoint-redis
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

checkpointer = AsyncRedisSaver(redis_url=os.environ["REDIS_URL"])
evaluator_graph = build_graph().compile(checkpointer=checkpointer)
```

**LangGraph Platform** does not require passing a checkpointer to `.compile()` -- the
platform wires one in automatically when you deploy the graph.

### Pydantic, not dataclasses, in state

Graph state that crosses the checkpoint boundary must be serializable. LangGraph's default
`JsonPlusSerializer` handles Pydantic models, `TypedDict`, primitives, and LangChain
message types natively. It does **not** handle plain `@dataclass` types -- they raise
`TypeError: Type is not msgpack serializable` during checkpointing.

`evaluator_agent.py` uses `pydantic.BaseModel` for `JudgeInfo`, `TraceInfo`, and
`EvaluationInfo` for that reason. If you add your own state fields, follow the same
pattern or convert to `dict` before storing.

## Backend wiring (FastAPI + ag-ui-langgraph)

```python
# server.py
from fastapi import FastAPI
from ag_ui_langgraph import LangGraphAgent, add_langgraph_fastapi_endpoint

from agents.evaluator_agent import evaluator_graph
from agents.investigator_agent import investigator_graph

app = FastAPI()

# The checkpointer compiled into each graph is what powers interrupt/resume.
add_langgraph_fastapi_endpoint(
    app,
    agent=LangGraphAgent(name="evaluator", graph=evaluator_graph),
    path="/evaluator",
)
add_langgraph_fastapi_endpoint(
    app,
    agent=LangGraphAgent(name="investigator", graph=investigator_graph),
    path="/investigator",
)

# uvicorn server:app --reload --port 8000
```

Each POST to `/evaluator` is a `RunAgentInput` carrying `threadId`, `runId`, `messages`,
and `state`. The adapter reuses the checkpointer to look up prior state for the same
`threadId`, streams AG-UI protocol events (RUN_STARTED, STEP_STARTED/FINISHED,
STATE_SNAPSHOT, MESSAGES_SNAPSHOT, CUSTOM, RUN_FINISHED) as SSE, and terminates the stream
on interrupt or completion. A subsequent POST with the same `threadId` is treated as a
resume.

## Frontend wiring (Next.js + CopilotKit runtime)

```ts
// app/api/copilotkit/route.ts
import { CopilotRuntime, copilotRuntimeNextJSAppRouterEndpoint } from "@copilotkit/runtime";
import { LangGraphHttpAgent } from "@copilotkit/runtime";

const runtime = new CopilotRuntime({
  agents: {
    evaluator: new LangGraphHttpAgent({ url: "http://localhost:8000/evaluator" }),
    investigator: new LangGraphHttpAgent({ url: "http://localhost:8000/investigator" }),
  },
});

export const { POST } = copilotRuntimeNextJSAppRouterEndpoint({
  runtime,
  endpoint: "/api/copilotkit",
});
```

```tsx
// app/layout.tsx
import { CopilotKit } from "@copilotkit/react-core";

export default function RootLayout({ children }) {
  return (
    <html>
      <body>
        <CopilotKit runtimeUrl="/api/copilotkit" agent="evaluator">
          {children}
        </CopilotKit>
      </body>
    </html>
  );
}
```

The `agent` prop tells CopilotKit which registered LangGraph agent to drive. The runtime
translates chat messages into AG-UI events and relays events from the Python backend back
to the browser over SSE.

## React Components

| File | Description |
|------|-------------|
| `components/EvaluationCard.tsx` | Renders evaluation results with score breakdowns and judge verdicts. |
| `components/TraceCard.tsx` | Displays trace metadata, span hierarchy, and timing information. |
| `components/JudgeVerdictCard.tsx` | Shows individual judge verdicts with pass/fail indicators. |
| `components/MetricCard.tsx` | Renders a single metric with trend visualization. |
| `components/ComplianceCard.tsx` | Displays compliance status with regulation-specific details. |
| `components/index.ts` | Barrel export for all components. |

## React Hooks

| File | Description |
|------|-------------|
| `hooks/useLayerLensActions.ts` | CopilotKit action hooks for triggering evaluations and investigations. |
| `hooks/useLayerLensContext.ts` | Context hook for sharing LayerLens state across components. |
| `hooks/index.ts` | Barrel export for all hooks. |

## Architecture

```
CopilotKit Frontend (React)
    |
    v  (AG-UI events over SSE)
@copilotkit/runtime (Next.js route handler)
    |
    v  (LangGraphHttpAgent -> HTTP POST)
ag-ui-langgraph FastAPI endpoint
    |
    v  (graph.astream with checkpointer-backed thread_id)
LangGraph StateGraph (evaluator_graph / investigator_graph)
    |
    v
LayerLens Python SDK (Stratix client)
    |
    v
LayerLens API
```

The agents emit AG-UI protocol events that CopilotKit renders as progress cards,
confirmation dialogs, and result summaries in the frontend. The React components and
hooks are provided as reference implementations for building your own LayerLens-powered UI.

## Troubleshooting

### `Cannot send 'RUN_STARTED' while a run is still active`

The most common cause is compiling an `interrupt()`-using graph without a checkpointer.
Verify `evaluator_graph.checkpointer` is not `None`:

```python
from agents.evaluator_agent import evaluator_graph
assert evaluator_graph.checkpointer is not None
```

If your custom graph compiles to `None`, see the "Migrating from `InMemorySaver`" section
above.

### `Type is not msgpack serializable`

Your graph state contains a plain `@dataclass` or other non-serializable type. Convert
nested DTOs to `pydantic.BaseModel` (or `dict`). See the Pydantic section above.

### `Deserializing unregistered type` warning

LangGraph is hardening type registration for checkpoint reloads across processes. For the
in-memory sample this warning is informational. In production, register your DTOs via
LangGraph's `allowed_msgpack_modules` setting or set
`LANGGRAPH_STRICT_MSGPACK=false` in dev. See the LangGraph docs for the per-version
specifics.
