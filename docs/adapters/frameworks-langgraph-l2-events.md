# LangGraph L2 `agent.code` events

The LangGraph adapter emits one L2 `AgentCodeEvent` (`agent.code`) per
node execution, in addition to the cross-cutting `agent.state.change`
already emitted on state mutations. This closes the depth-audit gap
flagged in `A:/tmp/adapter-depth-audit.md` §1.3 and satisfies the
mapping demanded by spec
[`04a-langgraph-adapter-spec.md`](../incubation-docs/adapter-framework/04-per-framework-specs/04a-langgraph-adapter-spec.md)
§3 mapping table line 57:

| LangGraph Concept | Event Type | Layer |
|-------------------|------------|-------|
| Node execution (each node in the graph) | `AgentCodeEvent` | L2 |

Per-node L2 emission enables artifact attestation, replay correlation
(deterministic identity hashes), and per-node compliance reporting that
were previously impossible because node executions were captured only
as state diffs.

---

## When `agent.code` fires

Both adapter entry points emit one `agent.code` per node execution:

| Entry point | Trigger | Path |
|-------------|---------|------|
| `NodeTracer.decorate(fn)` / `trace_node()` decorator | Per call into the wrapped node function | `nodes.py::NodeTracer.on_node_exit` |
| `NodeTracer.trace_node(name, state)` context manager | Per `__exit__` of the context | `nodes.py::NodeTracer.on_node_exit` |
| `LayerLensLangGraphAdapter.on_node_start` / `on_node_end` | Per call pair from a custom orchestrator (e.g. `HandoffDetector` driver) | `lifecycle.py::LayerLensLangGraphAdapter._emit_node_agent_code` |

The emission is **unconditional with respect to outcome** — fires on
both the success path and the error path. State mutation is irrelevant
to L2 emission; the cross-cutting `agent.state.change` event is the
state-aware companion event.

`agent.code` is gated by `CaptureConfig.l2_agent_code` (default:
`False`). To enable in production:

```python
from layerlens.instrument.adapters._base import CaptureConfig
from layerlens.instrument.adapters.frameworks.langgraph import LayerLensLangGraphAdapter

adapter = LayerLensLangGraphAdapter(
    capture_config=CaptureConfig(l2_agent_code=True),
)
```

---

## Schema

The L2 envelope follows the canonical
[`AgentCodeEvent` schema](../incubation-docs/adapter-framework/05-trace-schema-specification.md#3-agentcode-agentcodeevent):

```json
{
  "event_type": "agent.code",
  "layer": "L2",
  "code": {
    "repo": "myapp.agents.fundamental_analyst",
    "commit": "runtime",
    "artifact_hash": "sha256:c4b6...e2f1",
    "config_hash": "sha256:71d3...90af",
    "branch": null,
    "tag": null,
    "build_info": {
      "node_name": "fundamental_analyst",
      "node_qualname": "FundamentalAnalyst.analyze",
      "execution_duration_ns": 142000000,
      "input_state_keys": ["messages", "ticker"],
      "output_state_keys": ["analysis", "messages", "ticker"],
      "status": "success"
    }
  }
}
```

### Field semantics

| Field | Source | Notes |
|-------|--------|-------|
| `repo` | `node_callable.__module__` | Falls back to `"layerlens.instrument:runtime"` for module-less callables (lambdas, REPL). |
| `commit` | Always `"runtime"` at the adapter layer | The platform's trace finalizer / attestation layer overwrites this with the real git commit when one is resolvable. |
| `artifact_hash` | SHA-256 of `node_callable.__code__.co_code` (preferred), or source text, or `__qualname__` (last-resort fallback) | **Deterministic** — repeated executions of the same callable yield the same hash. Replay diff engines correlate per-node artifacts via this value. |
| `config_hash` | SHA-256 of `\|`-joined descriptor: `node_name`, `qualname`, signature, status, sorted input/output key names | Includes only key NAMES, never values — PII-safe. |
| `build_info.node_name` | Adapter-supplied node name | Free-form. |
| `build_info.node_qualname` | `node_callable.__qualname__` | Stable across invocations of the same callable. |
| `build_info.execution_duration_ns` | `time.time_ns()` delta between `on_node_enter` and `on_node_exit` | Wall-clock duration. |
| `build_info.input_state_keys` | Top-level keys of the input state (sorted) | Names only. |
| `build_info.output_state_keys` | Top-level keys of the output state (sorted) | Names only. |
| `build_info.status` | `"success"` or `"error"` | `"error"` when the node raised. |
| `build_info.error_class` | `type(exception).__name__`, present only on error path | E.g. `"RuntimeError"`. |
| `build_info.error_truncated` | `True` if the exception's `str()` exceeded 512 bytes and was truncated | Defensive bound on the L2 envelope size. |

---

## PII safety

The L2 envelope **never carries state values** — only key NAMES. This
is enforced at three points:

1. `_safe_state_keys(state)` extracts top-level keys and discards
   values before they ever enter the event constructor.
2. `config_hash` is computed over a descriptor that explicitly names
   only key NAMES, never values.
3. `build_info.input_state_keys` / `output_state_keys` are lists of
   strings (never dicts).

A regression test
(`tests/instrument/adapters/frameworks/langgraph/test_l2_agent_code_event.py::TestPiiSafety`)
asserts that a known-secret value placed in node input state does not
appear anywhere in the serialized L2 envelope.

Error messages on the error path are bounded to 512 UTF-8 bytes and
flagged via `build_info.error_truncated` so downstream sinks can
distinguish a clean error class signal from a partially-redacted
message.

---

## Replay correlation

Replay diff engines need to identify which two events correspond to the
same node executed at two different points in time (original vs
replay). The `artifact_hash` field is the join key for this:

```text
original.events[i].artifact_hash == replay.events[j].artifact_hash
  ⇒ both events represent the same node callable
```

Determinism is achieved by hashing the CPython bytecode of the wrapped
function. `functools.wraps` exposes the original via `__wrapped__`, so
decorated nodes hash to the same value as their undecorated form.
Source-text fallback applies when bytecode is unavailable; qualified
name is the final fallback for callables with neither bytecode nor
inspectable source.

Test coverage:
`TestArtifactHashDeterminism::test_same_callable_yields_same_artifact_hash_across_executions`.

---

## Event count budget

For an ai-hedge-fund-style supervisor graph (5-10 specialist agents,
1-2 tickers), the per-trace event budget changes as follows when L2 is
enabled:

| Event Type | Before this PR | With L2 enabled | Delta |
|------------|---------------|-----------------|-------|
| `agent.code` | 0 | 7-10 (matches node count) | **+7-10** |
| `agent.state.change` | 8-15 | 8-15 (unchanged) | 0 |
| All other types | unchanged | unchanged | 0 |

L2 events are fixed-size (~1 KB each in JSON) and sit alongside
existing emissions, so the storage delta is bounded by the node count.

---

## Spec / story coverage

This wiring closes the following spec gap:

- Spec `04a` table line 57: `Node execution → AgentCodeEvent (L2)` —
  previously emitted only as `agent.state.change`, now correctly
  emitted as both.
- Spec `04a` §7 step 2 walkthrough: `AgentCodeEvent (L2) — supervisor
  node execution with artifact hash` — now actually fires.
- Spec `04a` §7 step 3 walkthrough: `AgentCodeEvent (L2) — node
  execution metadata` — now actually fires.
- FEA-1902 user story: "Capture node executions" — now resolves to
  PASS instead of PARTIAL.

The depth audit row that previously read **GAP** (line 181) for
`L2 AgentCodeEvent for nodes` resolves to **PASS** with this PR.
