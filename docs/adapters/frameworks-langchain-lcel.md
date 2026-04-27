# LangChain LCEL Coverage

The LayerLens LangChain adapter instruments **LangChain Expression
Language (LCEL)** pipelines as a first-class observability surface.
This is the dominant authoring pattern in LangChain 0.2+ — pipelines
expressed via the `|` (pipe) operator over `Runnable` instances:

```python
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

LCEL composition produces a **tree of runnables** at runtime
(`RunnableSequence`, `RunnableParallel`, `RunnableLambda`,
`RunnablePassthrough`, `RunnableBranch`). The adapter tracks the
entire tree, emits per-runnable events with composition metadata, and
emits a synthetic `chain.composition` snapshot at root completion so
debuggers can see what was executed in one glance.

## Coverage matrix

| Runnable primitive             | Detected via                  | Per-step events                      | Composition metadata                                  |
| ------------------------------ | ----------------------------- | ------------------------------------ | ----------------------------------------------------- |
| `RunnableSequence`             | `name` starts with the string | `agent.input`, `agent.output`, `agent.code` | `kind=sequence`; child positions `seq:step:N`         |
| `RunnableParallel<a,b,...>`    | `name` starts with the string | `agent.input`, `agent.output`, `agent.code` | `kind=parallel`; declared branch keys parsed from name; child positions `map:key:K` |
| `RunnableLambda`               | `name == "RunnableLambda"`    | `agent.input`, `agent.output`, `agent.code` | `kind=lambda`; SHA-256 fingerprint over `(name, depth, position)` |
| `RunnablePassthrough`          | `name == "RunnablePassthrough"` | `agent.input`, `agent.output`, `agent.code` | `kind=passthrough`; payload carries `passthrough=true` |
| `RunnableBranch`               | `name` starts with the string | `agent.input`, `agent.output`, `agent.code` | `kind=branch`; child positions `condition:N` (predicates) and `branch:N` (bodies) |
| `RunnableConfig` (passthrough) | not a runnable; opaque kwargs | n/a — propagates run hierarchy via `parent_run_id` | tags + metadata forwarded into `agent.input` payloads |
| Non-LCEL runnable (prompts, parsers, models) | `name` does not match any LCEL prefix | Standard `model.invoke` / `tool.call` events fire as before; if observed under an LCEL parent, also emits `agent.code` with `kind=other` | parent linkage + composition position retained |

## Event reference

The adapter emits four LCEL-relevant event types:

### `agent.input` (L1) — per runnable start

Emitted when `on_chain_start` fires for an LCEL runnable.

```json
{
  "type": "agent.input",
  "payload": {
    "run_id": "...",
    "parent_run_id": "...",
    "runnable": {
      "kind": "lambda",
      "name": "RunnableLambda",
      "depth": 1,
      "position": {
        "parent_kind": "sequence",
        "label": "2",
        "role": "step"
      },
      "fingerprint": "cf97f2529fcb79e2"
    },
    "input": "<the input value passed to invoke()>"
  }
}
```

### `agent.output` (L1) — per runnable end

Symmetric with `agent.input`. The payload includes `duration_ns` and
`status` (`"ok"` or `"error"`). Errors carry a separate `error` field.

### `agent.code` (L2) — per runnable end (pipeline structure)

The L2 event the spec maps to LCEL pipeline structure (04b §3 & §4).
Emitted once per runnable in the executed tree, with per-runnable
metadata (kind, depth, position, optional passthrough/fingerprint
markers).

### `agent.code` (L2) with `kind="chain.composition"` — synthetic graph snapshot

Emitted **once per root runnable** when the root completes (success or
error). Carries the full subtree as a flat node list plus an
aggregate summary so the dashboard can render the executed DAG without
having to reconstruct it from the per-step stream:

```json
{
  "type": "agent.code",
  "payload": {
    "kind": "chain.composition",
    "composition": {
      "root_run_id": "...",
      "root_kind": "sequence",
      "root_name": "RunnableSequence",
      "node_count": 9,
      "max_depth": 2,
      "kind_counts": {
        "sequence": 1,
        "parallel": 1,
        "lambda": 1,
        "passthrough": 1,
        "branch": 1,
        "other": 4
      },
      "status": "ok",
      "nodes": [
        {
          "run_id": "...",
          "parent_run_id": null,
          "kind": "sequence",
          "name": "RunnableSequence",
          "depth": 0,
          "status": "ok",
          "duration_ns": 3092000,
          "child_run_ids": ["...", "...", "..."]
        }
        // ... one entry per runnable in the tree
      ]
    }
  }
}
```

## Capture-config gating

LCEL events follow the standard `CaptureConfig` layer model:

| Layer            | Field             | Default | Controls                                 |
| ---------------- | ----------------- | ------- | ---------------------------------------- |
| L1 (Agent I/O)   | `l1_agent_io`     | `True`  | Per-runnable `agent.input` / `agent.output` |
| L2 (Agent Code)  | `l2_agent_code`   | `False` | Per-runnable `agent.code` AND the synthetic `chain.composition` snapshot |
| Cross-cutting    | always-on         | -       | n/a — LCEL doesn't emit cross-cutting events |

**Recommended deployment:** `CaptureConfig.standard()` if you only need
inputs/outputs/model/tool calls (default). Switch to
`CaptureConfig.full()` (or set `l2_agent_code=True`) when you need the
pipeline DAG for debugging, replay, or visualization.

## Hierarchy and depth

The adapter computes depth from `parent_run_id` chains. A runnable
whose `parent_run_id` does NOT correspond to a runnable already
tracked by this handler is treated as a **new root** (depth 0). This
keeps the tracker resilient to:

* LangGraph nodes (handled by a separate code path that pre-empts LCEL
  tracking — see the langgraph adapter docs)
* Pre-existing legacy chain calls that wrap an LCEL pipeline
* Multiple concurrent root runnables driven by different invocations

When a sub-graph parent IS tracked, the child is recorded with
`depth = parent_depth + 1`. The composition snapshot's `max_depth`
field reflects the deepest tracked descendant in the tree.

## Lambda fingerprinting

`RunnableLambda` instances expose a `fingerprint` field on both
`agent.input` and `agent.code` events. The fingerprint is a 16-char
hex prefix of `SHA-256(name | depth | parent_kind | role | label)`.
The same lambda invoked at the same composition position produces the
same fingerprint, enabling "did this lambda change between two runs?"
diffs in the UI.

The fingerprint does NOT include the inner callable's source code —
LangChain doesn't expose source through the callback path. For
fine-grained "code-changed" detection at the source level, instrument
the lambda directly with a `@layerlens.observe` decorator.

## LangGraph interaction

LangGraph drives LCEL pipelines under the hood, so a graph node's
`on_chain_start` callback may carry both `metadata["langgraph_node"]`
AND a runnable `name` (e.g. `RunnableSequence`). The adapter
**prefers the LangGraph signal**: when a `langgraph_node` marker is
present, the existing LangGraph attribution path runs and LCEL
tracking is suppressed for that subtree. This avoids double-emission
on graphs that LangGraph itself drives.

If you want LCEL tracking inside a graph node, invoke the LCEL
pipeline directly with `pipeline.invoke(input, config={"callbacks":
[handler]})` from within the node's body — LangChain will not attach
the `langgraph_node` marker to a manual `.invoke()`, so LCEL tracking
will engage.

## See also

* Spec: `docs/incubation-docs/adapter-framework/04-per-framework-specs/04b-langchain-adapter-spec.md` §1 weakness #4 and §4
* Sample: `samples/instrument/langchain/lcel_main.py` — runnable LCEL
  pipeline with all five primitives, no API key required
* Tests: `tests/instrument/adapters/frameworks/langchain/test_lcel.py`
* Source: `src/layerlens/instrument/adapters/frameworks/langchain/lcel.py`
