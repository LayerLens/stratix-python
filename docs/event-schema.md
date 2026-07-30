# Event Schema Reference

> This document mirrors [`tests/instrument/_event_schema.py`](../tests/instrument/_event_schema.py),
> which **is** the enforced source of truth. That module is validated in CI: the
> `capture_trace` / `capture_framework_trace` fixtures validate every uploaded
> event, so every adapter unit suite participates in the lock without per-test
> wiring. If you change the schema, change it there first — the two **must stay
> in sync**, and adding a new event type is a deliberate act (add it in the same
> PR that introduces it).

This is the canonical payload vocabulary the adapters must emit. Any **new**
drift fails loudly; ratcheting the documented exceptions down is the future
§3.6 convergence work (the module is that worklist), not a renaming pass.

---

## Known event types (`KNOWN_EVENT_TYPES`)

Every payload uploaded by a unit suite must use an event type registered in
`KNOWN_EVENT_TYPES`. An unregistered type is a hard failure.

### Agent family
| Event type | Description |
|---|---|
| `agent.input` | Input handed to an agent. |
| `agent.output` | Output produced by an agent. |
| `agent.error` | Error raised during an agent run. |
| `agent.step` | A single step in an agent's execution. |
| `agent.code` | Code emitted/executed by a code-acting agent. |
| `agent.handoff` | Handoff/delegation between agents (e.g. collaborator handoff). |
| `agent.identity` | Agent identity metadata. |
| `agent.interaction` | An interaction involving the agent. |
| `agent.lifecycle` | Agent lifecycle event. |
| `agent.state.change` | A change in agent state. |
| `agent.node.enter` | Entry into a graph node. |
| `agent.node.exit` | Exit from a graph node. |

### Model / cost / environment
| Event type | Description |
|---|---|
| `model.invoke` | A model invocation (carries a `usage` token dict). |
| `cost.record` | A cost record (must carry token counts; `cost_usd` optional — see below). |
| `embedding.create` | An embedding creation call (carries a `usage` token dict). |
| `environment.config` | Environment configuration captured at call time. |
| `environment.metrics` | Environment/runtime metrics. |

### Tools / retrieval
| Event type | Description |
|---|---|
| `tool.call` | A tool invocation. |
| `tool.result` | A tool result. |
| `tool.logic` | Tool logic/decision event. |
| `tool.environment` | Tool environment event. |
| `retrieval.query` | A retrieval/RAG query. |

### Conversation (autogen group chat)
| Event type | Description |
|---|---|
| `conversation.started` | A group-chat conversation started. |
| `conversation.ended` | A group-chat conversation ended. |
| `conversation.message` | A message within a group-chat conversation. |

### Policy / evaluation
| Event type | Description |
|---|---|
| `policy.violation` | A policy violation was detected. |
| `evaluation.result` | An evaluation result. |

### Protocol family
| Event type | Description |
|---|---|
| `protocol.agent_card` | A protocol agent card. |
| `protocol.stream.event` | A streamed protocol event. |
| `protocol.lifecycle` | A protocol lifecycle event. |
| `protocol.task.submitted` | A protocol task was submitted. |
| `protocol.task.completed` | A protocol task completed. |
| `protocol.async_task` | An asynchronous protocol task. |
| `protocol.elicitation.request` | A protocol elicitation request. |
| `protocol.elicitation.response` | A protocol elicitation response. |
| `protocol.tool.structured_output` | Structured tool output over a protocol. |
| `protocol.mcp_app.invocation` | An MCP-app invocation over a protocol. |

### MCP
| Event type | Description |
|---|---|
| `mcp.tool.call` | An MCP tool call. |
| `mcp.tools.listed` | MCP tools were listed. |
| `mcp.async_task` | An asynchronous MCP task. |
| `mcp.elicitation` | An MCP elicitation. |
| `mcp.structured_output` | MCP structured output. |

### A2A (agent-to-agent)
| Event type | Description |
|---|---|
| `a2a.task.created` | An A2A task was created. |
| `a2a.task.updated` | An A2A task was updated. |
| `a2a.task.completed` | An A2A task completed. |
| `a2a.agent.card` | An A2A agent card. |
| `a2a.agent.card.served` | An A2A agent card was served. |
| `a2a.agent.discovered` | An A2A agent was discovered. |
| `a2a.delegation` | An A2A delegation. |

### AG-UI
| Event type | Description |
|---|---|
| `agui.message` | An AG-UI message. |
| `agui.tool_call` | An AG-UI tool call. |
| `agui.state` | An AG-UI state event. |

### Commerce
| Event type | Description |
|---|---|
| `commerce.supplier_discovered` | A supplier was discovered. |
| `commerce.catalog.browsed` | A catalog was browsed. |
| `commerce.checkout.started` | A checkout was started. |
| `commerce.checkout_completed` | A checkout completed. |
| `commerce.refund_issued` | A refund was issued. |
| `commerce.ui.surface_created` | A commerce UI surface was created. |
| `commerce.ui.user_action` | A user action on a commerce UI surface. |

### Payment
| Event type | Description |
|---|---|
| `payment.intent_mandate` | A payment intent mandate. |
| `payment.mandate_signed` | A payment mandate was signed. |
| `payment.receipt_issued` | A payment receipt was issued. |

---

## Token / usage vocabulary

There are **two** token vocabularies. Mixing them in a single payload is drift
and fails the lock.

### Provider-family events: the `usage` dict (`USAGE_KEYS`)

Provider-family events (`model.invoke` / `embedding.create`) carry a `usage`
dict. Its keys must come from `USAGE_KEYS`:

- `prompt_tokens`
- `completion_tokens`
- `total_tokens`
- `input_tokens`
- `output_tokens`
- `cached_tokens`
- `reasoning_tokens`
- `thinking_tokens`
- `cache_read_input_tokens`
- `cache_creation_input_tokens`

`usage` must be a dict, and every value must be `int` or `None`. Unknown keys
fail (extend `USAGE_KEYS` deliberately).

### Framework-family events: flat token fields (`FRAMEWORK_TOKEN_KEYS`)

Framework-family emitters use flat fields rather than a nested `usage` dict:

- `tokens_prompt`
- `tokens_completion`
- `tokens_total`

Each flat token value must be `int` or `None`. A payload may not mix the
framework flat vocabulary (`tokens_prompt`/`tokens_completion`) with the
provider vocabulary (`prompt_tokens`/`completion_tokens`/`total_tokens`).

---

## Canonical timing field

`latency_ms` (a number) is the canonical duration field. When present it must
be numeric.

### Drift exception: `duration_ns` (`DURATION_NS_EXCEPTIONS`)

`duration_ns` is non-canonical — new adapters must use `latency_ms`. It survives
only where the drift table records today's drift. The table is a set of
`(framework marker, event_type)` pairs, where `"*"` matches any event type from
that adapter:

| Adapter (marker) | Event type | Why it's grandfathered |
|---|---|---|
| `smolagents` | `*` (all events) | Pre-existing drift — smolagents emits `duration_ns` everywhere. |
| `crewai` | `*` (all events) | Pre-existing drift — crewai root events emit `duration_ns`. |
| `strands` | `*` (all events) | Pre-existing drift — strands root events emit `duration_ns`. |
| `google_adk` | `*` (all events) | Pre-existing drift — google_adk root events emit `duration_ns`. |

This is the §3.6 drift table from the stability report. The rule is **shrink it,
never grow it** — any other adapter emitting `duration_ns` fails the lock.

---

## `cost.record` requirement

A `cost.record` event **must** carry token counts (either the framework flat
fields or the provider-style fields); a `cost.record` with no token counts
fails.

`cost_usd` is **optional**: when present it must be a number, but it is not
required. This is because 15 of 18 framework emitters don't compute it — a
documented §3.6 gap and convergence-work item. Locking `cost_usd` as required
would be a rename-scale change, so for a priced model the expectation is that
`cost.record` carries `cost_usd`, but the lock does not yet enforce its presence.
