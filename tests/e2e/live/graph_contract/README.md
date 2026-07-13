# Agent-graph contract harness (G7)

End-to-end verification of the agent-graph contract across three layers:

```
SDK adapter events  ->  atlas graph engine (services.InferAgentGraph)  ->  FE render
```

It has two halves — a deterministic dual-oracle (always runnable, no stack) and a
live seed→read-back (gated on a running atlas).

## Files
- `oracle_expectations.json` — per-lane `shipped_nodes`/`shipped_column` (engine
  `honestGuard=false`) and `guarded_nodes`/`guarded_column` (`honestGuard=true`),
  generated from the **real** atlas Go engine (`services.InferAgentGraph` vs
  `inferAgentGraph(_, true)`) over `apps/backend/services/testdata/graph_honesty_fixtures.json`.
- `seed_events.json` — the raw `{lane: events}` from that same corpus (the SDK
  upload payloads for the live half).
- `_contract.py` — pure assertion helpers (dual-oracle math, node-set/column
  projection, the SDK-generic guard anchor). Unit-testable, no stack.
- `_live.py` — seed a lane via a real `Stratix()` client + raw-HTTP read-back
  (the SDK `Trace` model omits `graph`/`agent`, so read-back is raw API).
- `test_dual_oracle.py` — deterministic; the honestGuard divergence, surfaced.
- `test_live_contract.py` — gated live seed→read-back.

## test_dual_oracle.py (deterministic — no live stack)
The atlas engine ships `honestGuard=false` (ateam parity): it surfaces
producer-declared identities verbatim, including generic framework class-names
(`agno_agent`, `ToolCallingAgent`, `Strands Agents`). The guarded oracle drops
them. This test enumerates that divergence and asserts it is **bounded to
identities the SDK's own resolver (`_identity`) also rejects** — the SDK↔server
honesty agreement — so the shipped default's extra nodes are exactly the generic
labels, never a real agent. It also asserts the guard only removes (never
invents) and that genuinely-honest lanes are guard-invariant. This SURFACES the
`honestGuard` product decision as a monitored fact without changing it.

## test_live_contract.py (gated live)
Seeds each corpus lane through a real `Stratix()` client to
`LAYERLENS_STRATIX_BASE_URL` and asserts, over the raw trace-detail API:
`graph` present iff topology; server node-set == shipped oracle; count-aware
Agent column == graph projection; re-upload with agentless events UNSETS the
stale graph (keyed on `sdk_trace_id`); the live server's divergence vs the
guarded oracle is SDK-generic; and each served graph is FE-renderable (label +
`agent_type` per node — the G1 render contract).

### Running it
The whole `tests/e2e/live` tree is skipped unless `LAYERLENS_LIVE=1`. Then:
```
set -a; source tests/e2e/live/.env; set +a
LAYERLENS_LIVE=1 rye run pytest tests/e2e/live/graph_contract/
```
It is **base-URL-driven**: point `LAYERLENS_STRATIX_BASE_URL` at a local `:8080`
stack **or** a dev deploy — whichever carries the engine and a valid key.

**Honest note on where this was executed (2026-07-13):** the goal specified the
local `:8080` stack. That stack is running with the engine, but its app-key
validation goes through AWS (`GetAPIKeyMetadata`) and no local write credential
was available (the `.env` admin JWT is a cloud Cognito token that expired
2026-07-03), so seeding `:8080` was blocked on a credential — not code. The live
half was therefore executed against the **dev deploy** in `.env`
(`…execute-api…/prod`), which carries the merged engine (#2054) and a valid key:
**34 passed** (13 dual-oracle + 21 live), lanes `agno/autogen/bedrock_agents/
crewai/google_adk/langfuse/langgraph/pydantic_ai/smolagents/strands`. The exact
same suite runs against `:8080` unchanged once a local key is provided.
```
34 passed in 84.59s
```
Uploaded traces are tagged and KEPT (`LAYERLENS_LIVE_KEEP_TRACES=1`), never deleted.
