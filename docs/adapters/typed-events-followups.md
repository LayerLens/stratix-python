# Typed Events Migration Backlog

**Status: incomplete.** This PR (`feat/instrument-typed-events-foundation`)
ships the foundation (typed-event registry, dual-path emission contract,
`DeprecationWarning` on the legacy path) plus the **agno reference
migration only**. Every other framework adapter still emits via
`BaseAdapter.emit_dict_event` and triggers a `DeprecationWarning` on
each call.

This is honest disclosure per CLAUDE.md item 11: the deliverable is
the foundation + 1 of 17 adapters migrated, not "all 17 done".

## Site counts (as of this PR)

Counts are produced by:

```bash
grep -rcE "self\.emit_dict_event\(" src/layerlens/instrument/adapters/<adapter>/
```

### Framework adapters (16 remaining + 1 done)

| Adapter | Sites | Status |
|---|---|---|
| agno | **0** | Migrated in this PR |
| agentforce | 1 | Pending — split between top-level + subdir; sub-modules untracked on this branch |
| autogen | 8 | Pending — subdir lifecycle.py only; spec said 15 (likely counted untracked groupchat/wrappers/etc.) |
| bedrock_agents | 13 | Pending |
| crewai | 8 | Pending — spec said 10 (callbacks/delegation/metadata.py untracked) |
| embedding | 0 | No emissions in tracked code |
| google_adk | 11 | Pending |
| langchain | 1 | Pending — only callbacks.py tracked; spec said 15 (chains/agents/state untracked) |
| langfuse | 0 | No emissions in tracked code (importer-style) |
| langgraph | 5 | Pending — spec said 13 (nodes/tools/handoff/llm/state untracked) |
| llama_index | 12 | Pending |
| ms_agent_framework | 12 | Pending |
| openai_agents | 15 | Pending |
| pydantic_ai | 10 | Pending |
| semantic_kernel | 10 | Pending |
| smolagents | 7 | Pending |
| strands | 10 | Pending |
| **Total pending** | **123 sites across 14 adapters** | |

### Protocol adapters (3 remaining)

| Adapter | Sites | Status |
|---|---|---|
| protocols/agui | 0 | Submodules untracked on this branch (spec said 2) |
| protocols/a2a | 0 | Submodules untracked on this branch |
| protocols/mcp | 0 | Submodules untracked on this branch |
| protocols/a2ui | 0 | Pending if/when adapter ships emissions |
| protocols/ap2 | 0 | Pending if/when adapter ships emissions |
| protocols/ucp | 0 | Pending if/when adapter ships emissions |

### Provider adapters (`providers/_base/provider.py`)

| Adapter | Sites | Status |
|---|---|---|
| `providers/_base/provider.py` | 4 | Pending — shared base for all 9 LLM provider adapters |
| Per-provider adapter files | 0 | Provider adapters route emissions through the shared `_base/provider.py` |

Migrating `_base/provider.py` will retire all 9 provider adapter
emissions in one commit (anthropic, azure_openai, bedrock, cohere,
google_vertex, litellm, mistral, ollama, openai).

## Spec vs reality

The original PR spec listed projected site counts that included
sub-modules (e.g. `langchain/chains.py`, `langgraph/nodes.py`,
`autogen/groupchat.py`) that are not currently tracked on the
`feat/instrument-multitenancy-org-id-propagation` base branch. The
counts above reflect what is actually present in this branch's
worktree at the moment of writing. When the missing sub-modules land
(via PRs `feat/instrument-frameworks-langchain`,
`feat/instrument-frameworks-langgraph`, etc., which are stacked
behind `feat/instrument-base-foundation` PR #93), the per-adapter
counts will rise to match the spec's projections.

## Migration order (recommended)

Migrate in increasing complexity to keep PRs small and reviewable:

1. **agno** ✓ (done, this PR)
2. **agentforce** — 1 site, smallest surface
3. **langchain** — 1 site (will grow when sub-modules land)
4. **embedding** / **langfuse** — 0 sites today; revisit when emissions land
5. **langgraph** — 5 sites
6. **smolagents** — 7 sites
7. **autogen** / **crewai** — 8 sites each
8. **strands** / **pydantic_ai** / **semantic_kernel** — 10 sites each
9. **google_adk** — 11 sites
10. **llama_index** / **ms_agent_framework** — 12 sites each
11. **bedrock_agents** — 13 sites
12. **openai_agents** — 15 sites
13. **providers/_base/provider.py** — 4 sites, retires all 9 LLM provider adapters in one commit

## Per-adapter migration template

Every follow-up PR should:

1. Replace every `self.emit_dict_event(event_type, dict)` site with
   `self.emit_event(TypedModel.create(...))` in the adapter source.
2. Set `ALLOW_UNREGISTERED_EVENTS: bool = False` on the adapter
   class (default; only `True` for importer-style adapters).
3. Update the adapter's `test_<adapter>_adapter.py` to assert the
   canonical payload shape and update `_RecordingStratix` to capture
   typed payloads (mirror the agno changes in
   `tests/instrument/adapters/frameworks/test_agno_adapter.py`).
4. Add a `test_<adapter>_emits_typed_payloads_only` regression test.
5. Add a `test_<adapter>_emit_does_not_warn_after_migration` test
   that fails if any call site still triggers
   `DeprecationWarning`.
6. Verify `grep -c "self\.emit_dict_event(" src/.../<adapter>/`
   returns `0`.

## When does the legacy path get removed?

`emit_dict_event` will be removed in the next major SDK release
(2.0.0) once all 16+ adapters have migrated. Until then, the
`DeprecationWarning` is the visible signal that an adapter is
behind. CI should run `pytest -W error::DeprecationWarning` against
the post-migration adapter set to enforce that no new emit_dict
calls slip in.
