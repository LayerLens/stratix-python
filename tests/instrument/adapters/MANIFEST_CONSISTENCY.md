# Manifest consistency lint guards

`test_manifest_consistency.py` is a lint-style test file that keeps the
adapter manifest, capability declarations, emitted-event surface, and
shipping artifacts in agreement with each other. It walks the source
tree the same way `scripts/emit_adapter_manifest.py` does, and asserts
that the metadata the emitter would write actually matches what every
adapter implements.

The tests run without any optional framework runtimes (`pydantic_ai`,
`smolagents`, `boto3`, etc.) installed. Every check is a static AST or
filesystem assertion.

## What the lint catches

### 1. Capability ↔ hook consistency

If a lifecycle adapter implements an `on_<event>` hook, the matching
`AdapterCapability` MUST be declared in
`get_adapter_info().capabilities`. The reverse is also enforced:
declaring a capability without a satisfying hook is a contract lie that
shows up as a broken filter in the catalog UI.

The mapping between capabilities and the hook(s) that satisfy them is:

| Capability               | Satisfying hook(s)                                 |
| ------------------------ | -------------------------------------------------- |
| `TRACE_HANDOFFS`         | `on_handoff`                                       |
| `TRACE_TOOLS`            | `on_tool_use`, `on_function_start`, `on_function_end` |
| `TRACE_MODELS`           | `on_llm_call`, `on_model_invoke`                   |

Multiple hook names per capability accommodate frameworks whose native
callback shape differs from the canonical names (`semantic_kernel`'s
`on_function_*` and `on_model_invoke` are the canonical example).

**Real bug this catches:** `pydantic_ai`'s `lifecycle.py` defines an
`on_handoff` hook but its `get_adapter_info()` only declares
`TRACE_TOOLS`, `TRACE_MODELS`, and `TRACE_STATE` — `TRACE_HANDOFFS` is
missing. The catalog filter "agents that support handoffs" therefore
silently excludes pydantic_ai even though the adapter does emit handoff
events. This test is xfailed for pydantic_ai today; the sibling
pydantic_ai capabilities cleanup PR fixes the declaration.

### 2. Canonical event-type parity

Every framework adapter that ships a `lifecycle.py` MUST emit at least
one occurrence of each of the canonical event types:

- `agent.input` — L1 lifecycle start
- `agent.output` — L1 lifecycle end
- `tool.call` — L5a tool invocation
- `model.invoke` — L3 model call
- `environment.config` — Cross-cut adapter / agent configuration

These five form the minimum surface the trace UI in atlas-app renders
against. Adapters that hook into a different runtime shape (e.g.,
`semantic_kernel`'s per-function callbacks) still need to translate
their events into these canonical types.

**Real bug this catches:** prior to its remediation,
`bedrock_agents` emitted only a subset of the canonical types from
the `_process_trace` hook path, leaving `environment.config` and
`tool.call` unset. The audit caught the gap; the lint now prevents
regression.

### 3. Maturity ↔ artifacts

Every adapter listed in `scripts/emit_adapter_manifest._MATURE` MUST
have ALL FOUR shipping artifacts:

1. **Test file** at `tests/instrument/adapters/<category>/test_<key>_adapter.py`
   with at least 12 test functions
2. **Sample** at `samples/instrument/<key>/main.py`
3. **Reference doc** at `docs/adapters/<category>-<key>.md`
4. **STRATIX→LayerLens deprecation alias** in the package
   `__init__.py` (an attribute named `STRATIX*` aliasing the
   canonical class)

Missing any one of these means the catalog shouldn't be advertising
the adapter as GA. The test fails loudly if a key is promoted to
`_MATURE` without the artifacts in the same PR.

**Real bug this catches:** before this PR, `smolagents` was listed in
`_MATURE` despite missing both the doc
(`docs/adapters/frameworks-smolagents.md`) and the sample
(`samples/instrument/smolagents/main.py`). The audit caught the
mismatch. This same PR moves smolagents from `_MATURE` to
`_LIFECYCLE_PREVIEW` so the lint passes; the sibling PR that adds the
sample + doc moves it back into `_MATURE`.

## How `xfail` is used

Some real gaps cannot be fixed in this PR (they are owned by sibling
PRs that are still in review). For those gaps the test is annotated
with `@pytest.mark.xfail(strict=True, reason="...sibling PR...")`.
The assertion remains real:

- If the gap is still present, the test xfails (CI green).
- If the gap is fixed (sibling PR landed), the test xpasses, and
  `strict=True` flips that into a failure — forcing whoever lands the
  fix to also remove the xfail marker.

Today's xfail entries:

| Test                                                          | Sibling PR                                  |
| ------------------------------------------------------------- | ------------------------------------------- |
| `test_capability_hook_consistency[pydantic_ai]`               | pydantic_ai capabilities cleanup PR         |
| `test_mature_adapters_have_required_artifacts[<provider>]`    | Per-provider port PR (test+sample+doc+alias)|

## Running the lint

```bash
# Just the manifest-consistency suite:
uv run pytest tests/instrument/adapters/test_manifest_consistency.py -v

# Full instrument test tree:
uv run pytest tests/instrument/ -v

# The lint runs in CI on every PR via the standard test gate; no
# additional workflow file is required.
```

## When the lint fires, who fixes it?

The PR that introduced the inconsistency. Specifically:

- **Capability/hook mismatch** → fix `get_adapter_info().capabilities`
  in the adapter's `lifecycle.py` to match the implemented hooks.
- **Canonical event missing** → add the missing
  `self.emit_dict_event("agent.input", {...})` (or similar) call to
  the appropriate hook in `lifecycle.py`.
- **Maturity artifact missing** → either land the missing artifact
  (test file / sample / doc / alias), or move the key from `_MATURE`
  to `_LIFECYCLE_PREVIEW` until the artifact is ready.

Do not "fix" the lint by deleting the test or expanding the xfail set
without explicit reviewer approval — the lint is the contract.
