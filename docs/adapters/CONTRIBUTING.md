# Contributing an adapter

This guide covers porting an adapter from `ateam` to `stratix-python` at
the quality bar required by CLAUDE.md.

## Quality gate (non-negotiable)

Every PR must produce all of:
- mypy `--strict` clean on the new files
- pyright clean (project config) on the new files
- ruff clean on the new files
- pytest green for the new tests
- A live integration test gated by `@pytest.mark.live` and the relevant
  `*_API_KEY` env var (where the framework supports a real backing service)
- A runnable sample under `samples/instrument/<adapter>/`
- A reference doc under `docs/adapters/<category>-<name>.md`

CI matrix runs the new extra at both min-pin and latest-in-range.

## Naming convention

The `ateam` source uses `STRATIX*` class prefixes for public adapter classes
(e.g., `STRATIXCallbackHandler`, `STRATIXLangGraphAdapter`,
`STRATIXLiteLLMCallback`). When porting:

1. Rename the public class to `LayerLens*` (e.g., `STRATIXCallbackHandler` →
   `LayerLensCallbackHandler`).
2. Add a backward-compat alias at module scope: `STRATIXCallbackHandler = LayerLensCallbackHandler`.
3. Note the alias in the adapter's reference doc with a deprecation timeline
   (default: removed in v2.0).
4. Internal class names (`OpenAIAdapter`, `AnthropicAdapter`, etc.) that
   were never prefixed in `ateam` stay as-is.

The `LiteLLMAdapter` port (`src/layerlens/instrument/adapters/providers/litellm_adapter.py`)
is the canonical example.

## Compatibility constraints

- **Python 3.8+**: do NOT use `StrEnum`, `from datetime import UTC`, PEP 604
  union types in non-annotation contexts, or `match` statements. The
  `_compat.pydantic` shim covers Pydantic v1↔v2 differences (`BaseModel`,
  `Field`, `model_dump`, `field_validator`, `model_validator`).
- **No framework imports at SDK init time**: the framework SDK must be imported
  only inside methods that the user explicitly calls (`connect`,
  `_detect_framework_version`, etc.). The lazy-import test will catch
  regressions.
- **No new required deps**: every framework SDK goes in `[project.optional-dependencies]`,
  never in `[project] dependencies`. The default-install test enforces this.

## Adapter class checklist

When writing the new adapter class:

- [ ] Inherits from `BaseAdapter` (frameworks) or `LLMProviderAdapter` (LLMs)
- [ ] Sets `FRAMEWORK` and `VERSION` class attributes
- [ ] Implements `connect()`, `disconnect()`, `health_check()`,
      `get_adapter_info()`, `serialize_for_replay()` (or inherits the LLM
      provider variants)
- [ ] Exports `ADAPTER_CLASS = MyAdapter` at module scope (registry uses this
      for lazy loading)
- [ ] Adds an entry to `_ADAPTER_MODULES` and `_FRAMEWORK_PACKAGES` in
      `_base/registry.py`
- [ ] Adds a `pyproject.toml` extras entry with the framework's pip name and
      version range; gates Python-version markers if the framework requires
      3.10+
- [ ] Updates `tests/instrument/test_lazy_imports.py::_FORBIDDEN_PREFIXES`
      with the framework's import name

## Test checklist

Three tiers:

1. **Unit tests** (`tests/instrument/adapters/<category>/test_<name>.py`):
   - Mock the framework's SDK responses with `SimpleNamespace` objects
   - Cover success path, error path, all wrapped methods, capture-config
     gating, disconnect-restores-originals
   - Assert on event types, payload fields, and structural invariants

2. **Sink-level e2e** (covered by the existing
   `tests/instrument/test_sink_http_e2e.py`): every adapter that emits via
   `HttpEventSink` benefits from this test suite — no new test needed unless
   the adapter has a bespoke transport.

3. **Live integration** (`tests/instrument/adapters/<category>/test_<name>_live.py`):
   - Module-level `pytestmark` skips without `<FRAMEWORK>_API_KEY`
   - Hit the real service with a tiny request (max_tokens 5–10 to bound cost)
   - Assert that real response field names map to your event payload fields —
     this is what catches SDK schema drift

## Sample + doc checklist

- `samples/instrument/<adapter>/main.py`: runnable via `python -m
  samples.instrument.<adapter>.main`. Checks for env vars; gives clear
  diagnostic if missing. Uses `adapter.add_sink(sink)` (the public API).
- `samples/instrument/<adapter>/README.md`: install command, env-var summary,
  what events the user will see, link to the reference doc.
- `docs/adapters/<category>-<name>.md`: install, quick start, events emitted
  with table, framework-specific behavior, cost calculation notes, BYOK
  notes, capture-config notes.
