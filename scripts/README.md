# `scripts/`

Repository-level CLI scripts. Each script is invoked as
`python scripts/<name>.py`. The package `__init__.py` exists only so
that test-time imports (`from scripts.emit_adapter_manifest import ...`)
do not collide with `mypy .`'s top-level module discovery.

## `emit_adapter_manifest.py`

Emits `adapter_catalog/manifest.json` from the SDK registry. Used in
release CI to keep the atlas-app adapter catalog in sync with what
`stratix-python` actually ships.

### Maturity tiers

Each entry in the manifest carries a `maturity` field. The catalog UI in
atlas-app renders adapters differently depending on the tier:

| Tier                | Catalog badge | Meaning                                                                             |
| ------------------- | ------------- | ----------------------------------------------------------------------------------- |
| `mature`            | "GA"          | All four artifacts shipped: dedicated test file (>= 12 functions), reference doc, sample, and STRATIX→LayerLens deprecation alias. Customers can rely on this adapter for production. |
| `lifecycle_preview` | "Preview"     | Adapter ships the full lifecycle hook surface (`on_run_start`, `on_run_end`, `on_tool_use`, `on_llm_call`, etc.) and emits the canonical L1/L3/L5a event set, but one or more of the four artifacts is still being authored. Runtime works today, polish in flight. |
| `smoke_only`        | "Experimental"| Only covered by the bulk smoke-test suite. No graduation work has started.          |

The tier is determined by membership in two sets defined in
`emit_adapter_manifest.py`:

```python
_MATURE: set[str] = { ... }              # fully graduated
_LIFECYCLE_PREVIEW: set[str] = { ... }   # full lifecycle, missing artifacts
# Anything else falls through to "smoke_only".
```

A module-level assertion enforces the two sets are disjoint — an adapter
cannot be both "fully graduated" and "missing artifacts".

### Promotion workflow

When you ship a new framework adapter, choose the right tier:

1. **Adapter is brand-new and has only smoke coverage** → leave the key
   out of both sets. The emitter classifies it as `smoke_only` by
   default.
2. **Adapter ships full lifecycle hooks but doc / sample / alias is in
   flight** → add the key to `_LIFECYCLE_PREVIEW`.
3. **All four artifacts merged** → move the key from
   `_LIFECYCLE_PREVIEW` to `_MATURE` in the SAME PR that adds the
   missing artifacts.

`tests/instrument/adapters/test_manifest_consistency.py` enforces every
item in `_MATURE` actually has all four artifacts; CI fails loudly if a
key is promoted prematurely.

### Audit context

Prior to the PR that introduced `_LIFECYCLE_PREVIEW`, the emitter only
ever wrote `mature` or `smoke_only`. The `lifecycle_preview` value was
documented in the schema for months but never set. As a result, every
adapter that had hooks but lacked one artifact was rendered as
`smoke_only` in the catalog, hiding real lifecycle coverage from
customers. This README block exists so future maintainers know why all
three tiers matter.

### Usage

```bash
# Write to the default location (../atlas-app/...).
python scripts/emit_adapter_manifest.py

# Print to stdout for inspection / piping.
python scripts/emit_adapter_manifest.py --stdout

# Write to a custom path.
python scripts/emit_adapter_manifest.py --out /tmp/manifest.json
```

## Other scripts

| Script                    | Purpose                                                              |
| ------------------------- | -------------------------------------------------------------------- |
| `port_adapter.py`         | Boilerplate generator for porting a new framework adapter from ateam |
| `port_protocol.py`        | Same, for protocol adapters                                          |
| `regen_dep_baselines.py`  | Regenerate the `tests/instrument/_baselines/` snapshots              |
| `bootstrap`               | First-time developer environment setup                               |
| `format`                  | Run `ruff format` across the repo                                    |
| `lint`                    | Run all lint checks (ruff + pyright + mypy)                          |
| `test`                    | Run the pytest suite                                                 |
| `publish.sh`              | Build + upload to PyPI                                               |
| `push_release_tag.sh`     | Push a versioned release tag                                         |
| `validate_release_tag.sh` | Verify a tag matches the version in `_version.py`                    |
