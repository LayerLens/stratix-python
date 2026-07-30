# Adapter test infrastructure: coverage manifest + real-framework matrix

Three pieces keep ~30 fast-moving framework integrations honest
(LAY-3574/3580/3581):

| Piece | File(s) | What it does |
|---|---|---|
| Coverage manifest | `tests/adapter_manifest.toml` + `tests/test_adapter_manifest.py` | Declares the required test tiers per adapter; the gate fails on undeclared adapters, missing tier files, or dangling cross-references. Runs in base CI. |
| Real-framework matrix | `tests/matrix/frameworks.toml` + `run_matrix.py` + `.github/workflows/adapter-matrix.yaml` | One isolated uv venv per framework at the pinned **verified** version, running that framework's real test modules. A row fails on any failure, on empty collection, or on ANY skip — a skip while the framework is installed means coverage silently died (the B2 failure mode). PRs run only rows their diff touches; nightly runs everything. Zero LLM spend, no API keys. |
| Drift canary | `.github/workflows/adapter-drift-canary.yaml` | Weekly advisory rerun of the matrix with pins removed (latest upstream releases). Failures appear in the run's step summary and never block PRs or main. |

## Running locally

```bash
python3 tests/matrix/run_matrix.py --list                     # row names
python3 tests/matrix/run_matrix.py --framework crewai         # one row, pinned
python3 tests/matrix/run_matrix.py --framework crewai --latest  # canary mode
git diff --name-only main...HEAD | python3 tests/matrix/run_matrix.py --pick  # what a PR would run
```

Requires `uv` on PATH. Each row builds a throwaway venv (`--keep-venv` /
`--venv-dir` to inspect it).

## Adding a new adapter — checklist

1. Write the adapter + its real-framework unit module under
   `tests/instrument/adapters/...` (importorskip on the framework package;
   the skip target must be in `tests/test_skip_hygiene.py`'s allowlist).
2. Declare the adapter in `tests/adapter_manifest.toml` with its tiers
   (unit, redaction, concurrency, disconnect_restore, ... — see the tier
   vocabulary in that file's header). The manifest gate fails until you do.
3. Add a row to `tests/matrix/frameworks.toml` pinning the version you
   verified against, and run it locally:
   `python3 tests/matrix/run_matrix.py --framework <name>`.
4. If the adapter has a live scenario, register it in the relevant
   `tests/e2e/live/_*_registry.py` and reference the id as `live_id` in the
   manifest.

## Updating pins

Pins are the latest **verified** versions (the canary tells you when newer
releases break). To bump: change the pin, run the row locally, commit the
pin + any fixes together.
