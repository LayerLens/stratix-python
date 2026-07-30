# Contributing to stratix-python

Thanks for your interest in contributing. The fastest path to a merged PR is to open an issue first so we can align on direction before code.

## Before you start

- Browse [open issues](https://github.com/LayerLens/stratix-python/issues), especially anything tagged `good first issue`.
- For non-trivial changes, [open an issue](https://github.com/LayerLens/stratix-python/issues/new) describing the problem and your proposed approach. We'll respond within a few business days.
- For questions and design discussion, join us in [Discord](https://discord.gg/layerlens).

## Repo layout

- `src/layerlens/` is the SDK source (clients, resources, CLI).
- `tests/` is the test suite (unit, integration, sample E2E).
- `samples/` holds runnable code samples organized by topic: `core`, `cicd`, `cli`, `mcp`, `integrations`, `industry`, `modalities`, `claude-code`, `cowork`, `copilotkit`, `openclaw`, `data`.
- `docs/` is the source for the [GitBook docs site](https://layerlens.gitbook.io/stratix-python-sdk).
- `scripts/` holds developer scripts (`bootstrap`, `test`, `lint`, `format`, `test_coverage`).
- `pyproject.toml` is the Python project config and tool settings.
- `requirements.lock` and `requirements-dev.lock` are the pinned dependencies.
- `.husky/` holds Git hooks that run on commit (lint-staged formats and lints staged Python files).

## Local setup

The project uses [Rye](https://rye.astral.sh/) to manage Python and dependencies. The bootstrap script sets everything up:

```bash
git clone https://github.com/LayerLens/stratix-python.git
cd stratix-python
./scripts/bootstrap
source .venv/bin/activate
```

If you would rather use plain pip, ensure the Python version in `.python-version` is active, then:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.lock
pip install -e .
```

## Dev loop

```bash
./scripts/test     # run the test suite
./scripts/lint     # run the linter
./scripts/format   # format and auto-fix
```

A pre-commit hook runs `./scripts/format` and `./scripts/lint` against staged Python files automatically.

## Required CI checks

Every PR runs these workflows. They must pass before review:

- [`run-tests.yaml`](https://github.com/LayerLens/stratix-python/actions/workflows/run-tests.yaml) is the full test suite.
- [`invariants.yaml`](https://github.com/LayerLens/stratix-python/actions/workflows/invariants.yaml) — **Invariant Gates**: the privacy / observability / schema contract guards (`pytest -m invariant`). See [Instrumentation invariants](#instrumentation-invariants--guards).
- [`check-format.yaml`](https://github.com/LayerLens/stratix-python/actions/workflows/check-format.yaml) checks formatting.
- [`check-lint.yaml`](https://github.com/LayerLens/stratix-python/actions/workflows/check-lint.yaml) runs the linter.

Run them locally before pushing.

## Pull request guidelines

- One logical change per PR. Smaller PRs merge faster.
- Reference the issue your PR addresses in the description.
- Include a runnable sample under `samples/` when adding a new SDK capability.
- Update `docs/` when changing public API surface.
- Add or update tests under `tests/` when changing behavior.
- Make sure all CI checks are green before requesting review.

## Instrumentation invariants & guards

The SDK is a **trace-capture product**, so a leak of customer content, a silently-dropped failure, or a missing cost is a correctness bug — not a nicety. We learned (LAY-3620 post-mortem) that *finding* leaks isn't enough: bugs of the same class kept reappearing because nothing **enumerated the whole population and checked each member**. So the contracts below are enforced by **standing guards** — the required `Invariant Gates` check (`rye run pytest -m invariant`) plus autouse runtime checks in the full suite. The intent is that a human *can't* reintroduce the class: the build fails.

If you touch `src/layerlens/instrument/**` (an adapter, an `emit()` call-site, an event type, or the capture/redaction machinery), you own these. The PR template has the checklist.

| # | Invariant | Guard (where it bites) |
|---|-----------|------------------------|
| 1 | **No content leaves under `capture_content=False`** — prompts, messages, tool args/results, `error` free-text, code, state, queries, delegation targets, payment detail are stripped at the collector. | `redact_payload` backstop keyed off `_CONTENT_KEYS`; `test_redaction_backstop.py` + `test_no_content_sweep.py` (L7 SENTINEL sweep). |
| 2 | **Content fields are declared** — every content field on an event is in `_CONTENT_KEYS`; every content-bearing type has a strip list or is allowlisted content-free. | `test_content_keys_guard.py` (keys-must-match **and** reverse guard + emit-call-site scan). |
| 3 | **Layer toggles suppress** — every content-bearing type is in `_EVENT_TYPE_MAP` (or `_ALWAYS_ENABLED`), never fail-open, so `minimal()` / a disabled layer suppresses it. | `test_layer_suppression.py`. |
| 4 | **Failures stay observable when content is off** — `agent.error` carries a surviving category (`error_type`/`error_code`/`status`), not just free-text `error` ("redact without going blind"). | Runtime invariant in `_event_schema.validate_event` — runs over **every uploaded event in every adapter suite**. |
| 5 | **Priced calls carry cost** — a framework `cost.record` for a priced model has `cost_usd` (computed in `_emit`, each `_fire`, and langfuse). | `test_cost_usd_fire.py` + per-adapter cost tests. |
| 6 | **No secret reaches a trace** — keys/tokens/connection strings never land in an uploaded event; error strings are scrubbed at the collector chokepoint, independent of `capture_content`. | `scan_for_secrets` autouse guard; `test_secret_scrub.py`. |
| 7 | **New event types are registered** in `KNOWN_EVENT_TYPES`. | `_event_schema.validate_event` + the emitted-constant / emit-literal scans. |

### How to add safely

- **New event type** → register it in `KNOWN_EVENT_TYPES` (`tests/instrument/_event_schema.py`); map it in `_EVENT_TYPE_MAP` to the right L-layer (or `_ALWAYS_ENABLED`); if it carries content add a `_CONTENT_KEYS` entry, else allowlist it content-free in `test_content_keys_guard.py`.
- **New content field on an existing event** → add the field name to that event's `_CONTENT_KEYS` set.
- **New error emit** → set `error_type` (or `error_code`/`status`), not just `error`; pass exception strings that may carry a credential through `safe_error`.
- **New `cost.record`** → emit via `self._emit`/`self._fire`, or call `self._price_cost_record(payload)` before a raw `collector.emit`; set the adapter's `pricing_table` if its models aren't in the default `PRICING`.

### Discipline behind the guards

- **Test-first, and every test must BITE**: revert the fix; the test must fail. A test that passes with the fix reverted guards nothing — that's exactly how vacuous tests and a months-long no-op schema lock shipped before.
- **Prefer population-complete checks** (a runtime guard over every emitted event; a structural map cross-check) over example-based tests and over fragile static analysis. Invariant #4 is the model.
- **A guard you never trip is theater** — exercise it with a seeded positive case (a fake secret, a reverted fix) so green means something.

## Code of conduct

This project follows the [Code of Conduct](./CODE_OF_CONDUCT.md). By participating, you agree to abide by it.

## Reporting security issues

Do not file a public issue for security vulnerabilities. See [SECURITY.md](./SECURITY.md) for the private disclosure process.

## License

By contributing, you agree your contribution is licensed under the [Apache License 2.0](./LICENSE).
