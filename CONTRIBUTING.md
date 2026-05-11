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

## Code of conduct

This project follows the [Code of Conduct](./CODE_OF_CONDUCT.md). By participating, you agree to abide by it.

## Reporting security issues

Do not file a public issue for security vulnerabilities. See [SECURITY.md](./SECURITY.md) for the private disclosure process.

## License

By contributing, you agree your contribution is licensed under the [Apache License 2.0](./LICENSE).
