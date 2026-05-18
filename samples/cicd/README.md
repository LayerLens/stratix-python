# CI/CD Samples

AI quality assurance must be integrated into your development workflow to catch regressions
before they reach production. These samples provide ready-to-use components for embedding
LayerLens evaluations into continuous integration pipelines: a GitHub Actions quality gate
that blocks pull requests when AI output quality drops below threshold, a reusable gate
script for any CI system, and a pre-commit hook for local development.

## Prerequisites

```bash
pip install layerlens --index-url https://sdk.layerlens.ai/package
```

- A valid `LAYERLENS_STRATIX_API_KEY` set as an environment variable (locally or as a CI secret)
- For the GitHub Actions workflow: repository write access to add workflow files
- For the pre-commit hook: a local Git repository

## Quick Start

Run the quality gate script locally to verify your current evaluation pass rate:

```bash
export LAYERLENS_STRATIX_API_KEY=your-api-key
python quality_gate.py --threshold 0.85
```

Expected output: the script evaluates recent traces against all configured judges, prints the
aggregate pass rate, and exits with code 0 (pass) or 1 (fail) based on the threshold.

## Samples

| File | Scenario | Description |
|------|----------|-------------|
| `quality_gate.py` | CI/CD engineers adding AI quality checks to any pipeline | Evaluates recent traces against all judges and exits non-zero if the pass rate falls below a configurable threshold. Designed to be called from any CI system. |
| `github_actions_gate.yml` | Teams using GitHub Actions for pull request validation | A complete GitHub Actions workflow that runs the quality gate on every pull request. Copy to `.github/workflows/ai-quality-gate.yml` in your repository. |
| `pre_commit_hook.py` | Developers catching issues before committing | A Git pre-commit hook that runs a quick safety evaluation on staged changes. Prevents commits that introduce safety regressions. |

## Installation

### GitHub Actions

```bash
cp samples/cicd/github_actions_gate.yml .github/workflows/ai-quality-gate.yml
```

Add `LAYERLENS_STRATIX_API_KEY` as a repository secret in your GitHub settings.

### Pre-Commit Hook

```bash
ln -sf "$(pwd)/samples/cicd/pre_commit_hook.py" .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

## Expected Behavior

The quality gate script outputs a pass/fail summary with per-judge breakdown. In CI, a
failing gate will cause the pipeline step to exit non-zero, blocking the merge. The
pre-commit hook runs silently on success and prints a warning message on failure.
