# Stratix Quality Gate (GitHub Action)

A composite GitHub Action that runs a Stratix AI evaluation gate on pull
requests: scores a pinned trace set (or a baseline diff) against your
configured judges, fails the job below threshold, and posts a PR comment
listing every failed judge with its reasoning.

Publishable to the GitHub Actions Marketplace (composite action, unlike the
reusable-workflow sample this replaces).

## Inputs

| Input | Default | Notes |
|---|---|---|
| `api-key` | `''` | Stratix API key. Falls back to `LAYERLENS_STRATIX_API_KEY`. |
| `threshold` | `0.85` | Minimum pass rate (mode `threshold`). |
| `trace-set` | `''` | JSON file of pinned trace IDs. Omit = newest traces (warned, not comparable run to run). |
| `baseline-file` | `''` | JSON file mapping trace ID to last-known pass rate, for `mode: diff`. |
| `mode` | `threshold` | `threshold` (absolute pass rate) or `diff` (regression vs baseline). |
| `max-traces` | `10` | Max traces evaluated per run. |
| `max-evaluations` | `30` | Max (trace, judge) pairs. Sized for the default timeout. |
| `timeout-minutes` | `20` | Documented job timeout. Keep consistent with `max-evaluations` (worst case 60 s per evaluation). |
| `post-comment` | `true` | Post failures + judge reasoning as a (replacing) PR comment. |
| `python-version` | `3.12` | |
| `sdk-index` | `https://sdk.layerlens.ai/package` | The LayerLens SDK index. Do NOT use plain PyPI; the `layerlens` name there is an unrelated project at v0.1.1. |

## Outputs

`pass-rate`, `verdict` (`pass`/`fail`), `comment-url`.

## Usage (same repo)

```yaml
- uses: ./.github/actions/stratix-gate
  with:
    api-key: ${{ secrets.LAYERLENS_STRATIX_API_KEY }}
    trace-set: ".github/trace-set.json"
```

## Usage (published action, any repo)

```yaml
- uses: LayerLens/stratix-python@v1
  with:
    api-key: ${{ secrets.LAYERLENS_STRATIX_API_KEY }}
    trace-set: "trace-set.json"
```

## Marketplace publication checklist

1. Commit the action under `.github/actions/stratix-gate/` on the default branch.
2. Tag a release (e.g. `v1.0.0`) pointing at that commit.
3. Draft the Marketplace release ("Publish a new release", check "Publish to
   GitHub Actions Marketplace") and set the maintenance owner to
   LayerLens/support@layerlens.ai.

## What changed vs samples/cicd v1

- Install uses the LayerLens SDK index, not plain PyPI (`pip install layerlens` there pulls an unrelated project).
- Traces are pinned or baseline-diffed; scoring "newest traces" is opt-in and labeled.
- Failed judges and their reasoning are posted to the PR as a comment.
- `max-evaluations` (30) and the job timeout (20/45 min) are consistent, replacing 200 evaluations against a 15-minute timeout.
