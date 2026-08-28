# Recorded API response corpus

Response bodies recorded from atlas-app's production Go structs, parsed here by
`test_response_corpus.py` with the SDK's own pydantic models. This is the SDK half
of the `/api/v1` response-compatibility gate; the API half — the generator, the
serialization contract tests, and the written rule — lives in atlas-app at
`DOCS/api-contract/`.

## Refreshing

```
cd apps && go test ./backend/api/v1/evaluations/ -run TestSDKCorpus -update
```

then copy `DOCS/api-contract/corpus/*.json` over this directory. atlas-app runs the
same generator in verify mode under its normal `go test ./...`, so a wire change
that is not regenerated fails CI there.

**This copy can drift**, because it lives in a different repo from the generator.
The refresh above is a manual step. A shared artifact would be better; this is what
exists today.

## The files

| File | Records |
|---|---|
| `evaluations_get_one_not_computed.json` | `GET /evaluations/{id}` where readability/toxicity were never computed — the keys are **absent** |
| `evaluations_get_one_computed.json` | the same with every nullable field populated, including a **genuine** `readability_score` of `0` |
| `evaluations_get_many.json` | the list envelope (`EvaluationWithMeta`), one not-computed row and one computed row |
| `evaluations_get_many_admin.json` | the admin list envelope (`AdminEvaluation`), proving the wrapper adds no nullable keys |
| `evaluations_get_one_legacy_null_scores.json` | **what the DEPLOYED pre-fix build emits**: `readability_score: null`, `toxicity_score: null` |
| `evaluations_get_many_legacy_null_scores.json` | the list form of the same |
| `results_get_empty_page.json` | `GET /results` for an evaluation with no matching rows |
| `results_get_flat_metrics.json` | built-in metrics, and `duration` as the int64 **nanosecond** count the API sends |
| `results_get_scorer_metrics.json` | custom-scorer metrics: one object per scorer, including a failed scorer with `score: null` |

## Two things to keep in mind

**The fixture design is the load-bearing part, not the harness.** This corpus only
catches LAY-3765 because it contains an evaluation whose scores were never
computed. A corpus of fully-populated happy paths parses cleanly and proves
nothing. When the API gains a nullable field, add a fixture where it is nil.
`test_corpus_covers_the_not_computed_case` exists to stop someone "tidying" that
away.

**The `*_legacy_*` files are frozen recordings, not byte-identical captures.** They
are produced by shadowing the embedded struct's tags, which places the two shadowed
keys at the end of the object rather than in field order. Key presence, values and
nullability are exact; key *order* differs from a real response. JSON object order
is not semantically meaningful and no client depends on it — but do not describe
these as byte-identical to production. **Do not delete them when the fix deploys:**
they are the only executable record of what a pinned client receives.
