# Benchmark import sample

End-to-end demo of `BenchmarkImportAdapter` — writes a 3-row CSV to
a temp dir, calls `BenchmarkImportAdapter.import_csv` against it
with a schema mapping and tags, and prints the resulting
`ImportResult`.

> `BenchmarkImportAdapter` is a **data importer**, not a runtime
> trace adapter. It does not require any LLM credentials, does not
> instantiate an `HttpEventSink`, and does not emit runtime
> `agent.*` events. This is consistent with the adapter's design
> (per audit row for `benchmark_import`).

## Prerequisites

```bash
pip install 'layerlens[benchmark-import]'
```

The `benchmark-import` extra is intentionally empty — this sample
is replay-based and ships no additional runtime deps beyond
`layerlens` core.

No environment variables are required.

## Run

```bash
uv run python -m samples.instrument.benchmark_import.main
```

## What this demonstrates

| Component | What it proves (source: `main.py`) |
|---|---|
| `BenchmarkImportAdapter()` | Adapter constructs with no args — no sink, no capture config, no transport. |
| Temp CSV via `_write_sample_csv` | Three QA rows (math, geo, science) written via `csv.DictWriter`. |
| `adapter.import_csv(path=..., schema_mapping={"question": "prompt", "answer": "expected_output", "category": "category"}, tags=["sample", "qa"])` | Demonstrates the schema-mapping contract and tag attachment. |
| `result.success` / `result.errors` | Sample bails with exit code `1` and prints errors if the import fails. |
| `result.benchmark_id`, `result.records_imported`, `result.duration_ms`, `result.metadata.tags` | Sample prints every public field of the `ImportResult`. |

## Expected output

On success:

```text
Benchmark id: <uuid-or-id-string>
Records imported: 3
Duration: <float> ms
Tags: sample, qa
```

Exit code: `0`.

On failure:

```text
Import failed: [<error1>, <error2>, ...]
```

Exit code: `1`.

## Multi-tenancy note

This sample does not pass `org_id` to `BenchmarkImportAdapter`. The
constructor does not yet accept `org_id` — production multi-tenant
wiring lands with the PR #118 adapter-side contract (currently
DRAFT). For batch imports, the multi-tenant story is to scope the
`ImportResult.benchmark_id` to an `org_id` on the atlas-app side
when the record is persisted; the importer itself does not need
the org context to construct the dataset.
