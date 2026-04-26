# Benchmark import framework adapter

`layerlens.instrument.adapters.frameworks.benchmark_import.BenchmarkImportAdapter`
imports external benchmark datasets into Stratix evaluation spaces. Unlike
the other framework adapters, this is a **data importer**, not a runtime
instrumentation adapter — it reads benchmarks from disk or from
HuggingFace and produces normalized rows.

## Install

```bash
pip install 'layerlens[benchmark-import]'
```

The `benchmark-import` extra has no required dependencies. To use the
HuggingFace import path, additionally install `datasets`:

```bash
pip install datasets
```

## Quick start (CSV)

```python
from layerlens.instrument.adapters.frameworks.benchmark_import import (
    BenchmarkImportAdapter,
)

adapter = BenchmarkImportAdapter()

result = adapter.import_csv(
    path="my_benchmark.csv",
    schema_mapping={"question": "prompt", "answer": "expected_output"},
    max_records=1000,
    tags=["custom", "qa"],
)

print(f"Imported {result.records_imported} records into {result.benchmark_id}")
```

## Quick start (HuggingFace)

```python
result = adapter.import_huggingface(
    dataset_name="squad",
    split="validation",
    max_records=200,
    tags=["public", "qa"],
)
```

## Quick start (HELM)

```python
result = adapter.import_helm(
    path="/path/to/helm_results.json",
    tags=["helm", "leaderboard"],
)
```

## Public API

| Method | Description |
|---|---|
| `import_huggingface(dataset_name, split=, subset=, schema_mapping=, max_records=, tags=)` | Stream a HuggingFace dataset into Stratix. |
| `import_helm(path, tags=)` | Import HELM JSON results. |
| `import_csv(path, schema_mapping=, delimiter=, max_records=, tags=)` | Import a CSV benchmark. |
| `import_json(path, schema_mapping=, records_key=, max_records=, tags=)` | Import a JSON benchmark. |
| `import_parquet(path, schema_mapping=, max_records=, tags=)` | Import a Parquet benchmark (requires `pyarrow`). |

All methods return `ImportResult` with `success`, `benchmark_id`,
`records_imported`, `records_skipped`, `duration_ms`, `errors`, and
`metadata` (a `BenchmarkMetadata` Pydantic model).

## Schema mapping

Supplying a `schema_mapping` dict renames source columns to the canonical
Stratix evaluation schema:

| Stratix field | Common source columns |
|---|---|
| `prompt` | `question`, `input`, `query` |
| `expected_output` | `answer`, `target`, `reference`, `ground_truth` |
| `difficulty` | `difficulty`, `level` |
| `category` | `category`, `subject`, `topic` |

When no mapping is provided, the adapter applies a small set of automatic
heuristics (case-insensitive name match against the canonical fields).

## Persistence

If you pass a `store=` argument to `BenchmarkImportAdapter(...)` (something
that exposes `save_benchmark(metadata, records)`), the adapter writes
imported benchmarks through it. Otherwise records are returned to the
caller and held in `adapter._benchmarks` keyed by `benchmark_id`.

## Events emitted

This adapter does not emit telemetry events — it produces benchmark rows.
Once stored in atlas-app, the platform's evaluation runner can iterate the
benchmark and produce `model.invoke` / `evaluation.score` events through
the standard provider adapters.

## BYOK

Not applicable. The adapter reads files locally or downloads from
HuggingFace using the standard `datasets` library — no model API keys are
involved.
