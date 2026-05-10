# Benchmark import (data importer)

`layerlens.instrument.adapters.frameworks.benchmark_import.BenchmarkImportAdapter`
imports external benchmark datasets into Stratix evaluation spaces. Unlike
the other classes in `layerlens.instrument.adapters.frameworks.*`, this is
a **data importer**, not a runtime instrumentation adapter — it reads
benchmarks from disk or from HuggingFace and produces normalized rows.

> **Architectural note:** `BenchmarkImportAdapter` is deliberately a
> **bare class** — it does NOT extend `BaseAdapter`
> (`src/layerlens/instrument/adapters/frameworks/benchmark_import/adapter.py:70`
> declares `class BenchmarkImportAdapter:` with no superclass). It has no
> `connect()` / `disconnect()` lifecycle, no `AdapterCapability`
> declarations, no sinks, and emits no telemetry events. It is a one-shot
> ETL pipeline rather than a runtime instrumentation adapter. See the
> canonical specification at
> [`docs/adapters/benchmark_import.md`](./benchmark_import.md) §1 for the
> full architectural carve-out.

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

**Schema mapping is explicit-only in v1.x.** When no `schema_mapping` is
provided, source field names pass through unchanged —
`_apply_schema_mapping` short-circuits with `if not mapping: return record`
(`src/layerlens/instrument/adapters/frameworks/benchmark_import/adapter.py:423-434`).
There is **no automatic heuristic detection** (no case-insensitive
matching, no fuzzy aliasing) in the current implementation. To map a
source column to a canonical field, you must list it explicitly in
`schema_mapping`.

Automatic heuristic detection is on the v1.8 roadmap — see
[`docs/adapters/benchmark_import.md`](./benchmark_import.md) §3.3 and §7
("v1.8 — Schema-heuristic detection").

## Persistence

If you pass a `store=` argument to `BenchmarkImportAdapter(...)`, the
adapter persists imported benchmarks through it. The `store` is
duck-typed: it must expose a single method
`insert_row(table_name: str, row: dict)`. The adapter writes one row to
the `"benchmarks"` table containing `metadata.model_dump()`, then one
row per imported record to the `"benchmark_records"` table (each record
is augmented with `record["benchmark_id"] = metadata.benchmark_id`
before insertion). See
`src/layerlens/instrument/adapters/frameworks/benchmark_import/adapter.py:436-446`
for the `_persist` implementation. Persistence failures are swallowed
and logged at debug level — `import_*` will still return
`success=True` even if the store raises.

If `store` is `None`, the adapter only retains `BenchmarkMetadata`
objects in-memory (`adapter._benchmarks`, keyed by `benchmark_id`, used
by `list_benchmarks()` / `get_benchmark()`). The imported records
themselves are NOT retained on the adapter — they are produced inside
`import_*`, returned via `ImportResult.records_imported` (a count, not
the rows), and otherwise discarded. Callers that need the rows without
a real store should pass a shim whose `insert_row` captures them (e.g.
an in-test list).

For the deeper persistence contract, see
[`docs/adapters/benchmark_import.md`](./benchmark_import.md) §3.4
("Persistence contract").

## Events emitted

This adapter does not emit telemetry events — it produces benchmark rows.
Once stored in atlas-app, the platform's evaluation runner can iterate the
benchmark and produce `model.invoke` / `evaluation.score` events through
the standard provider adapters.

## BYOK

Not applicable. The adapter reads files locally or downloads from
HuggingFace using the standard `datasets` library — no model API keys are
involved.
