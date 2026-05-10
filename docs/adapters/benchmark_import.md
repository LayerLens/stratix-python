# Benchmark import adapter — specification

> Specification document for `layerlens.instrument.adapters.frameworks.benchmark_import`.
> Companion to the user-facing reference at `docs/adapters/frameworks-benchmark_import.md`.
> The depth audit at `A:/tmp/adapter-depth-audit.md` §1.18 flagged this adapter
> as having no spec; this document closes that gap.

| | |
|---|---|
| Spec ID | ADP-074 (benchmark import) |
| Module | `layerlens.instrument.adapters.frameworks.benchmark_import` |
| Source | `src/layerlens/instrument/adapters/frameworks/benchmark_import/adapter.py` |
| Adapter type | **Data importer** — does NOT extend `BaseAdapter`; not a runtime instrumentation adapter |
| Status | Functional, ports byte-near-identical from ateam |
| Tests | None dedicated — covered by `tests/instrument/adapters/frameworks/test_bulk_ported_smoke.py::test_benchmark_import_adapter_independent` |

---

## 1. Overview

The benchmark import adapter is a **data importer**. It reads benchmark
suites from external sources (HuggingFace Datasets, HELM result JSON,
local CSV/JSON/Parquet files), normalizes the records to LayerLens'
canonical evaluation schema, and persists them so the platform's
evaluation runner can consume them like any other dataset.

This adapter is **architecturally distinct** from every other adapter in
`frameworks/`. The other 17 are runtime instrumentation adapters that
extend `BaseAdapter`, wrap an in-process framework, and emit telemetry
events as the wrapped framework runs. `BenchmarkImportAdapter` does none
of that:

* It does NOT extend `BaseAdapter`. It is a standalone class.
* It does NOT have a `connect()` / `disconnect()` lifecycle.
* It does NOT emit any telemetry events.
* It does NOT wrap any third-party framework's runtime methods.
* It is closer in shape to the `langfuse` adapter's `importer.py` submodule
  than to LangChain's callback handler.

In ATEAM the smoke test for this adapter
(`test_benchmark_import_adapter_independent`) explicitly documents this
non-`BaseAdapter` shape; the audit (§1.18) classifies the architectural
inconsistency as intentional but flags that the framework-architecture
spec lacks a "data importer" carve-out for it. This spec provides that
carve-out.

The adapter exists because LayerLens evaluation spaces need a way to
import published benchmark suites — MMLU, SQuAD, HumanEval, and similar
HuggingFace datasets — without forcing every customer to write their own
ETL. The downstream evaluation runner (in atlas-app) iterates the
imported benchmark and produces the `model.invoke` / `evaluation.score`
events through standard provider adapters.

---

## 2. Capability surface

`BenchmarkImportAdapter` does NOT use the `AdapterCapability` enum and
does NOT publish an `AdapterInfo` record. The capability surface is
expressed instead through the public method set listed in §3.

What the adapter CAN do, sourced directly from `adapter.py`:

| Capability                                     | Source           |
|------------------------------------------------|------------------|
| Stream a HuggingFace dataset split             | `import_huggingface` (line 101–171). Uses `datasets.load_dataset(streaming=True)` to avoid materializing the whole split. |
| Parse HELM JSON results                        | `import_helm` (line 175–246). Handles the top-level-list, `{results: …}`, and `{scenarios: […]}` shapes. |
| Read a CSV file                                | `import_csv` (line 250–295). Uses stdlib `csv.DictReader`; `delimiter` is configurable. |
| Read a JSON file (array or object-with-key)    | `import_json` (line 297–349). The `records_key` argument selects the array inside an object payload (defaults to `records`, falling back to `data`). |
| Read a Parquet file                            | `import_parquet` (line 351–405). Lazily imports `pyarrow.parquet`; returns a clear error if `pyarrow` isn't installed. |
| Apply schema-mapping renames                   | `_apply_schema_mapping` (line 419–431). Pure key-renaming; no value transforms. |
| List / look up imported benchmarks             | `list_benchmarks` (line 409), `get_benchmark` (line 413). |
| Persist via an injected store                  | `_persist` (line 433–443). If the constructor receives a `store=` argument with `insert_row(table, row)`, the adapter writes one row to the `benchmarks` table and one row per record to the `benchmark_records` table. |

---

## 3. Contract

### 3.1 Public API

| Member                                          | Description                                                                |
|-------------------------------------------------|----------------------------------------------------------------------------|
| `__init__(store: Any \| None = None)`           | Construct with optional persistent store. Without a store, imports stay in memory. |
| `import_huggingface(dataset_name, split="test", subset=None, schema_mapping=None, max_records=None, tags=None) -> ImportResult` | Streamed HuggingFace import. |
| `import_helm(path, schema_mapping=None, tags=None) -> ImportResult` | HELM JSON import.            |
| `import_csv(path, schema_mapping=None, delimiter=",", max_records=None, tags=None) -> ImportResult` | CSV import. |
| `import_json(path, schema_mapping=None, records_key=None, max_records=None, tags=None) -> ImportResult` | JSON import. |
| `import_parquet(path, schema_mapping=None, max_records=None, tags=None) -> ImportResult` | Parquet import. |
| `list_benchmarks() -> list[BenchmarkMetadata]`  | All metadata held in this adapter instance.                                |
| `get_benchmark(benchmark_id) -> BenchmarkMetadata \| None` | Single lookup.                                                  |

### 3.2 Public data models

`BenchmarkMetadata` and `ImportResult` are Pydantic v1/v2-compatible
models defined in the same module (`adapter.py:34–63`).

`BenchmarkMetadata`:

| Field                | Type               | Default                                    |
|----------------------|--------------------|--------------------------------------------|
| `benchmark_id`       | `str`              | `f"bench-{uuid4().hex[:12]}"`              |
| `name`               | `str`              | required                                   |
| `source`             | `str`              | required (`huggingface`/`helm`/`csv`/`json`/`parquet`) |
| `source_identifier`  | `str`              | empty                                      |
| `version`            | `str`              | `"1.0.0"`                                  |
| `record_count`       | `int`              | 0                                          |
| `schema_mapping`     | `dict[str, str]`   | `{}`                                       |
| `imported_at`        | `str` (ISO-8601)   | `datetime.now(UTC).isoformat()`            |
| `imported_by`        | `str`              | empty                                      |
| `tags`               | `list[str]`        | `[]`                                       |

`ImportResult`:

| Field              | Type                          | Default |
|--------------------|-------------------------------|---------|
| `success`          | `bool`                        | `True`  |
| `benchmark_id`     | `str`                         | empty   |
| `records_imported` | `int`                         | 0       |
| `records_skipped`  | `int`                         | 0       |
| `duration_ms`      | `float`                       | 0.0     |
| `errors`           | `list[str]`                   | `[]`    |
| `metadata`         | `Optional[BenchmarkMetadata]` | `None`  |

### 3.3 Schema mapping

`schema_mapping` is a `dict[str, str]` of `source_field → canonical_field`.
The mapping is applied identically across all five import methods via
`_apply_schema_mapping`. The mapping is a pure key rename — values pass
through unmodified, and source keys NOT in the mapping pass through with
their original names.

There is **no automatic heuristic mapping** in the source. The
user-reference doc at `docs/adapters/frameworks-benchmark_import.md`
mentions "automatic heuristics" — this is not implemented in code today
(`_apply_schema_mapping` short-circuits to `return record` if the mapping
is None or empty). Treat that doc claim as aspirational pending a future
enhancement.

The recommended canonical fields, when imports will be consumed by the
LayerLens evaluation runner:

| Canonical field    | Common source field aliases                            |
|--------------------|--------------------------------------------------------|
| `prompt`           | `question`, `input`, `query`                           |
| `expected_output`  | `answer`, `target`, `reference`, `ground_truth`        |
| `difficulty`       | `difficulty`, `level`                                  |
| `category`         | `category`, `subject`, `topic`                         |

### 3.4 Persistence contract

The `store` constructor argument is duck-typed: any object with a single
method `insert_row(table_name: str, row: dict)` will be accepted. The
adapter writes:

1. One row to table `"benchmarks"` containing `metadata.model_dump()`.
2. One row per imported record to table `"benchmark_records"`, each
   record dict pre-augmented with `record["benchmark_id"] = metadata.benchmark_id`.

If `store` is `None`, the metadata is held in `self._benchmarks` (an
in-memory dict keyed by `benchmark_id`) but the records themselves are
NOT retained — they are produced and discarded inside the `import_*`
method. This is intentional: holding millions of HuggingFace records in
process memory would defeat the streaming load. Callers who need the
records without a store should pass a callable shim that captures them
(e.g. an in-test list).

### 3.5 Error handling

Every `import_*` method follows the same error-handling shape:

* `ImportError` from a lazy dependency import (`datasets`, `pyarrow`)
  → return `ImportResult(success=False, errors=["… library not installed …"])`.
* `FileNotFoundError`, `json.JSONDecodeError`, etc. → return
  `ImportResult(success=False, errors=[…])`.
* Any other exception → return `ImportResult(success=False, errors=[f"X import failed: {exc}"])`.

The methods do NOT raise. A caller can rely on the `result.success` flag
and the `result.errors` list. Persistence failures inside `_persist` are
swallowed and logged at DEBUG (`adapter.py:441–443`); the `ImportResult`
will still report `success=True` even though nothing reached the store.
This is a known gap — see §7 roadmap.

---

## 4. What we do NOT support

* **Not a `BaseAdapter`.** No `connect()`, `disconnect()`, `health_check()`,
  `get_adapter_info()`, `serialize_for_replay()`, sinks, capture config,
  or capability declarations. The class is freestanding.
* **No telemetry events.** `BenchmarkImportAdapter` does NOT emit
  `embedding.create`, `model.invoke`, `evaluation.score`, or any other
  event type. The downstream evaluation runner is responsible for
  generating events when it iterates the imported benchmark.
* **No live mid-run feedback.** This is a one-shot ETL: call
  `import_csv(...)`, get an `ImportResult`. There is no streaming
  interface, no progress callback, no per-batch event. For very large
  datasets the only way to bound memory is via `max_records`.
* **No MTEB importer.** The user-reference doc's parent-directory listing
  references "MTEB" generically; there is no `import_mteb(...)` method.
  MTEB datasets reachable through HuggingFace can be loaded via
  `import_huggingface("mteb/...")`, but there is no MTEB-specific
  schema mapping or score normalization. Adding native MTEB support is
  on the roadmap.
* **No MMLU-specific importer.** Same situation as MTEB — load it as a
  HuggingFace dataset (`import_huggingface("cais/mmlu", subset=...)`).
  No native MMLU-aware mapping ships today.
* **No BIG-bench, GPQA, HumanEval, SWE-bench, AGIEval, etc. native
  importers.** Same answer: any of these reachable through HuggingFace
  Datasets work via `import_huggingface`, but no benchmark-specific
  schema knowledge ships.
* **No automatic schema-heuristic detection.** The reference doc
  describes case-insensitive aliasing; the source does not implement it.
* **No cross-benchmark deduplication.** Reimporting the same dataset
  produces a fresh `benchmark_id` and a fresh row in the `benchmarks`
  table. There is no dedupe key.
* **No re-import / incremental update.** No cursor, no `since`, no diff
  mode. Every import is a fresh full-load (within `max_records`).
* **No per-record validation against the canonical evaluation schema.**
  The adapter renames keys via `schema_mapping`; it does NOT verify that
  a `prompt` field exists, that values are non-empty, or that types are
  what the evaluation runner expects.
* **No per-record signing or attestation.** Imported records are stored
  as plain rows. There is no merkle root, no source-hash, no provenance
  envelope.
* **No multi-tenant scoping inside the adapter.** The adapter has no
  `org_id` argument, does not stamp `org_id` onto rows, and does not
  inject tenant context into the `store.insert_row` call. Multi-tenant
  scoping is the responsibility of the **store** the caller injects —
  see §5.
* **No remote-source authentication.** HuggingFace public datasets work
  out of the box. Gated datasets require `HUGGINGFACE_HUB_TOKEN` set in
  the environment of the hosting process; the adapter does not surface
  this as an explicit kwarg.
* **No format-conversion outputs.** The adapter does not export to a
  different format. Records flow in, get persisted, and live in whatever
  shape the store keeps them.
* **No sample (in `samples/instrument/benchmark_import/`) for
  HuggingFace or HELM imports.** Only a CSV sample exists today.

---

## 5. BYOK and multi-tenancy

**BYOK is not applicable.** The adapter does not call any LLM provider
API and does not need a model API key. HuggingFace dataset loading uses
the `datasets` library's own credential resolution
(`HUGGINGFACE_HUB_TOKEN` env var) for gated datasets; this is outside
the LayerLens BYOK scope per the project's "BYOK = model API keys only"
convention.

**Multi-tenancy** is the responsibility of the injected `store`. The
adapter is tenant-agnostic by design — it does not know about
organizations, users, or tenant scoping. A multi-tenant deployment must:

1. Inject a `store` whose `insert_row` is itself tenant-aware (e.g.
   wraps the underlying SQL with a `tenant_id` column populated from
   request context).
2. Serialize benchmark imports per tenant — do NOT share a single
   `BenchmarkImportAdapter` instance across tenants because
   `self._benchmarks` (the in-memory metadata cache) does not carry
   tenant scoping. Callers running on a per-request basis should
   construct a fresh adapter per import.
3. If the platform later wants to surface "all benchmarks for org X",
   that lookup must go through the store, not through `list_benchmarks()`
   on a long-lived adapter instance.

The `imported_by` field on `BenchmarkMetadata` is meant to record the
user who triggered the import; it currently defaults to empty and must
be populated by the caller before persistence. The platform's CLI/API
layer is the right place to set it.

---

## 6. Test coverage

* **Dedicated tests:** none.
* **Bulk smoke coverage:** `tests/instrument/adapters/frameworks/test_bulk_ported_smoke.py::test_benchmark_import_adapter_independent`
  (lines 170–189). The smoke test exercises:
  * `BenchmarkMetadata(name="test", source="csv")` constructs and the
    `benchmark_id` has the expected `bench-` prefix.
  * `ImportResult(success=True, benchmark_id=…)` constructs.
  * `BenchmarkImportAdapter()` constructs.

  The smoke test does NOT exercise any actual import path. It is a
  module-importability and dataclass-shape check, not a behavioral test.

* **What's missing vs ateam:** ateam ships a dedicated integration test
  exercising at least the CSV path end-to-end (audit §1.18). That file
  was not ported. Restoring it is on the Tier-2 test-restoration queue
  (audit §3 item 16).

* **Sample:** `samples/instrument/benchmark_import/main.py` exercises
  `import_csv` end-to-end against a temporary CSV. There are no samples
  for HuggingFace, HELM, JSON, or Parquet paths.

A complete test pass for v1.7 would cover:

1. CSV import with explicit `schema_mapping`; verify `records_imported`,
   `metadata.source == "csv"`, all rows reachable through the schema
   mapping.
2. CSV import with `max_records=N`; verify the cap.
3. CSV import with custom `delimiter="\t"`.
4. JSON import — array form, object-with-`records`-key form,
   object-with-default-`data`-key form.
5. JSON import where the file is not valid JSON → `success=False` and
   a meaningful error string.
6. HELM import — the three accepted shapes (top-level list,
   `{results: …}`, `{scenarios: […]}`).
7. HELM import where the file is missing → `success=False`.
8. Parquet import without `pyarrow` installed → returns the expected
   error string. (Achievable in CI by uninstalling pyarrow in a separate
   tox env.)
9. HuggingFace import without `datasets` installed → returns the
   expected error string.
10. HuggingFace import with a small public dataset (`squad`,
    `max_records=2`) — gated behind `LAYERLENS_RUN_NETWORK_TESTS=1` so
    it doesn't run in default CI.
11. `_persist` with an injected stub store; verify the two table writes.
12. `_persist` failure path — store raises; verify the result is still
    returned (currently `success=True`, see §3.5 known gap).
13. `list_benchmarks` / `get_benchmark` round-trip after a successful
    import.

---

## 7. Roadmap

* **v1.7 — Native MTEB importer.** `import_mteb(task_name, ...)` that
  knows the MTEB task taxonomy and writes a benchmark per task. Likely
  built on top of `import_huggingface` with task-specific schema
  mappings.
* **v1.7 — Native MMLU importer.** Subject-aware mapping
  (`subject` → `category`), automatic answer-letter normalization
  (`A`/`B`/`C`/`D` → choice text).
* **v1.7 — `imported_by` plumbing.** Wire the CLI / API layer to
  populate `imported_by` from the authenticated user.
* **v1.7 — Persist-failure surface.** `_persist` errors should set
  `result.success = False` and append to `result.errors` instead of
  being swallowed at DEBUG.
* **v1.7 — Restored test parity with ateam** (Tier-2 item 16).
* **v1.8 — Schema-heuristic detection.** Implement the case-insensitive
  alias matching that the user-reference doc currently describes
  aspirationally.
* **v1.8 — Per-record validation.** Optional `validate=True` flag that
  rejects rows missing the canonical `prompt` field after mapping;
  populate `records_skipped` accordingly.
* **v1.8 — Source-hash provenance.** Stamp `metadata.source_hash =
  sha256(file_or_dataset_bytes)` for attestation. Required for
  audit-trail benchmarks shipped to regulated tenants.
* **v1.9 — Native HumanEval / SWE-bench / GPQA importers.** Mirrors the
  MMLU/MTEB pattern.
* **No-date-set — Incremental import.** Cursor-based reimport for
  benchmarks that grow over time. Practical only once the store layer
  exposes a `last_imported_record_id` lookup.
* **No-date-set — Cross-benchmark dedupe.** Decision pending product
  input on whether identical question text across benchmarks should
  collapse or remain distinct.

---

*End of spec.*
