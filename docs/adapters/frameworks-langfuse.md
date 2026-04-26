# Langfuse framework adapter

`layerlens.instrument.adapters.frameworks.langfuse.LangfuseAdapter` is a
**bidirectional trace import / export pipeline** between LayerLens and a
[Langfuse](https://langfuse.com) instance (cloud or self-hosted).

Unlike the runtime-wrapping framework adapters (LangChain, LangGraph,
CrewAI, AutoGen, etc.), the Langfuse adapter does not monkey-patch any
in-process Python SDK. Langfuse is itself a remote observability
backend, so the adapter speaks to its REST API in batch:

* **Import**: pull historical Langfuse traces (with their nested
  observations) into LayerLens canonical events. Useful for backfilling
  an existing Langfuse-instrumented agent into LayerLens for replay,
  scoring, or cost analytics without re-running the agent.
* **Export**: push LayerLens canonical events back into Langfuse so a
  team that already standardised on Langfuse for inspection can
  continue using their existing Langfuse UI workflow.
* **Bidirectional**: run import + export in a single sync cycle with
  cursor-based incremental progress and per-trace loop prevention.

## Install

```bash
pip install 'layerlens[langfuse-importer]'
```

The `[langfuse-importer]` extra is **empty** — the adapter talks to a
remote REST surface via `urllib` from the Python stdlib and pulls no
additional dependencies. The extra is declared anyway so the adapter
remains explicitly opt-in and the install set for `pip install
layerlens` is unchanged.

You will need a Langfuse instance and a key pair from
[**Langfuse → Settings → Project Settings → API Keys**](https://langfuse.com/docs/get-started):

| Setting        | Where to find it                                      |
|----------------|-------------------------------------------------------|
| `public_key`   | Langfuse UI → Project Settings → API Keys → "pk-lf-…" |
| `secret_key`   | Langfuse UI → Project Settings → API Keys → "sk-lf-…" |
| `host`         | Your Langfuse base URL (default `https://cloud.langfuse.com`) |

The adapter authenticates via HTTP **Basic auth**
(`base64(public_key:secret_key)`) — there is no separate OAuth flow.

## Quick start

```python
from datetime import datetime, timezone

from layerlens.instrument.adapters.frameworks.langfuse import LangfuseAdapter
from layerlens.instrument.adapters.frameworks.langfuse.config import (
    LangfuseConfig,
    SyncDirection,
)

config = LangfuseConfig(
    public_key="pk-lf-...",
    secret_key="sk-lf-...",
    host="https://cloud.langfuse.com",
    mode=SyncDirection.IMPORT,        # or EXPORT, or BIDIRECTIONAL
    page_size=50,
    max_retries=3,
)

adapter = LangfuseAdapter(stratix=my_stratix_client, config=config)
adapter.connect()                     # runs a health check against /api/public/health

# --- Import: backfill traces from Langfuse into LayerLens ---
result = adapter.import_traces(
    since=datetime(2026, 1, 1, tzinfo=timezone.utc),
    tags=["production"],
    limit=200,
)
print(
    f"Imported {result.imported_count} traces, "
    f"skipped {result.skipped_count}, failed {result.failed_count}"
)

# --- Export: push LayerLens canonical events back to Langfuse ---
events_by_trace = {
    "trace-001": [
        {"event_type": "agent.input",  "trace_id": "trace-001", "timestamp": "...", "sequence_id": 0, "payload": {...}},
        {"event_type": "model.invoke", "trace_id": "trace-001", "timestamp": "...", "sequence_id": 1, "payload": {...}},
        {"event_type": "agent.output", "trace_id": "trace-001", "timestamp": "...", "sequence_id": 2, "payload": {...}},
    ],
}
export_result = adapter.export_traces(events_by_trace=events_by_trace)
print(f"Exported {export_result.exported_count} traces to Langfuse")

adapter.disconnect()
```

A fully runnable, mocked end-to-end sample lives in
[`samples/instrument/langfuse/`](../../samples/instrument/langfuse/).

## Pipeline architecture

The adapter is split into per-concern modules under
`layerlens.instrument.adapters.frameworks.langfuse`:

| Module          | Purpose                                                  |
|-----------------|----------------------------------------------------------|
| `lifecycle.py`  | `LangfuseAdapter` — `BaseAdapter` subclass, public API   |
| `config.py`     | `LangfuseConfig`, `SyncState`, `SyncResult`, enums       |
| `client.py`     | `LangfuseAPIClient` — stdlib-only HTTP client            |
| `importer.py`   | `TraceImporter` — Langfuse → LayerLens batch backfill    |
| `exporter.py`   | `TraceExporter` — LayerLens → Langfuse batch push        |
| `mapper.py`     | Bidirectional trace ↔ canonical event mapping            |
| `sync.py`       | `BidirectionalSync` — cursor-tracked combined cycle      |

### Import pipeline (Langfuse → LayerLens)

```
LangfuseAPIClient.get_all_traces(...)          # paginated /api/public/traces
        │
        ▼  (per trace summary)
TraceImporter._is_quarantined / dedup / loop-prevention checks
        │
        ▼
LangfuseAPIClient.get_trace(trace_id)          # full trace + observations
        │
        ▼
LangfuseToLayerLensMapper.map_trace(...)       # → list[canonical event dict]
        │
        ▼
stratix.emit(event_type, payload)              # injected into LayerLens pipeline
        │
        ▼
SyncState.record_import(...)                   # cursor + dedup set updated
```

Loop prevention: traces previously exported by LayerLens are tagged with
both `layerlens-exported` (canonical) and `stratix-exported` (legacy
alias for backward compatibility). The importer skips any trace
carrying either tag so an export → import round-trip does not double-
ingest.

### Export pipeline (LayerLens → Langfuse)

```
LayerLensToLangfuseMapper.map_events_to_trace(events, trace_id)
        │
        ▼
LangfuseAPIClient.ingestion_batch([...])       # single /api/public/ingestion call
        │   (one trace-create + N {generation,span}-create events)
        ▼
SyncState.record_export(trace_id, datetime.now(UTC))
```

Quarantine + retries: the `LangfuseAPIClient` retries `429` and `5xx`
with capped exponential backoff (1s → 16s, configurable via
`LangfuseConfig.max_retries`). Per-trace failures are tracked in
`SyncState.quarantined_trace_ids`; traces that fail three times are
removed from subsequent import cycles until `clear_quarantine()` is
called.

## Event mapping

### Langfuse → LayerLens

| Langfuse                                     | LayerLens (canonical)              |
|----------------------------------------------|------------------------------------|
| `trace.input`                                | `agent.input`        (L1)          |
| `trace.output`                               | `agent.output`       (L1)          |
| `trace.metadata`                             | `environment.config` (L4a)         |
| `observation` type=`GENERATION`              | `model.invoke`       (L3)          |
| `observation` type=`GENERATION` + `totalCost`| `cost.record`        (cross)       |
| `observation` type=`SPAN` (tool/metadata)    | `tool.call`          (L5a)         |
| `observation` type=`SPAN` (other)            | `agent.code`         (L2)          |
| `observation` `level`=`ERROR`/`WARNING`      | `policy.violation`   (cross)       |

Trace-level Langfuse metadata (`sessionId`, `userId`, `tags`, `scores`)
is propagated onto every emitted canonical event under the
`metadata.langfuse_*` namespace.

### LayerLens → Langfuse

| LayerLens (canonical)  | Langfuse                                    |
|------------------------|---------------------------------------------|
| `agent.input`          | `trace.input` (+ optionally `trace.name`)   |
| `agent.output`         | `trace.output`                              |
| `environment.config`   | `trace.metadata.environment_config`         |
| `model.invoke`         | observation type=`GENERATION`               |
| `cost.record`          | `totalCost` attached to matching generation |
| `tool.call`            | observation type=`SPAN`, `metadata.type=TOOL` |
| `agent.code`           | observation type=`SPAN`                     |
| `agent.handoff`        | observation type=`SPAN`, `metadata.type=HANDOFF` |
| `agent.state.change`   | observation type=`SPAN`, `metadata.type=STATE_CHANGE` |

Exported traces are tagged with both `layerlens-exported` and
`stratix-exported` for loop prevention (see above).

## Capability matrix

| Capability                           | Supported          | Notes                                                                 |
|--------------------------------------|--------------------|-----------------------------------------------------------------------|
| `AdapterCapability.TRACE_TOOLS`      | yes                | Langfuse `SPAN` observations (incl. tool spans) → `tool.call`         |
| `AdapterCapability.TRACE_MODELS`     | yes                | Langfuse `GENERATION` observations → `model.invoke` + `cost.record`   |
| `AdapterCapability.REPLAY`           | yes                | `serialize_for_replay()` returns a `ReplayableTrace` of emitted events |
| Real-time runtime hooks              | no                 | Langfuse is a remote backend — there is no in-process SDK to patch    |
| Incremental sync (cursor)            | yes                | `SyncState.last_import_cursor` / `last_export_cursor`                 |
| Per-trace deduplication              | yes                | `SyncState.imported_trace_ids` / `exported_trace_ids`                 |
| Per-trace quarantine after N fails   | yes                | Default 3 failures; configurable via `record_failure(max_failures=…)` |
| Loop prevention (export → import)    | yes                | `layerlens-exported` / `stratix-exported` tag round-trip              |
| Tag / project filtering on import    | yes                | `LangfuseConfig.tag_filter` and `LangfuseConfig.project_filter`       |
| Time-window filtering on import      | yes                | `since` parameter on `import_traces()` and `LangfuseConfig.since`     |
| Conflict resolution                  | last-write-wins    | Configurable via `LangfuseConfig.conflict_strategy` (also `MANUAL`)   |
| Auto-retry on `429` / `5xx`          | yes                | Capped exponential backoff (1s → 16s)                                 |
| `pip install layerlens` blast radius | zero               | `[langfuse-importer]` extra is empty (stdlib only)                    |

## Version compatibility

| Component                | Supported                          | Notes                                                  |
|--------------------------|------------------------------------|--------------------------------------------------------|
| Python                   | 3.9, 3.10, 3.11, 3.12, 3.13        | Same as the rest of `layerlens.instrument`             |
| Langfuse server / cloud  | API revision matching `/api/public/*` (Langfuse v2 / v3 generation) | Adapter calls only the **public** REST endpoints so it is forward-compatible across minor server versions |
| Pydantic                 | **v2 only**                        | `LangfuseConfig` uses `field_validator`; importing the adapter under Pydantic v1 raises a clear error from `pydantic_compat.requires_pydantic(...)` |
| Langfuse Python SDK      | not required                       | The adapter does **not** depend on the `langfuse` Python package — it is a pure HTTP client |
| Networking               | `urllib.request` (stdlib)          | No `requests` / `httpx` dependency                     |

`LangfuseAdapter` exposes its Pydantic compatibility hint as
`requires_pydantic = PydanticCompat.V2_ONLY` so the manifest emitter
can surface this in the atlas-app catalog UI before customers pin an
incompatible runtime.

## Capture config

The adapter respects the standard `CaptureConfig` filter from
`layerlens.instrument.adapters._base`:

```python
from layerlens.instrument.adapters._base import CaptureConfig

# Recommended for compliance backfills.
adapter = LangfuseAdapter(
    config=config,
    capture_config=CaptureConfig.standard(),
)

# Hand-rolled — keep tokens / costs but redact prompt and completion content.
adapter = LangfuseAdapter(
    config=config,
    capture_config=CaptureConfig(
        l3_model_metadata=True,
        capture_content=False,
    ),
)
```

The capture config is applied at the `BaseAdapter` emission layer, not
inside the Langfuse mapper, so the same redaction rules apply
identically to every adapter in the suite.

## BYOK

The Langfuse adapter does not own any model API keys — Langfuse never
sees a model provider key. The Langfuse `public_key` / `secret_key`
pair is intended to live in the platform's BYOK store
(`byok_credentials` table — see `docs/adapters/byok.md`) once that
M1.B work ships, alongside the credentials for the runtime-wrapping
adapters.

## Replay

`adapter.serialize_for_replay()` returns a `ReplayableTrace` containing
every event the adapter emitted during the current process. Replay is a
re-emit operation: the adapter does not re-fetch from Langfuse during
replay.

```python
trace = adapter.serialize_for_replay()
# trace.events == list of canonical event dicts
# trace.metadata["sync_state"] == {"imported": N, "exported": M, "quarantined": K}
```

## Backward compatibility

Users coming from the `ateam` / `stratix` package layout (pre-LayerLens
rename) can keep importing the legacy class name:

```python
from layerlens.instrument.adapters.frameworks.langfuse import STRATIXLangfuseAdapter
```

`STRATIXLangfuseAdapter` is resolved via a PEP 562 module-level
`__getattr__` that emits a `DeprecationWarning` and returns
`LangfuseAdapter`. The alias will be removed in v2.0; new code should
import `LangfuseAdapter` directly.

The mapper additionally tags exported traces with **both**
`layerlens-exported` (canonical) and `stratix-exported` (legacy), and
the importer recognises **both** tags for loop prevention. This allows
a deployment to upgrade the adapter without re-importing traces that
were exported by the previous release.

## Operational notes

* **Rate limits.** Langfuse Cloud enforces per-project request quotas;
  the client retries `429` responses with capped exponential backoff
  (1s → 16s). Tune `LangfuseConfig.max_retries` (default `3`) for
  long-running backfills.
* **Cursor persistence.** `SyncState` is held in memory by default. To
  resume an incremental sync across process restarts, call
  `adapter.sync_state.model_dump()` on shutdown and re-hydrate the
  state on the next start.
* **Pagination.** `LangfuseConfig.page_size` (default `50`) controls
  how many traces are fetched per `/api/public/traces` request. The
  importer transparently iterates until exhausted.
* **Quarantine.** Per-trace failures are tracked in
  `SyncState.quarantined_trace_ids`; after three failures the trace is
  skipped on future imports. Call
  `adapter.sync_state.clear_quarantine(trace_id=...)` to retry a
  specific trace, or `clear_quarantine()` (no argument) to clear all
  quarantined traces.

## See also

* [`samples/instrument/langfuse/`](../../samples/instrument/langfuse/) — runnable mocked sample
* [`docs/adapters/frameworks-langchain.md`](frameworks-langchain.md) — runtime callback adapter for LangChain
* [`docs/adapters/byok.md`](byok.md) *(planned)* — credential storage for Langfuse keys
