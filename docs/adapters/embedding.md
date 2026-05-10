# Embedding adapter — specification

> Specification document for `layerlens.instrument.adapters.frameworks.embedding`.
> Companion to the user-facing reference at `docs/adapters/frameworks-embedding.md`.
> This file describes WHAT the adapter is contracted to do, what it explicitly
> does NOT do, and what is on the roadmap. The depth audit at
> `A:/tmp/adapter-depth-audit.md` §1.17 flagged this adapter as having no
> dedicated spec; this document closes that gap.

| | |
|---|---|
| Spec ID | ADP-060 (embedding) + ADP-061 (vector store) |
| Module | `layerlens.instrument.adapters.frameworks.embedding` |
| Source | `src/layerlens/instrument/adapters/frameworks/embedding/` (`embedding_adapter.py`, `vector_store_adapter.py`) |
| Adapter type | Provider — runtime wrapping of embedding-API calls and vector-store queries |
| Status | Functional, ports byte-near-identical from ateam |
| Tests | None dedicated — covered indirectly via `tests/instrument/adapters/frameworks/test_bulk_ported_smoke.py` |

---

## 1. Overview

The embedding adapter is a runtime instrumentation adapter that wraps the
client-side methods used to **create vector embeddings** and **query vector
stores**. It targets the data-preparation and retrieval halves of a typical
RAG pipeline, not the generative half (the LLM call itself is covered by the
provider adapters in `layerlens.instrument.adapters.providers.*`).

The module ships two distinct `BaseAdapter` subclasses — they are independent
and either can be used without the other.

* **`EmbeddingAdapter`** — wraps client methods that turn text into vectors.
  Today: OpenAI `client.embeddings.create`, Cohere `client.embed`, and any
  `sentence_transformers.SentenceTransformer.encode`.
  *(Source: `embedding_adapter.py:118–143`.)*
* **`VectorStoreAdapter`** — wraps client methods that retrieve vectors by
  similarity. Today: Pinecone `Index.query`, Weaviate `collection.query.near_vector`
  and `collection.query.near_text`, and Chroma `collection.query`.
  *(Source: `vector_store_adapter.py:114–145`.)*

Both adapters use the same monkey-patch + restore pattern as the rest of the
framework adapters: `connect()` registers the wrapper, `disconnect()` calls
`_restore_originals()` to put the original methods back. They emit dict events
through `BaseAdapter.emit_dict_event` rather than typed Pydantic events; the
adapter is therefore subject to the cross-cutting "typed events" gap called out
in audit §2 finding 1.

This adapter is conceptually different from a future RAG-retrieval adapter:
it captures the API call to the embedding/vector-store backend, but it does
NOT correlate retrieved chunks with the downstream LLM call that consumed
them. That correlation is the responsibility of the agent-framework adapter
(LangChain, LlamaIndex, etc.) sitting above it.

---

## 2. Capability surface

Both classes declare a single capability via `AdapterInfo.capabilities`:

| Class              | Declared capability  | Source                              |
|--------------------|----------------------|-------------------------------------|
| `EmbeddingAdapter` | `TRACE_MODELS`       | `embedding_adapter.py:101–103`      |
| `VectorStoreAdapter` | `TRACE_TOOLS`      | `vector_store_adapter.py:97–99`     |

The choice of `TRACE_MODELS` for embeddings reflects that an embedding API
call is a model invocation against a hosted embedding model
(`text-embedding-3-small`, `embed-english-v3.0`, etc.). The choice of
`TRACE_TOOLS` for vector stores reflects that a similarity query is treated
by the schema as a retrieval tool call rather than a model call.

**Capabilities NOT declared:**

* `STREAMING` — embedding APIs return whole vectors per call; there is no
  intermediate token stream to capture. This is consistent with the
  underlying provider APIs (OpenAI `embeddings.create` returns a synchronous
  `CreateEmbeddingResponse`; Cohere `embed` returns a list; sentence-transformers
  returns a numpy/torch tensor in one shot).
* `REPLAY` — both classes implement `serialize_for_replay()` (returning a
  `ReplayableTrace` containing `self._trace_events`), but neither declares
  the capability. This is the same `serialize_for_replay()`-without-`REPLAY`-
  declaration gap that the depth audit flagged across all adapters
  (audit §2 finding 3); it is tracked in PR #119.
* `TRACE_HANDOFFS`, `TRACE_STATE`, `TRACE_PROTOCOL_EVENTS` — not relevant to
  the embedding/vector-store call surface.

---

## 3. Contract

### 3.1 Public API — `EmbeddingAdapter`

| Member                          | Description                                                                       |
|---------------------------------|-----------------------------------------------------------------------------------|
| `__init__(stratix=, capture_config=, *, org_id=)` | Standard `BaseAdapter` constructor.                            |
| `connect()`                     | Marks the adapter healthy; takes no other action (no networking, no discovery).   |
| `disconnect()`                  | Restores all originals, marks disconnected, closes sinks.                         |
| `health_check() -> AdapterHealth` | Returns adapter status (`HEALTHY` / `DISCONNECTED` / etc.) and error count.     |
| `get_adapter_info() -> AdapterInfo` | Returns name, version, framework, capabilities, author, description.          |
| `serialize_for_replay() -> ReplayableTrace` | Returns the in-memory `_trace_events` for replay.                     |
| `wrap_openai(client) -> client` | Patches `client.embeddings.create`. Returns the same client.                      |
| `wrap_cohere(client) -> client` | Patches `client.embed`. Returns the same client.                                  |
| `wrap_sentence_transformer(model) -> model` | Patches `model.encode`. Returns the same model.                       |

Each `wrap_*` method is **idempotent against missing methods** but **not
idempotent against repeat calls**: calling `wrap_openai(client)` twice on the
same client wraps the wrapper. Callers should `disconnect()` (which restores)
between repeated wrappings, or wrap each client exactly once.

### 3.2 Public API — `VectorStoreAdapter`

| Member                          | Description                                                                       |
|---------------------------------|-----------------------------------------------------------------------------------|
| Lifecycle                       | Same as `EmbeddingAdapter` (mirror methods).                                      |
| `wrap_pinecone(index) -> index` | Patches `index.query`.                                                            |
| `wrap_weaviate(collection) -> collection` | Patches `collection.query.near_vector` and `collection.query.near_text`.|
| `wrap_chroma(collection) -> collection` | Patches `collection.query`.                                             |

### 3.3 Events emitted

All events are emitted via `emit_dict_event(event_type, payload)`. They are
NOT typed Pydantic events; that migration is tracked under audit §3 Tier-1
recommendation 5.

#### `embedding.create` (L3)

Emitted by `EmbeddingAdapter` after every wrapped call returns.

| Field           | Type    | Source                                                                | Notes |
|-----------------|---------|-----------------------------------------------------------------------|-------|
| `provider`      | str     | Hardcoded per wrapper: `"openai"` / `"cohere"` / `"sentence_transformers"` | |
| `model`         | str     | `kwargs["model"]` (OpenAI/Cohere) or literal `"local"` (ST)          | OpenAI defaults to `"unknown"` if model not in kwargs (`embedding_adapter.py:151`). Cohere defaults to `"embed-english-v3.0"` (line 188). |
| `batch_size`    | int     | `len(input)` if list, else `1`                                        | |
| `dimensions`    | int \| None | `len(result.data[0].embedding)` (OpenAI), `len(result.embeddings[0])` (Cohere), `result.shape[1]` (ST) | None if response shape unrecognized. |
| `total_tokens`  | int     | `result.usage.total_tokens` (OpenAI only)                             | Cohere/ST do not surface token counts in the same way and the field is omitted from those payloads. |
| `latency_ms`    | float   | Wall-clock measurement around the wrapped call (`time.monotonic`)     | Rounded to 2 decimals. |

#### `retrieval.query` (L3)

Emitted by `VectorStoreAdapter` after every wrapped query returns. The
**event type is `retrieval.query`, not `vector_store.query`** — the existing
user-reference doc disagrees with the source on this point; the source
(`vector_store_adapter.py:169, 201, 235`) is the authority.

| Field           | Type    | Notes                                                                         |
|-----------------|---------|-------------------------------------------------------------------------------|
| `provider`      | str     | `"pinecone"` / `"weaviate"` / `"chroma"`                                      |
| `top_k` / `n_results` / `limit` | int | Field name varies by provider to match the underlying SDK kwarg. |
| `match_count` / `result_count` | int | Number of hits returned.                                          |
| `has_filter` / `has_where` | bool | Whether a metadata filter was supplied (Pinecone `filter`, Chroma `where`). |
| `namespace`     | str     | Pinecone-specific.                                                            |
| `query_type`    | str     | Weaviate-specific: `"near_vector"` or `"near_text"`.                          |
| `score_min`/`score_max`/`score_mean` | float \| None | Pinecone — extracted from `result.matches[*].score`.       |
| `distance_min`/`distance_max` | float \| None | Chroma — from `result["distances"][0]`.                       |
| `latency_ms`    | float   | Always present.                                                               |

### 3.4 Lifecycle

Both adapters inherit the standard `BaseAdapter` lifecycle. `connect()` is
trivial (no remote handshake, no SDK discovery — embedding/vector clients
are passed in by the caller). `disconnect()` restores wrapped methods via
`_restore_originals()` and is the only operation that matters for clean
teardown.

### 3.5 Error handling

The wrappers do NOT swallow exceptions. If the wrapped client raises, the
exception propagates to the caller. The wrapper's `start = time.monotonic()`
is reached but the `emit_dict_event` after the call is not, so failed calls
are NOT emitted as `embedding.create` / `retrieval.query` events. The
underlying `BaseAdapter._error_count` is also not incremented on these
exceptions because the wrappers do not invoke
`_record_error()`. This is a known gap shared with most ported adapters
(audit §2 finding for "no PolicyViolationEvent path").

---

## 4. What we do NOT support

Each item below is an explicit non-goal as of this spec. None of these
behaviors should be inferred from the source — they are not present.

* **No async embedding clients.** The OpenAI v1 SDK exposes
  `AsyncOpenAI().embeddings.create()`; this adapter only wraps the sync
  `OpenAI` client (`hasattr(client, "embeddings")` check at
  `embedding_adapter.py:120`). The async coroutine method is NOT patched.
  Async support is on the v1.7 roadmap.
* **No streaming.** Embedding APIs return whole vectors; there are no
  intermediate stream events. The `STREAMING` capability is not declared
  and never will be for this adapter.
* **No retrieval-correlation events.** Vector-store queries emit a single
  `retrieval.query` event with aggregate scores — not per-document
  correlation IDs. An LLM consuming the retrieved chunks has no way to
  reference them back to a specific `retrieval.query` event without help
  from the framework adapter above (LangChain/LlamaIndex).
* **No content capture.** The wrapper records *batch_size* but does not
  record the input texts themselves, nor the returned vectors. This is
  intentional for privacy reasons and is consistent with `CaptureConfig`'s
  `capture_content=False` default for production presets. There is no
  knob today to opt into capturing input text or returned vectors.
* **No re-ranker support.** Cohere `rerank`, Voyage `rerank`, and other
  re-ranking endpoints are NOT wrapped. Only `embed`/`encode` paths.
* **No Voyage / Mistral-embed / Anthropic-embed / NVIDIA-embed / GTE / BGE
  hosted endpoints.** Only the three providers listed in §1 are wrapped.
  Adding new providers requires a new `wrap_<provider>(client)` method
  plus a `_make_<provider>_wrapper(original)` factory mirroring the
  existing pattern.
* **No vector-store mutation tracking.** Pinecone `upsert` / `delete`,
  Weaviate `data.insert`, Chroma `add` / `update` are NOT wrapped. Only
  read-side `query` operations are instrumented. Tracking writes is on
  the v1.7 roadmap.
* **No Qdrant, Milvus, FAISS, or pgvector adapters.** Only Pinecone /
  Weaviate / Chroma are supported today.
* **No batch-vs-single distinction event.** OpenAI lets you submit a
  batch in a single `embeddings.create`; we record `batch_size` but do
  not split the call into per-item events.
* **No typed Pydantic event payloads.** All events are emitted as plain
  dicts via `emit_dict_event`. Migration to typed events is tracked at
  audit §3 Tier-1 item 5 (cross-adapter, not adapter-specific).
* **No OTel `gen_ai.*` semantic conventions.** Audit §2 finding 2 applies.
  PR #125 tracks the cross-adapter migration.
* **No `org_id` in event envelopes by default.** The constructor accepts
  `org_id=` and stores it on `self._org_id`, but the emit path does NOT
  inject it into the event payload. PR #118 tracks the cross-adapter fix.

---

## 5. BYOK and multi-tenancy

The embedding adapter integrates with provider keys indirectly. The caller
is responsible for constructing the upstream client (`OpenAI()`, `cohere.Client()`,
`SentenceTransformer(...)`); the adapter never instantiates a provider client
of its own and never reads `OPENAI_API_KEY` / `COHERE_API_KEY` / similar
environment variables. Whatever credential the caller's client has, that's
what the wrapped call uses.

This means BYOK key resolution for embeddings happens **outside** the
adapter's scope, in two layers above it:

1. The caller's application chooses which key to use (org-scoped BYOK key,
   platform-managed key, or environment fallback).
2. The platform's BYOK store (atlas-app `/api/v1/model-keys`) is responsible
   for materializing the per-org key into the OpenAI/Cohere client before
   the embedding call is made.

The adapter's only contribution to multi-tenancy is the `org_id`
constructor argument (`embedding_adapter.py:69`,
`vector_store_adapter.py:65`). Today this is stored on the adapter
instance but **not stamped onto emitted events** — the cross-adapter
`org_id` propagation work tracked by PR #118 is the fix.

For SaaS deployments where a single shared sink is fed by per-tenant
adapter instances, the recommended pattern until PR #118 lands is to use
distinct adapter instances per tenant and let the sink layer (which knows
its own tenant context from its API-key-driven session) attach the
`org_id` envelope. Do NOT share a single `EmbeddingAdapter` instance
across tenants.

---

## 6. Test coverage

* **Dedicated tests:** none.
* **Bulk smoke coverage:** `tests/instrument/adapters/frameworks/test_bulk_ported_smoke.py:65–67`
  imports `EmbeddingAdapter` and exercises construction, `connect()`,
  metadata, and capability declaration via `_PARAMETRIZE_CASES`. The
  smoke pass for `VectorStoreAdapter` is implicit — it inherits the same
  base class but is not explicitly listed in the parametrize cases.
* **What's missing vs ateam:** ateam ships `tests/adapters/embedding/test_integration.py`
  exercising provider wrapping with mocked clients (audit §1.17). That
  file was NOT ported. Restoring it is on the Tier-2 test-restoration
  queue (audit §3 item 16).
* **Sample:** `samples/instrument/embedding/main.py` exercises
  `wrap_openai` end-to-end with a real OpenAI client. There is no
  vector-store sample.

A complete test pass for v1.7 would cover, at minimum:

1. Construction with each `CaptureConfig` preset.
2. `wrap_openai(client)` with a stubbed client that returns a
   `CreateEmbeddingResponse`-shaped object — verify the emitted event has
   the expected `provider`, `model`, `batch_size`, `dimensions`,
   `total_tokens`, `latency_ms`.
3. Same for Cohere (`embed`) and sentence-transformers (`encode`).
4. `wrap_pinecone(index)` with a stub returning `matches=[Mock(score=…)]` —
   verify `score_min/max/mean` aggregation.
5. `wrap_weaviate(collection)` for both `near_vector` and `near_text`.
6. `wrap_chroma(collection)` with the dict-shaped response.
7. `disconnect()` restores all originals (assert `client.embeddings.create
   is original`).
8. Exception in wrapped call does NOT emit an event but does NOT swallow.
9. `serialize_for_replay()` returns a `ReplayableTrace` with the events
   accumulated during the session.
10. Pydantic-compat smoke (`requires_pydantic = PydanticCompat.V1_OR_V2`)
    holds for both classes.

---

## 7. Roadmap

The following items are explicitly planned for v1.7 or later. Nothing in
§1–§6 should be read as committing to any of them.

* **v1.7 — Async client support.** Add `wrap_async_openai(client)` and
  equivalents for AsyncCohere. Wrappers will be `async def` and `await`
  the underlying call before emitting.
* **v1.7 — Vector-store write instrumentation.** Wrap Pinecone `upsert`,
  Weaviate `data.insert`, Chroma `add`/`update`. Emit
  `vector_store.upsert` / `vector_store.delete` events.
* **v1.7 — Voyage AI provider.** Add `wrap_voyage(client)` matching the
  shape of the existing OpenAI wrapper.
* **v1.7 — Restored test parity with ateam** (Tier-2 item 16).
* **v1.8 — Qdrant + pgvector adapters.** Both have widely-used Python
  clients with stable query surfaces; wrapping is straightforward.
* **v1.8 — Typed Pydantic events** (cross-adapter, audit Tier-1 item 5).
  Define `EmbeddingCreateEvent` and `RetrievalQueryEvent` Pydantic models;
  switch the wrappers from `emit_dict_event` to `emit_event`.
* **v1.8 — OTel `gen_ai.*` semconv** (cross-adapter, audit Tier-1 item 6).
  Add parallel `gen_ai.system="openai"`, `gen_ai.request.model=…`,
  `gen_ai.usage.input_tokens=…` attributes alongside the existing flat
  fields.
* **No-date-set — Re-ranker support.** Cohere `rerank` and Voyage `rerank`
  are reasonable candidates. Decision pending product input on whether
  re-rankers should be modeled as `tool.call` or as a new
  `retrieval.rerank` event.
* **No-date-set — Per-chunk retrieval correlation.** Requires upstream
  framework adapter cooperation (LangChain / LlamaIndex / a future
  retrieval-orchestration adapter); not solvable inside this adapter
  alone.

---

*End of spec.*
