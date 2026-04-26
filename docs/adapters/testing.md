# Testing the Instrument layer

The Instrument layer ships with three test tiers. CLAUDE.md is binding — every
test must fail when the feature is broken; tests that pass regardless of
behavior are flagged and removed.

## Tier 1 — Unit tests (fast, deterministic, mocked at SDK shape)

Path: `tests/instrument/test_base_layer.py`,
`tests/instrument/adapters/providers/test_openai_adapter.py`.

What they verify:

- `BaseAdapter` circuit breaker opens after 10 consecutive errors, recovers
  after the 60 s cooldown, and silently drops events while open.
- `CaptureConfig` gates events per layer; cross-cutting events bypass the
  gate; unknown layers default to disabled.
- `AdapterRegistry` is a singleton, lazy-loads adapter modules, and rejects
  classes without a `FRAMEWORK` class attribute.
- Provider adapters wrap the SDK client correctly and emit the expected event
  set (`model.invoke`, `cost.record`, `tool.call`, `policy.violation`).

What they do NOT catch:

- Real SDK schema drift (e.g., OpenAI renaming `usage.prompt_tokens`).
- Real network behavior (timeouts, rate limits, partial responses).
- Real streaming chunk sequences.

Tier 1 runs on every PR. Total runtime: ~20 s.

## Tier 2 — End-to-end transport (real HTTP, real bytes)

Path: `tests/instrument/test_sink_http_e2e.py`.

What they verify:

- `HttpEventSink` and `AsyncHttpEventSink` POST batches to a real
  `http.server.HTTPServer` bound on localhost — every byte traverses the
  loopback socket.
- The `X-API-Key` header reaches the server.
- Batching holds events until `max_batch` is reached, the flush interval
  elapses, or `close()` is called.
- Retries fire with exponential backoff on 5xx and 429.
- 4xx responses are dropped without retry.

These tests would FAIL if the sink ever stopped sending HTTP, sent the wrong
JSON shape, dropped the auth header, or got the retry policy wrong.

Tier 2 runs on every PR. Total runtime: ~3 s.

## Tier 3 — Live integration (real OpenAI, real cost, gated)

Path: `tests/instrument/adapters/providers/test_openai_adapter_live.py`.

Gated by `@pytest.mark.live` AND the presence of an `OPENAI_API_KEY` env var.
Skip cleanly otherwise.

What they verify:

- A real `chat.completions.create` call reaches OpenAI and the adapter routes
  the response through `HttpEventSink` to a localhost ingest server that
  mirrors the atlas-app contract.
- Real usage tokens from the response match the `model.invoke` payload —
  catches OpenAI SDK schema drift the moment it lands.
- Streaming consumption emits exactly one consolidated `model.invoke` on
  stream completion, regardless of chunk count.
- A real OpenAI error (invalid model name) produces both an error-variant
  `model.invoke` and a `policy.violation` event.

Tier 3 runs nightly via a separate CI workflow with the `OPENAI_API_KEY`
secret set. Cost per run: < $0.0001 (single-token completions). Same pattern
will be applied per adapter as more providers ship: nightly run hits a real
service, asserts on **structural invariants** (event types, required fields)
not exact byte values so the test stays stable across model output drift.

To run locally:

```bash
OPENAI_API_KEY=sk-... pytest tests/instrument/adapters/providers/test_openai_adapter_live.py -m live -v
```

## Per-adapter test matrix

Every new adapter ships with all three tiers:

| Adapter | Tier 1 (unit) | Tier 2 (transport e2e) | Tier 3 (live integration) |
|---|---|---|---|
| OpenAI provider | ✅ shipped | shared via HttpEventSink suite | ✅ shipped |
| Anthropic provider | ⏳ pending | shared | ⏳ pending |
| LangChain framework | ⏳ pending | shared | ⏳ pending |
| (other adapters) | per-adapter PR | shared | per-adapter PR |

The transport tier is shared — every adapter that uses `HttpEventSink` or
`AsyncHttpEventSink` benefits from the same e2e coverage on the wire format
and retry behavior.

## Cross-repo end-to-end (M1.D)

A separate suite under `atlas-app/e2e/cross-repo-adapters/` brings up the
real atlas-app stack via docker-compose, installs `layerlens[providers-openai]`
in a sidecar, runs a real OpenAI call through the adapter, and asserts the
events reach `/api/v1/adapters/health`. That suite is the gate on M1
completion. It is not in this repo.

## Default-install integrity

`tests/instrument/test_default_install.py` reads the installed package
metadata and asserts the runtime dependency list (`Requires-Dist` minus
extras) equals the canonical baseline. Adding extras MUST NOT grow the
default install.

## Lazy-import integrity

`tests/instrument/test_lazy_imports.py` imports `layerlens` and
`layerlens.instrument` and asserts no framework module (langchain, llama_index,
crewai, openai, anthropic, etc.) appears in `sys.modules`. The single
load-bearing guarantee of the v1.x stable client SDK.
