# Changelog

All notable changes to the Stratix Python SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

**Latest version:** [1.10.0](https://github.com/LayerLens/stratix-python/releases/tag/v1.10.0) — 2026-08-06

## [Unreleased]

Things we're actively working on. Want to help? Check the [issues](https://github.com/LayerLens/stratix-python/issues) or [discussions](https://github.com/LayerLens/stratix-python/discussions).

### Added

### Changed

### Fixed

### Deprecated

### Removed

## [1.10.0] - 2026-08-06

### Added

- **Token usage on evaluations and per-prompt results.** `Evaluation` gains `total_input_tokens`, `total_output_tokens`, `avg_input_tokens_per_prompt`, and `avg_output_tokens_per_prompt`; `Result` gains `input_tokens` and `output_tokens`. Aggregates count the evaluated model's successful attempts only (no failed retries, judge/grader calls, or prompt-cache tokens) and are `None` — not `0` — for runs that predate token capture
- README: `Adapters`, `Requirements`, `Versioning and Compatibility`, and `Data Handling` sections. The adapters section documents `auto()` / `discover_installed()`, explicit per-framework wiring, and offline hash-chain verification via `layerlens.attestation`

### Changed

- README: corrected the reference-data claim to 172 models and 78 benchmarks, replaced the PyPI badges (they pointed at an unrelated `layerlens` project on PyPI, not the private index), and refreshed the architecture tree to the real package layout

## [1.9.0] - 2026-07-30

The instrumentation release: `layerlens.instrument` captures agent traces from the frameworks, LLM providers, and agent protocols you already use, and ships them to LayerLens for trace evaluation.

### Added

- **`layerlens.instrument` — agent instrumentation and tracing engine.** Public API: `auto()`, `trace()`, `span()`, `emit()`, `TraceCollector`, `CaptureConfig`, `BaseAdapter`, `AdapterInfo`, and `discover_installed()`. `auto()` detects the agent libraries installed in the environment and wires up the matching adapters; `trace()` / `span()` mark run and step boundaries by hand when you want explicit control
- **39 adapters** across three layers:
  - 8 LLM providers — OpenAI, Anthropic, Azure OpenAI, Bedrock (including Amazon Nova `invoke_model`), Google Vertex, LiteLLM, Ollama, OpenRouter
  - 25 frameworks — LangChain, LangGraph, LlamaIndex, CrewAI, AutoGen, Agno, Haystack, Semantic Kernel, OpenAI Agents, Google ADK, Bedrock Agents, Agentforce, Strands, smolagents, Pydantic AI, Microsoft Agent Framework, DSPy, Instructor, Marvin, Mirascope, OpenInference, browser-use, Langfuse, embedding, vector store
  - 6 protocols — MCP, A2A, AG-UI, A2UI, AP2, UCP
- **W3C trace context propagation** — `inject_headers()`, `extract_headers()`, `get_trace_context()`, `new_traceparent()`, and the `trace_context` context manager, so a trace survives a hop across services and protocol boundaries
- **Agent-graph contract** — adapters emit `agent_name` and handoff edges, so multi-agent runs reconstruct as a real DAG instead of a flat span list
- **Attestation** — every wire event carries a per-event hash for OTLP conformance and tamper evidence
- **Cost tracking** — a spend ledger and provider pricing tables attach `cost_usd` to captured runs, including costs the framework reports itself
- **Upload data-loss observability** — `set_upload_loss_callback()` and `get_upload_loss_stats()` surface dropped events instead of failing silently
- **`strict` flag on `traces.get()` / `traces.get_many()`** (sync + async, default `False`). When `True`, a 200 response with an empty or unparseable body raises `StratixError` instead of returning `None`, distinguishing contract drift from a genuine miss. A real 404 still raises `NotFoundError`
- CLI: `judge result` command; trace-evaluation IDs no longer get routed to `evaluate get`

### Changed

- **BREAKING (`capture_content=False` only):** `model.invoke.parameters` redaction is now deny-by-default. The collector previously stripped a fixed deny-list, so any parameter the SDK had not seen before passed through and could carry prompt or response content. It now keeps only a vetted allowlist of non-content metrics (sampling and limit parameters), recursing into nested containers like `generation_config` and `options` so safe sub-keys survive and content sub-keys are dropped. Impact is metadata loss on custom or provider-specific parameters, never a content leak. The default `capture_content=True` path is unchanged
- Every resource method now raises from the SDK exception taxonomy. `_request_cast` previously mapped only `httpx.HTTPStatusError`, letting raw transport and decode failures escape; timeouts now surface as `APITimeoutError`, transport errors as `APIConnectionError`, and response decode/validation failures as `APIResponseValidationError`
- Per-event byte cap on captured events, and upload filenames are sanitized fail-fast
- Dropped the `browser-use` extra — its `openai` pin conflicts with the SDK's. The browser-use adapter still works when you install the package yourself

### Fixed

- **Privacy — `capture_content=False` leaks closed across all adapters**. Follow-ups: Google ADK system prompts and CrewAI task descriptions, protocol-layer content gating and redaction, and arbitrary user-supplied trace metadata in the Langfuse adapter
- **SSRF guard on presigned uploads** — the upload target is validated before the request goes out
- Async provider clients are routed onto the async wrap path instead of the sync one
- Per-adapter concurrent-run isolation — parallel runs no longer bleed spans into each other
- Provider-only traces emit a real captured root span rather than a synthesized placeholder
- `agent.identity` is captured canonically at flush time
- LangGraph and LangChain event serialization
- Agentforce importer rewritten against the real Salesforce STDM, and `bedrock_agents` rewritten against the real `InvokeAgent` completion EventStream
- `autogen`, `crewai`, and `llamaindex` now honor a caller-bound collector instead of falling back to the global one
- A2A: protobuf `TaskStatus.state` enum now maps to the canonical status string
- Telemetry fidelity for LiteLLM streaming; adapter schema-lock re-arm, Vertex/Azure coverage, vector-store and provider linkage

## [1.8.0] - 2026-05-26

### Added

- `benchmark_key`, `model_key_1`, and `model_key_2` parameters on `comparisons.compare_models` (sync + async). Address the benchmark and the two models by their unique key (e.g., `aime2024`, `openai/gpt-4o`) instead of by UUID; the existing `*_id` parameters keep working. Exactly one of `*_id` or `*_key` must be provided per entity — passing both, or neither, raises `ValueError`. Unknown keys raise `ValueError` with the offending key in the message.

## [1.7.0] - 2026-05-20

### Added

- `extra_payload` parameter on `models.create_custom` and `models.update_custom` (sync + async). Optional JSON object merged into every outgoing chat-completions request body; customer values win on conflict with our hardcoded defaults. Lets customers add provider-specific fields (`top_p`, `max_completion_tokens`) or override values like `temperature` for providers that reject our defaults.

## [1.6.1] - 2026-05-15

### Added

- CLI authentication command (`layerlens auth`) (#72)
- `models.update_custom(model_id, *, api_url, api_key, max_tokens)` (sync + async) — repoint a custom model's mutable fields without recreating it (#169)
- `models.delete_custom(model_id)` (sync + async) — full teardown that disables the record, strips it from `Project.Models`, and releases the name for reuse (#169)
- 70+ production-ready SDK samples across 12 categories: core, industry, cowork, modalities, integrations, cicd, cli, openclaw, mcp, copilotkit, claude-code, data (#73)
- MCP server sample exposing LayerLens as tools
- CopilotKit sample with LangGraph CoAgents, React components, and hooks
- New trace samples (#144)

### Changed

- `models.add()` / `models.remove()` now operate on the full project model list (public + custom). The previous `type="public"` filter silently dropped custom-model IDs from `Project.Models` on every call (#169)
- Expanded SDK documentation and README (#139, #167)

### Fixed

- Trace evaluations bug (#74)
- CopilotKit evaluator graph now compiles with a checkpointer so `interrupt()` works over AG-UI. Includes a `RunIdPreservingAgent` workaround for the upstream `ag-ui-langgraph` runId-overwrite bug ([ag-ui-protocol/ag-ui#1582](https://github.com/ag-ui-protocol/ag-ui/issues/1582)) (#92)

## [1.6.0] - 2026-03-25

### Added

- Prompts exposed on the private client (#70)

## [1.5.0] - 2026-03-23

### Added

- Full-featured command-line interface via `layerlens` / `stratix`
- `client.scorers` resource with full CRUD: create, get, list, update, delete
- `client.evaluation_spaces` resource with get, list, create, update, delete
- `client.integrations` resource with get, list, create, update, delete, and test
- CLI getting started guide, command reference, and examples
- Scorers API reference documentation

### Changed

- Updated evaluations, models & benchmarks, and public client docs with new parameters

### Fixed

- `filter` by categories/languages/companies/regions/licenses now returns correct results

## [1.4.0] - 2026-03-17

### Added

- `unique` parameter on `evaluations.get_many()` and `public_evaluations.get_many()` that deduplicates results by model+dataset pair, keeping only the latest evaluation per pair

### Fixed

- Model comparison now passes `unique=True` when fetching evaluations, ensuring the correct (latest) evaluation is used for each model+benchmark pair instead of potentially picking up duplicates

## [1.3.3] - 2026-03-17

### Added

- Missing methods on `benchmarks` and `models` resources

### Fixed

- Inconsistent API naming across the SDK now follows a unified convention. Affected resources: comparisons, evaluations, judges, results, trace evaluations, traces, public benchmarks/evaluations/models (#61)
- `SUMMARY.md` structure and examples updated to match new naming

## [1.3.2] - 2026-03-13

### Added

- Documentation pages for GitBook: getting-started, troubleshooting, security

### Fixed

- `trace_evaluations.get_results()` no longer returns empty/None results. The API returns evaluation data (score, passed, reasoning, steps) directly, but the SDK was looking for a non-existent results array. `TraceEvaluationResultsResponse` now correctly maps to the API response shape and inherits from `TraceEvaluationResult`
- `TraceEvaluationStep` model now matches actual API fields (`tool`, `args`, `result`) instead of the incorrect (`step`, `reasoning`)

## [1.3.1] - 2026-03-13

### Added

- Automatic retry with exponential backoff for transient errors (HTTP 429, 500, 502, 503, 504) in both sync and async clients (up to 2 retries, respects `Retry-After` header, max 8s delay)
- Expanded documentation: updated README, examples for models/benchmarks, public API, and retrieving results

## [1.3.0] - 2026-03-13

### Changed

- Expanded model and benchmark result models with additional fields

### Fixed

- CI/CD publish workflows

## [1.2.0] - 2026-03-13

### Added

- `Stratix` / `AsyncStratix` clients (rebrand from Atlas)
- Judges resource with full CRUD
- Trace upload (JSON/JSONL up to 50 MB via presigned S3) and `trace_evaluations` resource
- Judge optimizations resource for tuning judge configurations
- `PublicClient` — a dedicated client for public endpoints (models, benchmarks, evaluations, comparisons), also accessible via `client.public`
- `get_by_key`, `add`, `remove`, `create_custom`, `create_smart` methods on Model & Benchmark resources
- `comparisons` resource for comparing evaluation results
- Apache 2.0 license

### Changed

- Expanded benchmark and model models with additional fields

### Deprecated

- `Atlas` client name — use `Stratix` instead (legacy `Atlas` aliases kept for backward compatibility)

### Fixed

- Evaluation status enum values

## [1.0.2] - 2026-03-13

### Changed

- Updated publish-to-AWS packaging job

## [1.0.1] - 2026-03-13

### Fixed

- Version bump

## [1.0.0] - 2026-03-13

### Added

- Initial release of the LayerLens evaluation SDK
- Sync and async clients for the LayerLens evaluation API
- `evaluations`, `results`, `models`, and `benchmarks` resources
- Typed exception hierarchy for API errors

[Unreleased]: https://github.com/LayerLens/stratix-python/compare/v1.9.0...HEAD
[1.9.0]: https://github.com/LayerLens/stratix-python/compare/v1.8.0...v1.9.0
[1.8.0]: https://github.com/LayerLens/stratix-python/compare/v1.7.0...v1.8.0
[1.7.0]: https://github.com/LayerLens/stratix-python/compare/v1.6.1...v1.7.0
[1.6.1]: https://github.com/LayerLens/stratix-python/compare/v1.6.0...v1.6.1
[1.6.0]: https://github.com/LayerLens/stratix-python/compare/v1.5.0...v1.6.0
[1.5.0]: https://github.com/LayerLens/stratix-python/compare/v1.4.0...v1.5.0
[1.4.0]: https://github.com/LayerLens/stratix-python/compare/v1.3.3...v1.4.0
[1.3.3]: https://github.com/LayerLens/stratix-python/compare/v1.3.2...v1.3.3
[1.3.2]: https://github.com/LayerLens/stratix-python/compare/v1.3.1...v1.3.2
[1.3.1]: https://github.com/LayerLens/stratix-python/compare/v1.3.0...v1.3.1
[1.3.0]: https://github.com/LayerLens/stratix-python/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/LayerLens/stratix-python/compare/v1.0.2...v1.2.0
[1.0.2]: https://github.com/LayerLens/stratix-python/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/LayerLens/stratix-python/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/LayerLens/stratix-python/releases/tag/v1.0.0
