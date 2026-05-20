# Changelog

All notable changes to the Stratix Python SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

**Latest version:** [1.7.0](https://github.com/LayerLens/stratix-python/releases/tag/v1.7.0) — 2026-05-20

## [Unreleased]

Things we're actively working on. Want to help? Check the [issues](https://github.com/LayerLens/stratix-python/issues) or [discussions](https://github.com/LayerLens/stratix-python/discussions).

### Added

### Changed

### Fixed

### Deprecated

### Removed

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

[Unreleased]: https://github.com/LayerLens/stratix-python/compare/v1.6.1...HEAD
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
