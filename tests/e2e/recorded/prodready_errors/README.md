# Production-readiness error/edge corpus

Recorded fixtures captured 2026-06-26 by provoking REAL provider + atlas-app error
responses. These are **recorded fixtures for CI to replay** — CI must NOT make the
live calls. Each fixture holds the real HTTP status + body shape and, where
relevant, how the SDK's provider adapter surfaces the error.

Capture marker: `prodready-l5-*` (provider/redaction lane), `prodready-l3-*`
(atlas-app authz/error lane), `prodready-l7-*` (resilience/upload-edge lane).

## Provider errors (raw HTTP body shapes)
- `anthropic_badkey_401` — 401 `{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"},"request_id":...}`. The bad key is NOT echoed in the body.
- `anthropic_malformed_400` — 400 `invalid_request_error` "max_tokens: Field required".
- `anthropic_badmodel_404` — 404 `not_found_error` "model: <id>".
- `openai_badkey_401` — 401 `{"error":{"message":"Incorrect API key provided: sk-proj-***...nary",...,"code":"invalid_api_key"}}`. OpenAI MASKS the key in its own message (prefix + last 4 chars survive).
- `openai_malformed_400` — 400 "Missing required parameter: 'messages'".
- `ollama_badmodel_404` — 404 `{"error":"model '<id>' not found"}`.
- `ollama_badjson_400` — 400 `{"error":"invalid character ... looking for beginning of object key string"}`.

## SDK error-handling surface (the key check)
For each provider, the SDK client was driven under a live `TraceCollector` so the
adapter's `emit_llm_error` path ran. See `*_sdk_surface.json` + `full_config_error_scrub.json`:
- The provider exception becomes an `agent.error` event with `error_type` (exception class name) + `name` + `latency_ms`.
- Under the DEFAULT `capture_content=False`, the `error` text field is STRIPPED by `redact_payload` (`agent.error` content keys = `{error, error_message}`) — the message never reaches telemetry, only `error_type`.
- Under `full()` (`capture_content=True`), the `error` message survives BUT a full real-shaped key embedded in it is scrubbed to `[REDACTED-SECRET]` by `safe_error` + `scrub_payload`.
- In every probe the API key did NOT survive into the stored payload, and `find_secrets()` found no live secret patterns.

## atlas-app errors (real status + body)
- `atlas_badkey_401` — 401 `{"status":"error","error":"Unknown API key"}`.
- `atlas_malformed_upload_4xx` — 415 "invalid file type".
- `atlas_create_missing_file_4xx` — 400 (S3 GetObject NoSuchKey surfaced to the client).
- `atlas_oversize_presign` — 413 "file too large (max allowed: 50MB)".
- `atlas_oversize_client_guard` — client-side `ValueError` (50MB guard in `traces.upload`).
- `atlas_trace_notfound_4xx` — 404 "trace with ID '...' not found".
- `atlas_malformed_json_create` — 400 backend rejects malformed JSON trace file at the create step.
- `atlas_duplicate_trace_id` — same `trace_id` uploaded twice yields TWO distinct backend record ids (NOT idempotent / no de-dup).
- `atlas_crossorg_bola_200` — **BOLA F-L3-001**: org-A app key reads org-B integrations (200). Sensitive field VALUES redacted in the fixture; key names + count retained.
- `atlas_crossorg_traces_401_control` — control: the traces route DOES enforce org membership (401 "user is not the member of the organization").
- `atlas_sdk_badkey_surface` — `Stratix(api_key=bad)` raises `AuthenticationError` at construction; the bad key is NOT in `str(exc)`.

## Spend
~$0.00004 — one real `claude-haiku-4-5` call (`anthropic_real_haiku_success`) to
anchor the success path (real token usage + real `cost_usd`). All error probes
used bad keys / bad models / malformed bodies (= $0) or free local Ollama.
