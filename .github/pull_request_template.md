<!-- Thanks for the PR. Fill in the summary, then the invariant checklist if you touched the instrumentation/adapters. -->

## Summary

<!-- What & why. Link the Linear issue (LAY-####). -->

## Telemetry / privacy invariants

> Skip this section only if the PR does **not** touch `src/layerlens/instrument/**` (adapters, an `emit()` call-site, an event type, or the capture/redaction machinery). These are enforced by the **Invariant Gates** CI check (`rye run pytest -m invariant`) — but confirm them here so review catches intent, not just mechanics. See `CONTRIBUTING.md` → *Invariants & guards*.

- [ ] **New event types** are registered in `tests/instrument/_event_schema.py` (`KNOWN_EVENT_TYPES`).
- [ ] **Every content field** I emit (prompt / message / input / output / arguments / result / error / code / state / query / context / …) is in `_CONTENT_KEYS` for its event type, so `capture_content=False` strips it. *(The keys-must-match + reverse guard enforce this; the L7 sweep proves no SENTINEL survives.)*
- [ ] **New content-bearing event types are layer-mapped** in `_EVENT_TYPE_MAP` (so disabling a layer / `minimal()` suppresses them) — not left fail-open. Content-free types are explicitly allowlisted in the guard.
- [ ] **`agent.error` (and error events) carry a surviving category** (`error_type` / `error_code` / `status`) — not just a free-text `error` (which the backstop strips, leaving a failure indistinguishable from success).
- [ ] **`cost.record` for a priced model carries `cost_usd`** — emitted via `_emit` / `_fire` / `_price_cost_record`, not a raw `collector.emit`.
- [ ] **No secret-shaped value can reach an uploaded event** — error strings go through `safe_error` / the collector secret-scrub; new secret shapes added to `_secret_scrub.SECRET_PATTERNS`.
- [ ] **Every new test is adversarially bite-proven**: I reverted the fix and confirmed the test FAILS (a test that passes with the fix reverted guards nothing).

## Verification

- [ ] `rye run pytest -m invariant` is green locally.
- [ ] Affected per-adapter venvs are green (`/Users/.../.audit-venvs/<adapter>`), or N/A.
- [ ] Full suite (`rye run pytest`) green, or CI will run it.

---
🤖 Authoring assisted by [Claude Code](https://claude.com/claude-code)
