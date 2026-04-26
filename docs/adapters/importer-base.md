# Importer Base — Shared Hardening Helpers

`layerlens.instrument.adapters._base.importer` provides a small, reusable
hardening surface for any adapter that **imports traces or records from
an upstream system** — Langfuse today, planned ServiceNow / Zendesk /
HubSpot tomorrow, plus anything else that fits the pattern of "list →
fetch → map → ingest".

The contract was extracted from the production-hardened Agentforce
importer (`frameworks/agentforce/auth.py` +
`frameworks/agentforce/importer.py`) so subsequent importers inherit
the same semantics by default rather than re-deriving them from scratch
(and missing edge cases).

This document is the contract for **future importer adapters**: when
you write the next one, build on the helpers below.

---

## What's in the box

| Symbol                       | Purpose                                                                  |
| ---------------------------- | ------------------------------------------------------------------------ |
| `validate_id` / `require_valid_id` | Regex-anchored ID validation BEFORE query interpolation             |
| `ID_PATTERN_*` constants     | Pre-compiled patterns for UUID, slug, integer, Salesforce, date, ts      |
| `parse_rate_limit_headers`   | Normalised view of `X-RateLimit-*` and `Retry-After`                     |
| `RateLimitInfo`              | Immutable dataclass — `.is_throttled`, `.sleep_seconds()`, `.usage_ratio`|
| `paginate` / `apaginate`     | Sync iterator + async generator over cursor-paged endpoints              |
| `batched_in`                 | Chunk a list of IDs for `WHERE x IN (…)` queries                          |
| `retry_with_backoff` / `aretry_with_backoff` | Decorrelated full-jitter backoff, rate-limit aware       |
| `RetryableHTTPError`         | Carries optional `RateLimitInfo` so the retry loop can sleep until reset |
| `BaseImporter`               | Abstract orchestrator: `fetch_records` → `process_record` with retry     |
| `ImporterResult`             | Standard counters: imported / skipped / failed / quarantined             |

---

## When to use it

Use the importer base when **every** statement below is true:

1. The adapter pulls data from an upstream HTTP API (or any cursored,
   rate-limited source — gRPC, GraphQL, message queue with ACK
   semantics).
2. The adapter does NOT instrument a running framework (those use
   `BaseAdapter` directly; importers compose `BaseImporter` *and*
   `BaseAdapter` — see `frameworks/langfuse/lifecycle.py`).
3. Records have stable, validatable IDs that are interpolated into
   subsequent fetch URLs / queries.

Don't use it for runtime-wrapping adapters (LangGraph callbacks,
LangChain handlers, Bedrock client decorators) — those have their own
contract via `_base/adapter.py` and `_base/capture.py`.

---

## Minimum viable importer

```python
from collections.abc import Iterator
from typing import Any

from layerlens.instrument.adapters._base.importer import (
    ID_PATTERN_UUID,
    BaseImporter,
    paginate,
    require_valid_id,
)


class MyImporter(BaseImporter[MyClient, MyState]):
    SOURCE = "myservice"

    def fetch_records(self) -> Iterator[dict[str, Any]]:
        # paginate handles cursor termination, empty-data short-circuit,
        # cursor-loop detection, and the max_pages safety bound.
        yield from paginate(
            lambda cursor: self._client.list_records(cursor=cursor),
            cursor_field="next_cursor",
            data_field="records",
        )

    def process_record(self, record: dict[str, Any]) -> bool:
        # Validate BEFORE any interpolation into a query string.
        rid = require_valid_id(record["id"], ID_PATTERN_UUID, label="record_id")
        full = self._client.get_record(rid)  # safe: rid passed validation
        events = self._mapper.map(full)
        for event in events:
            self._stratix.emit(event["type"], event["payload"])
        return True

    def record_id(self, record: dict[str, Any]) -> str:
        return record.get("id", "")
```

`run()` then drives the loop:

```python
result = MyImporter(client, state).run(dry_run=False)
print(f"imported={result.imported_count} failed={result.failed_count}")
```

---

## Hardening rules

These are the rules every importer adapter MUST follow. They map 1:1 to
the supra-spec features that the audit at
`A:/tmp/adapter-cross-pollination-audit.md` (§2.7-2.9) identified as
critical for importer adapters.

### 1. Always validate IDs before query interpolation

Any identifier that came from the upstream is untrusted until it has
been matched against an anchored regex.

```python
# WRONG — direct interpolation; vulnerable to injection.
url = f"/api/records/{record_id}"

# RIGHT — validate against the canonical pattern first.
if not validate_id(record_id, ID_PATTERN_UUID):
    logger.warning("Skipping malformed record id: %r", record_id)
    return None
url = f"/api/records/{record_id}"
```

The five built-in patterns cover the common cases:

| Constant                  | Matches                                                |
| ------------------------- | ------------------------------------------------------ |
| `ID_PATTERN_UUID`         | 8-4-4-4-12 hex (with or without hyphens, any case)     |
| `ID_PATTERN_SLUG`         | URL slugs — letters / digits / hyphens / underscores   |
| `ID_PATTERN_INTEGER`      | Decimal integers (signed, up to 19 digits)             |
| `ID_PATTERN_SALESFORCE`   | Salesforce 15-or-18-char alphanumeric record IDs       |
| `ID_PATTERN_DATE`         | `YYYY-MM-DD`                                           |
| `ID_PATTERN_TIMESTAMP`    | ISO 8601 datetime with optional offset / fractional s  |

If your upstream uses a different shape, compile and pin your own:

```python
import re
ID_PATTERN_MYAPP = re.compile(r"^MA-[A-Z0-9]{12}$")
```

### 2. Honour rate-limit headers

Don't retry blindly into a 429 — sleep until the explicit reset.

```python
from layerlens.instrument.adapters._base.importer import (
    RetryableHTTPError,
    parse_rate_limit_headers,
)

try:
    response = http.get(url)
except HTTPError as exc:
    rl = parse_rate_limit_headers(exc.response.headers)
    if exc.code == 429 or exc.code >= 500:
        # retry_with_backoff will sleep until rl.reset_at if rl is set,
        # else fall back to its computed exponential backoff.
        raise RetryableHTTPError("transient", status_code=exc.code, rate_limit=rl)
    raise  # 4xx other than 429 is terminal
```

Surface the most recent observation via a `last_rate_limit` property
so dashboards can warn at 80 % usage *before* the upstream throttles
(see `LangfuseAPIClient._warn_if_throttle_imminent`).

### 3. Use `paginate` / `apaginate` — don't roll your own

The shared helpers handle:

* Cursor termination on `null` / empty / missing.
* **Empty-data short-circuit** — providers occasionally return
  `next_cursor` on an empty final page; rolling your own loop will
  loop forever.
* **Cursor-loop guard** — if the provider buggily returns the same
  cursor twice, terminate rather than iterate forever.
* `max_pages` ceiling — defaults to 10 000, raise `RuntimeError` when
  exceeded so runaway pagination becomes a visible failure rather than
  an OOM.

```python
yield from paginate(
    lambda cursor: client.list(cursor=cursor),
    cursor_field="next_cursor",
    data_field="data",
    max_pages=10_000,
)
```

### 4. Use `retry_with_backoff` for transient failures

Real retry policy, not a sleep-and-pray loop:

* **Decorrelated full-jitter** sampling per the AWS Architecture Blog
  (2015) — best at avoiding synchronised retry storms when many
  importer instances fail at once.
* **Rate-limit-aware** — if the inner function raises
  `RetryableHTTPError(rate_limit=…)`, the loop sleeps until that
  explicit deadline (capped by `max_delay`) instead of using its own
  computed backoff.
* **Caps** — `max_delay` is enforced even when `Retry-After` is
  unreasonable; `max_retries` is enforced and re-raises the original
  exception type so callers see the same exception they always do.

```python
result = retry_with_backoff(
    lambda: client.fetch_one(record_id),
    max_retries=5,
    base_delay=1.0,
    max_delay=30.0,
    jitter=True,
)
```

### 5. Batch IDs through `WHERE x IN (…)` queries

Most upstream APIs cap the number of identifiers in a single `IN`
clause. Use `batched_in` rather than ad-hoc slicing:

```python
from layerlens.instrument.adapters._base.importer import batched_in

for batch in batched_in(parent_ids, batch_size=200):  # SOQL limit
    safe = [require_valid_id(p, ID_PATTERN_SALESFORCE) for p in batch]
    query = f"SELECT Id FROM Child WHERE ParentId IN ('{chr(39).join(safe)}')"
    yield from connection.query(query)
```

### 6. Quarantine after repeated failures

`BaseImporter.run()` will quarantine a record after
`quarantine_threshold` (default 3) consecutive failures, so a
poison-pill record doesn't keep failing forever and consuming retries.

If your state object exposes `is_quarantined(record_id) -> bool` and
`record_failure(record_id, max_failures: int) -> bool`, `BaseImporter`
will use them automatically. The Langfuse `SyncState` is the canonical
example; see `frameworks/langfuse/config.py`.

---

## Reference implementation

The Langfuse adapter is the first consumer of every helper in this
module:

* `frameworks/langfuse/client.py` — uses `retry_with_backoff` +
  `parse_rate_limit_headers` + `paginate` + `RetryableHTTPError`.
* `frameworks/langfuse/importer.py` — uses `validate_id` against
  `ID_PATTERN_UUID` + a second `retry_with_backoff` layer for
  per-trace fetches.
* `frameworks/langfuse/exporter.py` — uses `validate_id` +
  `retry_with_backoff` for batch ingestion pushes.

Use it as the template for the next importer.

---

## Acceptance checklist for new importers

Before merging an importer adapter PR, verify all of:

- [ ] Every upstream-supplied ID is validated via `validate_id` /
      `require_valid_id` before being interpolated into ANY query
      string.
- [ ] `parse_rate_limit_headers` is called on every successful AND
      failed response; results are surfaced via a public property.
- [ ] All cursor-paged endpoints use `paginate` / `apaginate` (no
      hand-rolled `while True` loops).
- [ ] All transient failures (429, 5xx, network errors) are wrapped in
      `RetryableHTTPError` and driven through `retry_with_backoff`.
- [ ] Terminal failures (4xx other than 429) raise the adapter's own
      `*APIError` and do **not** retry.
- [ ] `BaseImporter.run()` is the single entry point — no bespoke loops.
- [ ] Tests exercise: invalid-ID rejection, transient-retry recovery,
      terminal-4xx no-retry, rate-limit-aware sleep, quarantine
      promotion.
- [ ] `mypy --strict` clean on the new module.
- [ ] `ruff check` clean.
