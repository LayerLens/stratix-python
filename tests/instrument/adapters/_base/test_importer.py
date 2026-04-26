"""Unit tests for the shared importer base.

Covers every public symbol exported from
``layerlens.instrument.adapters._base.importer``:

* :func:`validate_id` / :func:`require_valid_id` against each
  ``ID_PATTERN_*``.
* :func:`parse_rate_limit_headers` for the de-facto-standard headers
  plus edge cases (missing headers, malformed values, RFC 7231
  ``Retry-After`` HTTP-dates, absolute-vs-relative reset).
* :class:`RateLimitInfo` semantics — :attr:`is_throttled` and
  :meth:`sleep_seconds`.
* :func:`paginate` (cursor termination, empty-data short-circuit,
  cursor-loop guard, ``max_pages`` bound).
* :func:`apaginate` async generator parity.
* :func:`batched_in` chunking semantics.
* :func:`retry_with_backoff` — success first-try, success after
  retries, exhaustion, rate-limit-aware sleep override, jitter
  off vs on, ``max_retries`` validation.
* :class:`BaseImporter` end-to-end orchestration including quarantine,
  retry, dry-run, and per-record failure isolation.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Iterator, Optional
from datetime import datetime, timezone, timedelta

import pytest

from layerlens.instrument.adapters._base.importer import (
    ID_PATTERN_DATE,
    ID_PATTERN_SLUG,
    ID_PATTERN_UUID,
    ID_PATTERN_INTEGER,
    ID_PATTERN_TIMESTAMP,
    ID_PATTERN_SALESFORCE,
    BaseImporter,
    RateLimitInfo,
    ImporterResult,
    RetryableHTTPError,
    paginate,
    apaginate,
    batched_in,
    validate_id,
    require_valid_id,
    retry_with_backoff,
    parse_rate_limit_headers,
)

UTC = timezone.utc


# ---------------------------------------------------------------------------
# validate_id / require_valid_id
# ---------------------------------------------------------------------------


class TestValidateId:
    def test_uuid_canonical_form_accepts(self) -> None:
        assert validate_id("550e8400-e29b-41d4-a716-446655440000", ID_PATTERN_UUID)

    def test_uuid_no_hyphens_accepts(self) -> None:
        assert validate_id("550e8400e29b41d4a716446655440000", ID_PATTERN_UUID)

    def test_uuid_uppercase_accepts(self) -> None:
        assert validate_id("550E8400-E29B-41D4-A716-446655440000", ID_PATTERN_UUID)

    def test_uuid_rejects_non_hex(self) -> None:
        assert not validate_id("550e8400-e29b-41d4-a716-44665544000z", ID_PATTERN_UUID)

    def test_slug_accepts_typical(self) -> None:
        assert validate_id("user-123_alpha", ID_PATTERN_SLUG)

    def test_slug_rejects_space(self) -> None:
        assert not validate_id("user 123", ID_PATTERN_SLUG)

    def test_slug_rejects_quote(self) -> None:
        # Critical: SQL-injection-style payloads MUST be rejected.
        assert not validate_id("user'); DROP TABLE--", ID_PATTERN_SLUG)

    def test_integer_positive(self) -> None:
        assert validate_id("12345", ID_PATTERN_INTEGER)

    def test_integer_negative(self) -> None:
        assert validate_id("-42", ID_PATTERN_INTEGER)

    def test_integer_rejects_decimal(self) -> None:
        assert not validate_id("3.14", ID_PATTERN_INTEGER)

    def test_salesforce_15_char(self) -> None:
        assert validate_id("0011x00000abcde", ID_PATTERN_SALESFORCE)

    def test_salesforce_18_char(self) -> None:
        assert validate_id("0011x00000abcdeAAA", ID_PATTERN_SALESFORCE)

    def test_salesforce_rejects_short(self) -> None:
        assert not validate_id("0011x", ID_PATTERN_SALESFORCE)

    def test_date_iso_8601(self) -> None:
        assert validate_id("2026-04-25", ID_PATTERN_DATE)

    def test_date_rejects_us_format(self) -> None:
        assert not validate_id("04/25/2026", ID_PATTERN_DATE)

    def test_timestamp_with_z(self) -> None:
        assert validate_id("2026-04-25T13:45:00Z", ID_PATTERN_TIMESTAMP)

    def test_timestamp_with_offset(self) -> None:
        assert validate_id("2026-04-25T13:45:00+02:00", ID_PATTERN_TIMESTAMP)

    def test_timestamp_with_microseconds(self) -> None:
        assert validate_id("2026-04-25T13:45:00.123456Z", ID_PATTERN_TIMESTAMP)

    def test_non_string_returns_false(self) -> None:
        # Must not raise — defensive against upstream payload variation.
        assert not validate_id(None, ID_PATTERN_UUID)
        assert not validate_id(12345, ID_PATTERN_UUID)
        assert not validate_id(["uuid"], ID_PATTERN_UUID)
        assert not validate_id({"id": "uuid"}, ID_PATTERN_UUID)

    def test_require_valid_id_returns_input_when_valid(self) -> None:
        value = "550e8400-e29b-41d4-a716-446655440000"
        assert require_valid_id(value, ID_PATTERN_UUID) is value

    def test_require_valid_id_raises_with_label(self) -> None:
        with pytest.raises(ValueError, match="Invalid trace_id format"):
            require_valid_id("not-a-uuid", ID_PATTERN_UUID, label="trace_id")

    def test_require_valid_id_raises_on_non_string(self) -> None:
        with pytest.raises(ValueError):
            require_valid_id(None, ID_PATTERN_UUID)


# ---------------------------------------------------------------------------
# parse_rate_limit_headers / RateLimitInfo
# ---------------------------------------------------------------------------


class TestParseRateLimitHeaders:
    def test_empty_headers_yields_all_none(self) -> None:
        info = parse_rate_limit_headers({})
        assert info.remaining is None
        assert info.limit is None
        assert info.reset_at is None
        assert info.retry_after_s is None
        assert info.usage_ratio is None
        assert not info.is_throttled

    def test_none_headers_safe(self) -> None:
        info = parse_rate_limit_headers(None)
        assert info.remaining is None

    def test_x_ratelimit_remaining_and_limit(self) -> None:
        info = parse_rate_limit_headers(
            {"X-RateLimit-Remaining": "10", "X-RateLimit-Limit": "100"}
        )
        assert info.remaining == 10
        assert info.limit == 100
        # 90% used → ratio 0.9
        assert info.usage_ratio == pytest.approx(0.9, abs=1e-6)

    def test_unprefixed_ratelimit_headers(self) -> None:
        # IETF draft uses unprefixed names.
        info = parse_rate_limit_headers(
            {"RateLimit-Remaining": "5", "RateLimit-Limit": "20"}
        )
        assert info.remaining == 5
        assert info.limit == 20

    def test_case_insensitive_lookup(self) -> None:
        info = parse_rate_limit_headers(
            {"x-ratelimit-remaining": "1", "X-RATELIMIT-LIMIT": "2"}
        )
        assert info.remaining == 1
        assert info.limit == 2

    def test_retry_after_seconds(self) -> None:
        info = parse_rate_limit_headers({"Retry-After": "30"})
        assert info.retry_after_s == 30.0
        assert info.is_throttled
        assert info.sleep_seconds() == 30.0

    def test_retry_after_http_date(self) -> None:
        future = datetime.now(UTC) + timedelta(seconds=120)
        # RFC 1123 format
        http_date = future.strftime("%a, %d %b %Y %H:%M:%S GMT")
        info = parse_rate_limit_headers({"Retry-After": http_date})
        assert info.retry_after_s is not None
        # Allow ±5s for parse-vs-now slop.
        assert 110 <= info.retry_after_s <= 125

    def test_retry_after_malformed_returns_none(self) -> None:
        info = parse_rate_limit_headers({"Retry-After": "not a duration"})
        assert info.retry_after_s is None

    def test_reset_absolute_timestamp(self) -> None:
        # Far-future absolute epoch.
        future_epoch = int(datetime.now(UTC).timestamp() + 3600)
        info = parse_rate_limit_headers({"X-RateLimit-Reset": str(future_epoch)})
        assert info.reset_at is not None
        delta = (info.reset_at - datetime.now(UTC)).total_seconds()
        assert 3500 <= delta <= 3700

    def test_reset_relative_seconds(self) -> None:
        info = parse_rate_limit_headers({"X-RateLimit-Reset": "60"})
        # 60 < threshold → treated as relative.
        assert info.reset_at is not None
        delta = (info.reset_at - datetime.now(UTC)).total_seconds()
        assert 55 <= delta <= 65

    def test_remaining_zero_is_throttled(self) -> None:
        info = parse_rate_limit_headers({"X-RateLimit-Remaining": "0"})
        assert info.is_throttled

    def test_sleep_seconds_prefers_retry_after_over_reset(self) -> None:
        info = parse_rate_limit_headers(
            {"Retry-After": "5", "X-RateLimit-Reset": "300"}
        )
        # Retry-After (5) wins over computed reset delta (~300).
        assert info.sleep_seconds() == 5.0

    def test_sleep_seconds_falls_back_to_reset(self) -> None:
        info = parse_rate_limit_headers({"X-RateLimit-Reset": "30"})
        slp = info.sleep_seconds()
        assert 25 <= slp <= 35

    def test_sleep_seconds_zero_when_no_signal(self) -> None:
        assert RateLimitInfo().sleep_seconds() == 0.0

    def test_handles_request_response_style_headers(self) -> None:
        # requests.structures.CaseInsensitiveDict has .items()
        class CaseInsensitive(dict):  # type: ignore[type-arg]
            pass

        h = CaseInsensitive({"X-RateLimit-Remaining": "7"})
        info = parse_rate_limit_headers(h)
        assert info.remaining == 7


# ---------------------------------------------------------------------------
# paginate / apaginate / batched_in
# ---------------------------------------------------------------------------


class TestPaginate:
    def test_single_page(self) -> None:
        def fetch(cursor: Optional[str]) -> Dict[str, Any]:
            assert cursor is None
            return {"data": [1, 2, 3]}

        assert list(paginate(fetch)) == [1, 2, 3]

    def test_multi_page_with_cursor(self) -> None:
        pages = [
            {"data": [1, 2], "next_cursor": "p2"},
            {"data": [3, 4], "next_cursor": "p3"},
            {"data": [5], "next_cursor": None},
        ]
        idx = {"i": 0}
        cursors_seen: List[Optional[str]] = []

        def fetch(cursor: Optional[str]) -> Dict[str, Any]:
            cursors_seen.append(cursor)
            page = pages[idx["i"]]
            idx["i"] += 1
            return page

        assert list(paginate(fetch)) == [1, 2, 3, 4, 5]
        assert cursors_seen == [None, "p2", "p3"]

    def test_empty_data_terminates_even_with_cursor(self) -> None:
        # Defensive: provider sent next_cursor on an empty page.
        def fetch(cursor: Optional[str]) -> Dict[str, Any]:
            return {"data": [], "next_cursor": "p2"}

        assert list(paginate(fetch)) == []

    def test_cursor_loop_guard(self, caplog: pytest.LogCaptureFixture) -> None:
        # Provider buggily returns same cursor twice → loop, must terminate.
        def fetch(cursor: Optional[str]) -> Dict[str, Any]:
            return {"data": ["x"], "next_cursor": "stuck"}

        # First call: cursor=None, return cursor "stuck".
        # Second call: cursor="stuck", returned cursor matches → terminate.
        result = list(paginate(fetch))
        assert result == ["x", "x"]

    def test_max_pages_raises(self) -> None:
        def fetch(cursor: Optional[str]) -> Dict[str, Any]:
            next_cursor = str(int(cursor or "0") + 1)
            return {"data": [int(next_cursor)], "next_cursor": next_cursor}

        with pytest.raises(RuntimeError, match="max_pages"):
            list(paginate(fetch, max_pages=3))

    def test_custom_data_field(self) -> None:
        def fetch(cursor: Optional[str]) -> Dict[str, Any]:
            return {"records": ["a", "b"], "next_cursor": None}

        assert list(paginate(fetch, data_field="records")) == ["a", "b"]

    def test_apaginate_async(self) -> None:
        pages = [
            {"data": [1, 2], "next_cursor": "p2"},
            {"data": [3], "next_cursor": None},
        ]
        idx = {"i": 0}

        async def fetch(cursor: Optional[str]) -> Dict[str, Any]:
            page = pages[idx["i"]]
            idx["i"] += 1
            return page

        async def collect() -> List[Any]:
            out: List[Any] = []
            async for record in apaginate(fetch):
                out.append(record)
            return out

        assert asyncio.run(collect()) == [1, 2, 3]


class TestBatchedIn:
    def test_chunks_of_size(self) -> None:
        result = list(batched_in(["a", "b", "c", "d", "e"], batch_size=2))
        assert result == [["a", "b"], ["c", "d"], ["e"]]

    def test_empty_input(self) -> None:
        assert list(batched_in([])) == []

    def test_batch_size_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            list(batched_in(["a"], batch_size=0))


# ---------------------------------------------------------------------------
# retry_with_backoff
# ---------------------------------------------------------------------------


class TestRetryWithBackoff:
    def test_success_on_first_attempt(self) -> None:
        calls: List[int] = []

        def fn() -> str:
            calls.append(1)
            return "ok"

        sleeps: List[float] = []
        assert retry_with_backoff(fn, sleep=sleeps.append) == "ok"
        assert calls == [1]
        assert sleeps == []

    def test_success_after_retries(self) -> None:
        attempts = {"n": 0}

        def fn() -> str:
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise RetryableHTTPError("transient", status_code=500)
            return "done"

        sleeps: List[float] = []
        result = retry_with_backoff(
            fn, base_delay=0.001, max_delay=0.01, jitter=False, sleep=sleeps.append
        )
        assert result == "done"
        assert attempts["n"] == 3
        # Two sleeps before success.
        assert len(sleeps) == 2

    def test_exhaustion_re_raises_last_exception(self) -> None:
        def fn() -> None:
            raise RetryableHTTPError("always", status_code=503)

        sleeps: List[float] = []
        with pytest.raises(RetryableHTTPError, match="always"):
            retry_with_backoff(
                fn,
                max_retries=2,
                base_delay=0.001,
                max_delay=0.001,
                jitter=False,
                sleep=sleeps.append,
            )
        # 1 initial + 2 retries → 2 sleeps before giving up.
        assert len(sleeps) == 2

    def test_non_retryable_exception_passthrough(self) -> None:
        def fn() -> None:
            raise ValueError("hard error")

        sleeps: List[float] = []
        with pytest.raises(ValueError, match="hard error"):
            retry_with_backoff(fn, sleep=sleeps.append)
        # No retry attempted for non-retryable exception.
        assert sleeps == []

    def test_jitter_off_uses_predictable_backoff(self) -> None:
        def fn() -> None:
            raise ConnectionError("net")

        sleeps: List[float] = []
        with pytest.raises(ConnectionError):
            retry_with_backoff(
                fn,
                max_retries=3,
                base_delay=1.0,
                max_delay=10.0,
                jitter=False,
                sleep=sleeps.append,
            )
        # Without jitter: 1.0, 2.0, 4.0 — predictable.
        assert sleeps == [1.0, 2.0, 4.0]

    def test_jitter_on_caps_within_bound(self) -> None:
        def fn() -> None:
            raise ConnectionError("net")

        sleeps: List[float] = []
        with pytest.raises(ConnectionError):
            retry_with_backoff(
                fn,
                max_retries=4,
                base_delay=1.0,
                max_delay=8.0,
                jitter=True,
                sleep=sleeps.append,
            )
        # Each sleep is in [0, min(8.0, 1.0 * 2**i)].
        assert all(0.0 <= s <= 8.0 for s in sleeps)

    def test_rate_limit_aware_sleep_overrides_backoff(self) -> None:
        # First call: 429 with Retry-After=2.
        # Second call: success.
        attempts = {"n": 0}

        rl = parse_rate_limit_headers({"Retry-After": "2"})

        def fn() -> str:
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise RetryableHTTPError("throttled", status_code=429, rate_limit=rl)
            return "done"

        sleeps: List[float] = []
        retry_with_backoff(
            fn,
            base_delay=0.001,  # tiny — would be picked if rate-limit ignored
            max_delay=10.0,
            jitter=False,
            sleep=sleeps.append,
        )
        # The single sleep should be the rate-limit value (~2.0), NOT
        # the tiny backoff base — proves rate_limit took precedence.
        assert len(sleeps) == 1
        assert sleeps[0] == pytest.approx(2.0, abs=0.1)

    def test_rate_limit_sleep_is_capped_by_max_delay(self) -> None:
        rl = parse_rate_limit_headers({"Retry-After": "9999"})

        def fn() -> str:
            raise RetryableHTTPError("forever", status_code=429, rate_limit=rl)

        sleeps: List[float] = []
        with pytest.raises(RetryableHTTPError):
            retry_with_backoff(
                fn,
                max_retries=1,
                max_delay=5.0,
                jitter=False,
                sleep=sleeps.append,
            )
        assert sleeps == [5.0]

    def test_max_retries_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="max_retries"):
            retry_with_backoff(lambda: None, max_retries=-1)

    def test_custom_retry_on(self) -> None:
        class MyError(Exception):
            pass

        attempts = {"n": 0}

        def fn() -> str:
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise MyError("custom")
            return "ok"

        sleeps: List[float] = []
        result = retry_with_backoff(
            fn,
            retry_on=(MyError,),
            base_delay=0.001,
            max_delay=0.001,
            jitter=False,
            sleep=sleeps.append,
        )
        assert result == "ok"
        assert attempts["n"] == 2


# ---------------------------------------------------------------------------
# BaseImporter
# ---------------------------------------------------------------------------


class _FakeState:
    """Minimal state object exposing the BaseImporter quarantine API."""

    def __init__(self) -> None:
        self.failures: Dict[str, int] = {}
        self.threshold = 3

    def is_quarantined(self, record_id: str) -> bool:
        return self.failures.get(record_id, 0) >= self.threshold

    def record_failure(self, record_id: str, max_failures: int = 3) -> bool:
        self.failures[record_id] = self.failures.get(record_id, 0) + 1
        self.threshold = max_failures
        return self.failures[record_id] >= max_failures


class _FakeImporter(BaseImporter[None, _FakeState]):
    SOURCE = "fake"

    def __init__(
        self,
        state: _FakeState,
        records: List[Dict[str, Any]],
        *,
        fail_ids: Optional[set[str]] = None,
        fetch_fail_count: int = 0,
    ) -> None:
        super().__init__(client=None, state=state, max_retries=2, base_delay=0.001, max_delay=0.001)
        self._records = records
        self._fail_ids = fail_ids or set()
        self._fetch_fail_count = fetch_fail_count
        self._fetch_attempts = 0
        self.processed: List[str] = []

    def fetch_records(self) -> Iterator[Dict[str, Any]]:
        self._fetch_attempts += 1
        if self._fetch_attempts <= self._fetch_fail_count:
            raise RetryableHTTPError("transient list error", status_code=500)
        for r in self._records:
            yield r

    def process_record(self, record: Dict[str, Any]) -> bool:
        rid = record.get("id", "")
        if rid in self._fail_ids:
            raise RuntimeError(f"forced failure for {rid}")
        self.processed.append(rid)
        return True

    def record_id(self, record: Dict[str, Any]) -> str:
        return str(record.get("id", ""))


class TestBaseImporter:
    def test_run_imports_all_records(self) -> None:
        state = _FakeState()
        importer = _FakeImporter(
            state,
            [{"id": "r1"}, {"id": "r2"}, {"id": "r3"}],
        )
        result = importer.run()
        assert result.imported_count == 3
        assert result.failed_count == 0
        assert result.ok
        assert importer.processed == ["r1", "r2", "r3"]
        assert result.duration_ms >= 0.0

    def test_dry_run_does_not_process(self) -> None:
        state = _FakeState()
        importer = _FakeImporter(state, [{"id": "r1"}, {"id": "r2"}])
        result = importer.run(dry_run=True)
        assert result.imported_count == 2
        assert importer.processed == []
        assert result.dry_run

    def test_per_record_failure_isolated(self) -> None:
        state = _FakeState()
        importer = _FakeImporter(
            state,
            [{"id": "r1"}, {"id": "r2"}, {"id": "r3"}],
            fail_ids={"r2"},
        )
        result = importer.run()
        assert result.imported_count == 2
        assert result.failed_count == 1
        assert "r2" in result.errors[0]
        assert importer.processed == ["r1", "r3"]

    def test_quarantine_after_repeated_failures(self) -> None:
        state = _FakeState()
        # Force r1 to fail every time.
        importer = _FakeImporter(state, [{"id": "r1"}], fail_ids={"r1"})

        # Run 1: fails, failure_count=1.
        # The retry loop will attempt 3 times (max_retries=2 + 1 initial)
        # but each attempt raises a non-retryable RuntimeError, so retry
        # gives up after the first attempt.
        result1 = importer.run()
        assert result1.failed_count == 1
        assert state.failures["r1"] == 1

        # Run 2 + 3 push the failure count to 3 → quarantined.
        importer.run()
        result3 = importer.run()
        assert state.failures["r1"] >= 3
        assert result3.quarantined_count >= 1

        # Run 4: r1 is quarantined, never reaches process_record.
        importer.processed.clear()
        result4 = importer.run()
        assert result4.quarantined_count == 1
        assert importer.processed == []

    def test_fetch_retried_on_transient_error(self) -> None:
        state = _FakeState()
        importer = _FakeImporter(
            state,
            [{"id": "r1"}],
            fetch_fail_count=2,  # First two fetch attempts raise.
        )
        result = importer.run()
        # Retry should recover and import.
        assert result.imported_count == 1
        assert importer.processed == ["r1"]

    def test_fetch_terminal_failure_records_error(self) -> None:
        state = _FakeState()
        importer = _FakeImporter(
            state,
            [{"id": "r1"}],
            fetch_fail_count=99,  # More than max_retries can cover.
        )
        result = importer.run()
        assert result.imported_count == 0
        assert result.failed_count == 1
        assert any("fetch_records" in e for e in result.errors)

    def test_record_id_exception_safe(self) -> None:
        class BadRecordIdImporter(_FakeImporter):
            def record_id(self, record: Dict[str, Any]) -> str:
                raise RuntimeError("boom")

        state = _FakeState()
        importer = BadRecordIdImporter(state, [{"id": "r1"}])
        # Should not crash — record_id failure is silently logged.
        result = importer.run()
        assert result.imported_count == 1


class TestImporterResult:
    def test_total_sums_counters(self) -> None:
        r = ImporterResult(
            imported_count=2, skipped_count=3, failed_count=1, quarantined_count=4
        )
        assert r.total == 10

    def test_ok_when_no_errors_or_failures(self) -> None:
        assert ImporterResult().ok

    def test_not_ok_with_errors(self) -> None:
        r = ImporterResult(errors=["x"])
        assert not r.ok

    def test_not_ok_with_failures(self) -> None:
        r = ImporterResult(failed_count=1)
        assert not r.ok
