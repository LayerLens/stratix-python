"""Shared base for trace-importer adapters.

Provides the abstract :class:`BaseImporter` class plus reusable helpers
that any importer adapter (Langfuse, future ServiceNow, Zendesk,
HubSpot, etc.) can compose to deliver commercial-grade behaviour:

* **Regex ID validation** — :func:`validate_id` with ready-to-use
  patterns for UUID, slug, integer, and Salesforce-style IDs. Importers
  building queries from upstream-supplied IDs MUST validate before
  interpolation to block injection.
* **Rate-limit header parsing** — :func:`parse_rate_limit_headers`
  understands the de-facto standard (``X-RateLimit-Remaining``,
  ``X-RateLimit-Reset``, ``Retry-After``) and surfaces a structured
  :class:`RateLimitInfo` so callers can sleep until reset rather than
  retrying blindly into a 429 storm.
* **Cursor-based pagination** — :func:`paginate` (sync) and
  :func:`apaginate` (async generator) drive any "fetch-page → next
  cursor → repeat" upstream into a flat iterator of records, batching
  parent IDs through ``WHERE x IN (…)``-style clauses while honouring
  upstream batch limits.
* **Retry with exponential backoff + jitter** — :func:`retry_with_backoff`
  implements decorrelated full-jitter backoff (AWS Architecture Blog,
  *Exponential Backoff and Jitter*, 2015) and respects rate-limit
  headers when the failure is HTTP 429.

The pattern is lifted from the Agentforce importer
(``frameworks/agentforce/auth.py`` + ``frameworks/agentforce/importer.py``)
where it has been hardened against real Salesforce production loads.
This module factors the cross-cutting bits out so subsequent importer
adapters (next planned: Langfuse refactor in this PR; future:
ServiceNow, Zendesk, HubSpot) inherit the same semantics by default.

Threading model: helpers are thread-safe. :class:`BaseImporter`
subclasses MUST treat instance state as single-writer per ``run()``
call (no concurrent ``run()`` invocations on the same instance).
"""

from __future__ import annotations

import re
import abc
import time
import random
import asyncio
import logging
from typing import (
    Any,
    Dict,
    List,
    Generic,
    TypeVar,
    Callable,
    Iterator,
    Optional,
    Awaitable,
    AsyncIterator,
)
from datetime import datetime, timezone
from dataclasses import field, dataclass

UTC = timezone.utc  # Python 3.11+ has datetime.UTC; alias for 3.9/3.10 compat.

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ID validation patterns
# ---------------------------------------------------------------------------

#: Canonical 8-4-4-4-12 hex UUID (with or without hyphens, case-insensitive).
ID_PATTERN_UUID: re.Pattern[str] = re.compile(
    r"^[0-9a-fA-F]{8}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{12}$"
)

#: Lowercase URL slug — letters, digits, hyphens, underscores. Length 1-256.
ID_PATTERN_SLUG: re.Pattern[str] = re.compile(r"^[a-zA-Z0-9_\-]{1,256}$")

#: Decimal integer (positive or negative). Up to 19 digits (signed 64-bit max).
ID_PATTERN_INTEGER: re.Pattern[str] = re.compile(r"^-?\d{1,19}$")

#: Salesforce record ID — 15 case-sensitive or 18 case-insensitive alphanumeric.
ID_PATTERN_SALESFORCE: re.Pattern[str] = re.compile(r"^[a-zA-Z0-9]{15}(?:[a-zA-Z0-9]{3})?$")

#: ISO 8601 date (YYYY-MM-DD).
ID_PATTERN_DATE: re.Pattern[str] = re.compile(r"^\d{4}-\d{2}-\d{2}$")

#: ISO 8601 timestamp (date + 'T' + time, optional offset/zulu).
ID_PATTERN_TIMESTAMP: re.Pattern[str] = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?$"
)


def validate_id(value: Any, pattern: re.Pattern[str]) -> bool:
    """Return ``True`` iff ``value`` is a string fully matching ``pattern``.

    Use this BEFORE interpolating an upstream-supplied identifier into
    any query string (SOQL, SQL, GraphQL, JSON-Path, URL path segment).
    Validation against a compiled regex is the difference between an
    importer that rejects malformed input cleanly and one that issues
    an injection-vulnerable query upstream.

    Args:
        value: Candidate identifier. Non-strings always return ``False``
            (typed as :class:`Any` so the function is safe to invoke on
            untrusted upstream payloads where the field type is not
            statically guaranteed).
        pattern: Compiled regex; must contain ``^…$`` anchors. Use one of
            the ``ID_PATTERN_*`` constants in this module or supply a
            tighter project-specific pattern.

    Returns:
        ``True`` when ``value`` is a string and the pattern matches the
        entire string. ``False`` otherwise — never raises.
    """
    if not isinstance(value, str):
        return False
    return pattern.match(value) is not None


def require_valid_id(value: Any, pattern: re.Pattern[str], *, label: str = "id") -> str:
    """Validate and return ``value`` or raise :class:`ValueError`.

    Strict variant of :func:`validate_id` for code paths where an
    invalid id is a hard error (e.g. SOQL/SQL interpolation). The
    ``label`` argument is included in the error message so callers can
    locate the offending field at the boundary.

    Args:
        value: Candidate identifier.
        pattern: Compiled regex; see :func:`validate_id`.
        label: Field name used in the error message.

    Returns:
        The original ``value``, unchanged, when valid.

    Raises:
        ValueError: When ``value`` is not a string or the pattern does
            not match.
    """
    if not validate_id(value, pattern):
        raise ValueError(f"Invalid {label} format: {value!r}")
    # validate_id has confirmed isinstance(value, str); narrow for mypy.
    assert isinstance(value, str)  # noqa: S101 — tightening type after validate.
    return value


# ---------------------------------------------------------------------------
# Rate-limit header parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RateLimitInfo:
    """Parsed view of a response's rate-limit headers.

    Headers vary between providers, so :func:`parse_rate_limit_headers`
    normalises the common shapes:

    * ``X-RateLimit-Remaining`` — request budget left in the current window.
    * ``X-RateLimit-Limit`` — total budget per window (when advertised).
    * ``X-RateLimit-Reset`` — unix timestamp (seconds) when the budget
      replenishes. Some providers also send the value in seconds-from-now;
      we treat values lower than 10 years' worth of seconds as relative.
    * ``Retry-After`` — RFC 7231 §7.1.3. Either delta-seconds or an
      HTTP-date. Surfaced as :attr:`retry_after_s`.

    Attributes:
        remaining: Budget left in the current window, or ``None`` when
            the header was absent or unparseable.
        limit: Total budget per window, or ``None``.
        reset_at: Absolute timestamp (UTC) when the budget replenishes,
            or ``None``.
        retry_after_s: Seconds the caller SHOULD sleep before retrying
            (Retry-After header), or ``None``.
        usage_ratio: ``(limit - remaining) / limit`` when both are
            known, else ``None``. ``0.8`` means 80 % of the budget has
            been consumed.
    """

    remaining: Optional[int] = None
    limit: Optional[int] = None
    reset_at: Optional[datetime] = None
    retry_after_s: Optional[float] = None
    usage_ratio: Optional[float] = None

    @property
    def is_throttled(self) -> bool:
        """True when the upstream has explicitly asked us to back off.

        This is the case when either ``Retry-After`` was sent (response
        was 429 / 503) or ``remaining`` is zero.
        """
        if self.retry_after_s is not None and self.retry_after_s > 0:
            return True
        if self.remaining is not None and self.remaining <= 0:
            return True
        return False

    def sleep_seconds(self) -> float:
        """Recommended sleep duration before the next attempt.

        Prefers ``Retry-After`` when present; otherwise falls back to
        seconds-until-reset; otherwise zero.
        """
        if self.retry_after_s is not None and self.retry_after_s > 0:
            return float(self.retry_after_s)
        if self.reset_at is not None:
            now = datetime.now(UTC)
            delta = (self.reset_at - now).total_seconds()
            if delta > 0:
                return float(delta)
        return 0.0


# Ten years of seconds — anything smaller than this is treated as a
# relative seconds-from-now value rather than an absolute unix epoch.
# Picked deliberately to be large enough that 1970-style epoch values
# above year ~1980 are still parsed as absolute.
_RESET_RELATIVE_THRESHOLD_S = 10 * 365 * 24 * 3600


def _parse_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value.strip())
    except (ValueError, AttributeError):
        return None


def _parse_retry_after(value: Optional[str]) -> Optional[float]:
    """Parse RFC 7231 Retry-After (delta-seconds OR HTTP-date)."""
    if value is None:
        return None
    raw = value.strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        pass
    # Fallback: HTTP-date. Use email.utils which understands RFC 1123.
    from email.utils import parsedate_to_datetime

    try:
        dt = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    delta = (dt - datetime.now(UTC)).total_seconds()
    return max(0.0, delta)


def _parse_reset(value: Optional[str]) -> Optional[datetime]:
    """Parse a rate-limit reset header.

    Accepts an absolute unix timestamp (seconds since epoch) OR a
    seconds-from-now value. Heuristic: values below
    ``_RESET_RELATIVE_THRESHOLD_S`` are treated as relative.
    """
    parsed = _parse_int(value)
    if parsed is None:
        return None
    now = datetime.now(UTC)
    if parsed < _RESET_RELATIVE_THRESHOLD_S:
        return now.fromtimestamp(now.timestamp() + parsed, UTC)
    return datetime.fromtimestamp(float(parsed), UTC)


def _normalise_headers(headers: Any) -> Dict[str, str]:
    """Lower-case the header keys for case-insensitive lookup.

    Accepts dict-likes (including ``http.client.HTTPMessage``,
    ``requests.structures.CaseInsensitiveDict``, plain dicts).
    """
    out: Dict[str, str] = {}
    if headers is None:
        return out
    if hasattr(headers, "items"):
        items = headers.items()
    else:
        try:
            items = list(headers)
        except TypeError:
            return out
    for k, v in items:
        if k is None:
            continue
        out[str(k).lower()] = "" if v is None else str(v)
    return out


def parse_rate_limit_headers(headers: Any) -> RateLimitInfo:
    """Parse common rate-limit headers from an HTTP response.

    Looks for (case-insensitively):

    * ``X-RateLimit-Remaining`` / ``RateLimit-Remaining``
    * ``X-RateLimit-Limit`` / ``RateLimit-Limit``
    * ``X-RateLimit-Reset`` / ``RateLimit-Reset``
    * ``Retry-After``

    Args:
        headers: Mapping of header names to values. Accepts plain dicts,
            ``requests.Response.headers``, ``http.client.HTTPMessage``,
            and any object exposing ``items()``.

    Returns:
        A :class:`RateLimitInfo`. All fields default to ``None`` when
        the corresponding header is absent or unparseable — this never
        raises.
    """
    h = _normalise_headers(headers)

    remaining = _parse_int(h.get("x-ratelimit-remaining") or h.get("ratelimit-remaining"))
    limit = _parse_int(h.get("x-ratelimit-limit") or h.get("ratelimit-limit"))
    reset_at = _parse_reset(h.get("x-ratelimit-reset") or h.get("ratelimit-reset"))
    retry_after_s = _parse_retry_after(h.get("retry-after"))

    usage_ratio: Optional[float] = None
    if limit is not None and limit > 0 and remaining is not None:
        usage_ratio = max(0.0, min(1.0, (limit - remaining) / limit))

    return RateLimitInfo(
        remaining=remaining,
        limit=limit,
        reset_at=reset_at,
        retry_after_s=retry_after_s,
        usage_ratio=usage_ratio,
    )


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------

T = TypeVar("T")


def paginate(
    fetch_fn: Callable[[Optional[str]], Dict[str, Any]],
    cursor_field: str = "next_cursor",
    data_field: str = "data",
    max_pages: int = 10_000,
) -> Iterator[T]:
    """Sync cursor-based pagination iterator.

    Calls ``fetch_fn(cursor)`` repeatedly until the response no longer
    advances the cursor, yielding records from ``response[data_field]``
    one at a time so callers can stream large result sets without
    materialising every page in memory.

    The cursor is read from ``response[cursor_field]``. The loop
    terminates when:

    * ``response[cursor_field]`` is missing, ``None`` or an empty string
      (provider has no further pages), OR
    * ``response[data_field]`` is empty (defensive — some providers send
      ``next_cursor`` even when no records remain), OR
    * ``max_pages`` iterations have been performed (safety bound — set
      conservatively for unattended runs).

    Args:
        fetch_fn: Callable invoked once per page. First call receives
            ``None``; subsequent calls receive the cursor returned by
            the previous response. Must return a mapping with at least
            ``data_field`` and (optionally) ``cursor_field`` keys.
        cursor_field: Key on the response dict that holds the next-page
            cursor. Default ``"next_cursor"`` matches Langfuse;
            providers that use ``"nextRecordsUrl"`` (Salesforce) or
            ``"page_token"`` (Google) supply their own value.
        data_field: Key on the response dict that holds the list of
            records to yield. Default ``"data"``.
        max_pages: Hard cap on the number of pages fetched. Reaching
            this raises :class:`RuntimeError` rather than silently
            truncating — operators should be alerted when an importer
            chases a runaway cursor.

    Yields:
        Each record from ``response[data_field]`` across all pages, in
        the order returned by the upstream.

    Raises:
        RuntimeError: If ``max_pages`` is reached before the upstream
            stops paginating.
    """
    cursor: Optional[str] = None
    pages_seen = 0

    while pages_seen < max_pages:
        response = fetch_fn(cursor)
        pages_seen += 1

        records = response.get(data_field) or []
        for record in records:
            yield record

        next_cursor = response.get(cursor_field)
        # Empty data ALWAYS terminates — providers occasionally send
        # ``next_cursor`` on a final empty page.
        if not records:
            return
        if not next_cursor:
            return
        if next_cursor == cursor:
            # Defensive guard against cursor loops (provider bug).
            logger.warning(
                "paginate: cursor did not advance (%r) — terminating to avoid loop",
                cursor,
            )
            return
        cursor = str(next_cursor)

    raise RuntimeError(
        f"paginate: max_pages={max_pages} reached without exhausting cursor — "
        "raise the bound or investigate runaway upstream pagination"
    )


async def apaginate(
    fetch_fn: Callable[[Optional[str]], Awaitable[Dict[str, Any]]],
    cursor_field: str = "next_cursor",
    data_field: str = "data",
    max_pages: int = 10_000,
) -> AsyncIterator[T]:
    """Async generator counterpart to :func:`paginate`.

    Identical contract to the sync variant — see :func:`paginate` — but
    ``fetch_fn`` is awaited instead of called synchronously. Use this
    when the upstream client is built on ``httpx.AsyncClient`` /
    ``aiohttp``.

    Args:
        fetch_fn: Coroutine function invoked once per page. See
            :func:`paginate`.
        cursor_field: See :func:`paginate`.
        data_field: See :func:`paginate`.
        max_pages: See :func:`paginate`.

    Yields:
        Each record from ``response[data_field]`` across all pages.

    Raises:
        RuntimeError: As :func:`paginate`.
    """
    cursor: Optional[str] = None
    pages_seen = 0

    while pages_seen < max_pages:
        response = await fetch_fn(cursor)
        pages_seen += 1

        records = response.get(data_field) or []
        for record in records:
            yield record

        next_cursor = response.get(cursor_field)
        if not records:
            return
        if not next_cursor:
            return
        if next_cursor == cursor:
            logger.warning(
                "apaginate: cursor did not advance (%r) — terminating to avoid loop",
                cursor,
            )
            return
        cursor = str(next_cursor)

    raise RuntimeError(
        f"apaginate: max_pages={max_pages} reached without exhausting cursor — "
        "raise the bound or investigate runaway upstream pagination"
    )


def batched_in(values: List[str], batch_size: int = 100) -> Iterator[List[str]]:
    """Yield ``values`` in fixed-size batches for ``WHERE … IN (…)`` use.

    Most upstream APIs cap the number of identifiers permitted in a
    single ``IN`` clause (Salesforce SOQL: 200; PostgreSQL: 32 767;
    Langfuse: per-endpoint). Use this to chunk parent IDs before
    issuing a related-records query.

    Args:
        values: Identifiers to chunk. Empty input yields nothing.
        batch_size: Maximum batch size. Must be positive; otherwise
            raises :class:`ValueError`.

    Yields:
        Sub-lists of ``values`` of length ``<= batch_size``. The final
        batch may be shorter than ``batch_size``.

    Raises:
        ValueError: When ``batch_size <= 0``.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    for i in range(0, len(values), batch_size):
        yield values[i : i + batch_size]


# ---------------------------------------------------------------------------
# Retry with backoff + jitter
# ---------------------------------------------------------------------------


class RetryableHTTPError(Exception):
    """Marker exception that carries optional rate-limit context.

    Importers should raise this from the inner ``fn`` passed to
    :func:`retry_with_backoff` when an HTTP error is transient (429,
    5xx). The ``rate_limit`` attribute, when present, lets the retry
    loop sleep until the explicit reset deadline rather than using its
    computed backoff — which prevents thundering-herd behaviour against
    a clearly-throttled upstream.

    Args:
        message: Human-readable description.
        status_code: Original HTTP status code, if known.
        rate_limit: :class:`RateLimitInfo` parsed from the response
            headers, if available.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        rate_limit: Optional[RateLimitInfo] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.rate_limit = rate_limit


# Default retryable exception types. Subclasses of these are also
# retried — :func:`retry_with_backoff` accepts a custom tuple via the
# ``retry_on`` argument.
_DEFAULT_RETRYABLE: tuple[type[BaseException], ...] = (
    RetryableHTTPError,
    ConnectionError,
    TimeoutError,
    OSError,
)


def _compute_backoff(
    attempt: int,
    base_delay: float,
    max_delay: float,
    jitter: bool,
) -> float:
    """Decorrelated full-jitter backoff (AWS architecture blog, 2015).

    ``attempt`` is zero-indexed: the first retry uses ``attempt=0``, so
    the unjittered delay is ``base_delay * 2**attempt``. When
    ``jitter=True`` we sample uniformly from ``[0, capped_delay]`` —
    the *full jitter* variant — which is provably the best at avoiding
    synchronised retry storms when many clients fail at once.

    The result is always clamped to ``[0, max_delay]``.
    """
    capped: float = min(max_delay, base_delay * (2**attempt))
    if capped < 0.0:
        capped = 0.0
    if jitter:
        return random.uniform(0.0, capped)
    return capped


R = TypeVar("R")


def retry_with_backoff(
    fn: Callable[[], R],
    *,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: bool = True,
    retry_on: Optional[tuple[type[BaseException], ...]] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> R:
    """Run ``fn`` with exponential-backoff retry and full jitter.

    Real retry policy — not a stub. Honors :class:`RateLimitInfo` when
    the inner function raises :class:`RetryableHTTPError` carrying one,
    sleeping until the explicit ``Retry-After`` / reset deadline rather
    than retrying blindly.

    Args:
        fn: Zero-arg callable. Re-invoked on each attempt.
        max_retries: Maximum retry attempts AFTER the initial call. So
            ``max_retries=5`` gives up to 6 total invocations. Must be
            ``>= 0``.
        base_delay: Backoff base in seconds. The unjittered delay for
            attempt ``i`` (zero-indexed) is ``base_delay * 2**i``.
        max_delay: Hard upper bound on any single sleep, in seconds.
        jitter: When ``True`` (default), apply full-jitter randomisation
            to each sleep. Set ``False`` only when reproducible timing
            is required by the call site (test fixtures excluded — they
            should override ``sleep`` instead).
        retry_on: Tuple of exception types to retry. Defaults to
            :class:`RetryableHTTPError` plus standard transient
            failures (connection / timeout / OS errors). Pass an
            explicit tuple to extend or restrict.
        sleep: Override for ``time.sleep``. Test fixtures pass a
            recording function; production code keeps the default.

    Returns:
        The value returned by ``fn`` on its first successful call.

    Raises:
        ValueError: When ``max_retries < 0``.
        BaseException: The exception raised by ``fn`` on the final
            attempt — the original type is preserved (no wrapping).
    """
    if max_retries < 0:
        raise ValueError(f"max_retries must be >= 0, got {max_retries}")

    retryable: tuple[type[BaseException], ...] = retry_on if retry_on is not None else _DEFAULT_RETRYABLE

    last_exc: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except retryable as exc:
            last_exc = exc
            if attempt >= max_retries:
                break

            # Honor explicit rate-limit info when present.
            rate_limit_sleep = 0.0
            if isinstance(exc, RetryableHTTPError) and exc.rate_limit is not None:
                rate_limit_sleep = exc.rate_limit.sleep_seconds()

            if rate_limit_sleep > 0:
                # Cap so a misconfigured Retry-After can't block forever.
                delay = min(rate_limit_sleep, max_delay)
                logger.debug(
                    "retry_with_backoff: rate-limit-aware sleep %.2fs (attempt %d/%d)",
                    delay,
                    attempt + 1,
                    max_retries,
                )
            else:
                delay = _compute_backoff(attempt, base_delay, max_delay, jitter)
                logger.debug(
                    "retry_with_backoff: backoff sleep %.2fs (attempt %d/%d)",
                    delay,
                    attempt + 1,
                    max_retries,
                )
            sleep(delay)

    # Exhausted: re-raise the last exception, preserving traceback.
    assert last_exc is not None  # noqa: S101 — invariant: loop must have set this.
    raise last_exc


async def aretry_with_backoff(
    fn: Callable[[], Awaitable[R]],
    *,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: bool = True,
    retry_on: Optional[tuple[type[BaseException], ...]] = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> R:
    """Async counterpart of :func:`retry_with_backoff`.

    Identical semantics but ``fn`` is awaited and ``sleep`` defaults to
    :func:`asyncio.sleep`. See :func:`retry_with_backoff` for argument
    descriptions and behaviour.
    """
    if max_retries < 0:
        raise ValueError(f"max_retries must be >= 0, got {max_retries}")

    retryable: tuple[type[BaseException], ...] = retry_on if retry_on is not None else _DEFAULT_RETRYABLE

    last_exc: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        try:
            return await fn()
        except retryable as exc:
            last_exc = exc
            if attempt >= max_retries:
                break

            rate_limit_sleep = 0.0
            if isinstance(exc, RetryableHTTPError) and exc.rate_limit is not None:
                rate_limit_sleep = exc.rate_limit.sleep_seconds()

            if rate_limit_sleep > 0:
                delay = min(rate_limit_sleep, max_delay)
            else:
                delay = _compute_backoff(attempt, base_delay, max_delay, jitter)
            await sleep(delay)

    assert last_exc is not None  # noqa: S101
    raise last_exc


# ---------------------------------------------------------------------------
# BaseImporter
# ---------------------------------------------------------------------------


@dataclass
class ImporterResult:
    """Standardised result object for batch importer runs.

    Mirrors the Agentforce ``ImportResult`` shape but is generic across
    record types so different importers can share a single result API.
    Subclasses MAY add fields; the core counters here are enough for
    log-line dashboards (``imported / skipped / failed / quarantined``).
    """

    imported_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0
    quarantined_count: int = 0
    errors: List[str] = field(default_factory=list)
    duration_ms: float = 0.0
    dry_run: bool = False

    @property
    def total(self) -> int:
        """Sum of all per-record counters."""
        return self.imported_count + self.skipped_count + self.failed_count + self.quarantined_count

    @property
    def ok(self) -> bool:
        """True when no errors were recorded."""
        return not self.errors and self.failed_count == 0


C = TypeVar("C")  # client type
S = TypeVar("S")  # state type


class BaseImporter(abc.ABC, Generic[C, S]):
    """Abstract base for trace-importer adapters.

    Concrete subclasses (Langfuse, future ServiceNow, Zendesk, HubSpot,
    …) implement :meth:`fetch_records` and :meth:`process_record`;
    :meth:`run` orchestrates the loop, applying validation, retry, and
    rate-limit handling consistently.

    Type parameters:
        C: Concrete upstream client type (e.g. ``LangfuseAPIClient``).
        S: Concrete state type (e.g. ``SyncState``).

    Subclass contract:

    1. Override :meth:`fetch_records` to yield records from the
       upstream — typically by calling :func:`paginate` against a
       client method.
    2. Override :meth:`process_record` to ingest one record into the
       LayerLens pipeline. Return ``True`` for success, ``False`` for
       a non-fatal skip; raise to register a failure.
    3. Override :meth:`record_id` to return the upstream-side ID for
       quarantine tracking.

    The base :meth:`run` method handles:

    * Per-record try/except so one bad record does not abort the run.
    * Rate-limit-aware retry of :meth:`fetch_records` and
      :meth:`process_record`.
    * Result accumulation into :class:`ImporterResult`.
    * Duration measurement.
    """

    #: Subclasses MUST override.
    SOURCE: str = ""

    def __init__(
        self,
        client: C,
        state: S,
        *,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
    ) -> None:
        self._client: C = client
        self._state: S = state
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay

    # --- Subclass hooks ---

    @abc.abstractmethod
    def fetch_records(self) -> Iterator[Dict[str, Any]]:
        """Yield records from the upstream, one at a time.

        Implementations typically delegate to :func:`paginate` against
        a client list endpoint. The iterator MAY raise
        :class:`RetryableHTTPError` mid-iteration; the orchestrator
        will catch and retry the entire ``fetch_records`` call.
        """

    @abc.abstractmethod
    def process_record(self, record: Dict[str, Any]) -> bool:
        """Ingest a single record. Return ``True`` on success.

        Implementations should:

        * Validate the record's ID using :func:`require_valid_id`.
        * Map to LayerLens canonical events.
        * Push to the configured sink / store.

        Returning ``False`` increments ``skipped_count`` (deliberate
        skip — e.g. record was already imported, dry-run mode).

        Raising :class:`RetryableHTTPError` triggers retry. Raising
        any other exception increments ``failed_count`` and quarantines
        the record after :attr:`quarantine_threshold` consecutive
        failures.
        """

    @abc.abstractmethod
    def record_id(self, record: Dict[str, Any]) -> str:
        """Return the upstream-side identifier for quarantine tracking."""

    #: Number of consecutive failures before a record is quarantined.
    quarantine_threshold: int = 3

    # Subclass MAY override these if its state object exposes a
    # different quarantine API. Default is a no-op (no quarantining).
    def is_quarantined(self, record_id: str) -> bool:
        """Return True when ``record_id`` is in the quarantine set."""
        state = self._state
        if hasattr(state, "is_quarantined"):
            return bool(state.is_quarantined(record_id))
        return False

    def record_failure(self, record_id: str) -> bool:
        """Track a failure; return True when the record is now quarantined."""
        state = self._state
        if hasattr(state, "record_failure"):
            return bool(state.record_failure(record_id, self.quarantine_threshold))
        return False

    # --- Orchestration ---

    def run(self, dry_run: bool = False) -> ImporterResult:
        """Drive the import loop and return a populated result.

        Args:
            dry_run: When ``True``, :meth:`process_record` is NOT
                called — the loop only counts what *would* have been
                imported. Useful for cost estimates before a real run.

        Returns:
            :class:`ImporterResult` with per-record counters populated.
            ``duration_ms`` is set from :func:`time.monotonic`.
        """
        result = ImporterResult(dry_run=dry_run)
        started = time.monotonic()

        try:
            records = self._fetch_with_retry()
        except Exception as exc:
            logger.exception("%s: fetch_records failed terminally", self.SOURCE)
            result.errors.append(f"fetch_records: {exc}")
            result.failed_count = 1
            result.duration_ms = (time.monotonic() - started) * 1000.0
            return result

        for record in records:
            rid = self._safe_record_id(record)

            if rid and self.is_quarantined(rid):
                result.quarantined_count += 1
                continue

            if dry_run:
                result.imported_count += 1
                continue

            try:
                ok = self._process_with_retry(record)
            except Exception as exc:
                quarantined = self.record_failure(rid) if rid else False
                logger.warning(
                    "%s: process_record failed for %r: %s", self.SOURCE, rid or "<no-id>", exc
                )
                result.failed_count += 1
                if quarantined:
                    result.quarantined_count += 1
                result.errors.append(f"{rid or '<no-id>'}: {exc}")
                continue

            if ok:
                result.imported_count += 1
            else:
                result.skipped_count += 1

        result.duration_ms = (time.monotonic() - started) * 1000.0
        return result

    # --- Internals ---

    def _fetch_with_retry(self) -> List[Dict[str, Any]]:
        """Materialise :meth:`fetch_records` under :func:`retry_with_backoff`.

        We materialise to a list because :func:`retry_with_backoff`
        operates on callables that return a value — supporting a
        partially-consumed generator on retry would silently
        double-yield records the first attempt already produced.
        """

        def _do() -> List[Dict[str, Any]]:
            return list(self.fetch_records())

        return retry_with_backoff(
            _do,
            max_retries=self._max_retries,
            base_delay=self._base_delay,
            max_delay=self._max_delay,
        )

    def _process_with_retry(self, record: Dict[str, Any]) -> bool:
        """Invoke :meth:`process_record` under :func:`retry_with_backoff`."""

        def _do() -> bool:
            return self.process_record(record)

        return retry_with_backoff(
            _do,
            max_retries=self._max_retries,
            base_delay=self._base_delay,
            max_delay=self._max_delay,
        )

    def _safe_record_id(self, record: Dict[str, Any]) -> str:
        """Wrap :meth:`record_id` so a missing/malformed id can't crash the run."""
        try:
            rid = self.record_id(record)
        except Exception:
            logger.debug("%s: record_id() raised on %r", self.SOURCE, record, exc_info=True)
            return ""
        return rid or ""


__all__ = [
    "ID_PATTERN_DATE",
    "ID_PATTERN_INTEGER",
    "ID_PATTERN_SALESFORCE",
    "ID_PATTERN_SLUG",
    "ID_PATTERN_TIMESTAMP",
    "ID_PATTERN_UUID",
    "BaseImporter",
    "ImporterResult",
    "RateLimitInfo",
    "RetryableHTTPError",
    "apaginate",
    "aretry_with_backoff",
    "batched_in",
    "paginate",
    "parse_rate_limit_headers",
    "require_valid_id",
    "retry_with_backoff",
    "validate_id",
]
