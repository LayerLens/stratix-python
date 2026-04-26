"""Shared Server-Sent Events (SSE) parser for LayerLens framework adapters.

Implements W3C SSE wire-format parsing in a single, reusable utility so
individual framework adapters do not have to re-implement ``data:`` line
splitting, multi-line ``data`` concatenation, ``[DONE]`` sentinel
handling, or chunk-boundary buffering.

Spec reference
~~~~~~~~~~~~~~

The format is defined by the WHATWG HTML "Server-sent events" section
(https://html.spec.whatwg.org/multipage/server-sent-events.html#parsing-an-event-stream).
This implementation follows the dispatch algorithm described there:

* The stream is decoded as UTF-8.
* Lines are separated by ``\\n``, ``\\r``, or ``\\r\\n``.
* A line beginning with ``:`` is a comment and is ignored.
* A line containing ``:`` is split into a ``field`` (left of the first
  colon) and a ``value`` (right of the first colon, with one leading
  space stripped if present).
* A line not containing ``:`` is treated as a field with an empty value.
* Recognised fields: ``data``, ``event``, ``id``, ``retry``.
* Multiple ``data:`` lines within the same event are concatenated with
  ``\\n`` separators (the trailing newline IS preserved between data
  fields, but the final concatenated data field has no trailing
  newline).
* An empty line dispatches the event using the accumulated buffer.
* If the dispatched event has no ``data`` and no ``event`` and no
  ``id`` field set during the block, it is dropped (the data buffer
  alone being empty after a single ``data:`` line is permitted because
  the spec dispatches whenever the data buffer is non-empty after a
  blank line).

Edge cases handled
~~~~~~~~~~~~~~~~~~

* **Partial events at chunk boundaries.** Bytes are buffered until a
  newline arrives. A chunk that ends mid-line does not lose data.
* **UTF-8 multi-byte boundary safety.** Bytes are accumulated as
  ``bytearray`` and decoded with an incremental decoder so that a
  multi-byte codepoint split across chunks is reassembled correctly.
* **Mixed newlines.** ``\\r``, ``\\n``, and ``\\r\\n`` are all
  recognised as line terminators; consecutive different terminators
  are not collapsed.
* **OpenAI-style ``[DONE]`` sentinel.** A convenience flag on each
  emitted event surfaces the sentinel for callers that want to stop
  consuming early.
* **Malformed input.** Missing colons, blank field names, and unknown
  field names are tolerated per spec (silently ignored).

This module deliberately has no third-party dependencies; everything
needed for the adapter contract is in the standard library.
"""

from __future__ import annotations

import codecs
import logging
from typing import List, Iterable, Iterator, Optional, AsyncIterator

from layerlens._compat.pydantic import Field, BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public model
# ---------------------------------------------------------------------------


class SSEEvent(BaseModel):
    """A single dispatched Server-Sent Event.

    Mirrors the four fields the SSE spec defines plus a derived
    ``done`` convenience flag for the OpenAI-compatible ``data: [DONE]``
    sentinel.

    Attributes:
        data: The concatenated ``data:`` field value with multiple
            lines joined by ``\\n``. May be the empty string.
        event: The ``event:`` field value, or ``None`` if the producer
            did not set one. The spec defaults this to ``"message"`` on
            the receiver side; we leave it as ``None`` here so callers
            can distinguish "not provided" from "explicitly message".
        id: The ``id:`` field value, or ``None`` if not set during this
            block. The SSE spec maintains a "last event ID" across
            dispatches; that is the consumer's responsibility, not the
            parser's.
        retry: The ``retry:`` field as an integer (milliseconds), or
            ``None`` if not set or unparseable.
        done: ``True`` if this event's ``data`` is exactly the literal
            string ``"[DONE]"`` (the OpenAI / Agentforce convention).
            The parser still emits the event; consumers may choose to
            stop iteration when they observe ``done=True``.
    """

    data: str = Field(default="", description="Concatenated data lines")
    event: Optional[str] = Field(default=None, description="Event name (event: field)")
    id: Optional[str] = Field(default=None, description="Last event ID (id: field)")
    retry: Optional[int] = Field(default=None, description="Reconnect time in ms (retry: field)")
    done: bool = Field(default=False, description="True if data is the [DONE] sentinel")


# ---------------------------------------------------------------------------
# Single-line parsing
# ---------------------------------------------------------------------------


def parse_event(line: str) -> Optional[SSEEvent]:
    """Parse a single complete SSE block (text containing one event).

    Convenience entry point for callers that already have a fully
    framed event payload — for example, a ``data: ...\\n\\n`` chunk
    extracted from a larger response, or a unit test wanting to verify
    field decoding on a fixed string. For incremental streaming, use
    :class:`SSEParser` or :func:`parse_stream`.

    Args:
        line: A complete SSE block. May contain multiple ``\\n``-
            separated field lines but represents ONE event. The block
            does not need a trailing blank line; this function
            dispatches the buffer at end-of-input.

    Returns:
        The parsed :class:`SSEEvent`, or ``None`` if the block was
        empty / contained only comments / produced no fields. Returning
        ``None`` matches the spec rule that empty buffers do not
        dispatch.
    """
    if not line:
        return None

    parser = SSEParser()
    events = parser.feed_text(line)
    # Force a final dispatch even if the caller did not include a
    # trailing blank line — the convenience contract is "this is one
    # event, give me the parsed result".
    events.extend(parser.flush())
    if not events:
        return None
    # parse_event returns at most one event by contract — if the input
    # somehow contained multiple, take the first non-empty one.
    return events[0]


# ---------------------------------------------------------------------------
# Incremental parser
# ---------------------------------------------------------------------------


class SSEParser:
    """Incremental Server-Sent Events parser.

    Stateful parser that consumes raw bytes (or already-decoded text)
    and emits :class:`SSEEvent` instances as complete events arrive.
    Designed for use from both synchronous iteration (e.g. the
    ``requests`` library's ``iter_content`` generator) and asynchronous
    iteration (e.g. ``httpx.AsyncClient.stream`` body).

    The parser maintains:

    * A byte buffer for any trailing partial line, so chunk boundaries
      that fall mid-line do not lose data.
    * An incremental UTF-8 decoder, so chunk boundaries that fall
      mid-codepoint do not produce ``UnicodeDecodeError``.
    * A per-event field accumulator (data buffer, event name, id,
      retry) that resets on every dispatch.

    The parser is **not thread-safe**; callers running multiple
    streams concurrently must use one parser per stream.

    Example:
        >>> parser = SSEParser()
        >>> events = parser.feed(b"data: hello\\n\\n")
        >>> events[0].data
        'hello'
    """

    # Maximum bytes a single line may occupy before we consider the
    # stream malformed and discard the buffer. Producers commonly cap
    # SSE messages at <64 KiB; this ceiling protects against runaway
    # accumulation from a stream that never emits a line terminator
    # (e.g. a misconfigured upstream sending raw HTML).
    MAX_LINE_BYTES = 1_048_576  # 1 MiB

    def __init__(self) -> None:
        self._byte_buffer: bytearray = bytearray()
        # ``incremental_decoder`` lets us call ``.decode(chunk)`` and
        # have any incomplete trailing multi-byte codepoint held back
        # internally until the next chunk arrives. Without this, a
        # 4-byte emoji split across two chunks would raise.
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

        # Per-event field buffers. Reset on every dispatch.
        self._data_lines: List[str] = []
        self._event_name: Optional[str] = None
        self._event_id: Optional[str] = None
        self._retry: Optional[int] = None
        self._has_field: bool = False

        # Carry-over text awaiting a line terminator.
        self._line_remainder: str = ""

    # -- Public feed API ----------------------------------------------------

    def feed(self, chunk: bytes) -> List[SSEEvent]:
        """Feed a raw byte chunk and return any newly-completed events.

        Args:
            chunk: Bytes from the underlying transport. May be empty.

        Returns:
            A list of zero or more :class:`SSEEvent` instances that
            became complete as a result of this chunk. A chunk that
            ends mid-event simply buffers state and returns ``[]``.
        """
        if not chunk:
            return []

        # Hard cap on buffered bytes to avoid OOM from a hostile or
        # broken upstream. We measure the in-flight buffer plus the
        # incoming chunk; if the total exceeds the cap, we drop the
        # buffered bytes (NOT the new chunk) and continue. Logging the
        # event lets operators detect the condition.
        if len(self._byte_buffer) + len(chunk) > self.MAX_LINE_BYTES:
            logger.warning(
                "SSEParser buffer exceeded %d bytes without a line terminator; "
                "discarding %d buffered bytes",
                self.MAX_LINE_BYTES,
                len(self._byte_buffer),
            )
            self._byte_buffer.clear()
            self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

        # Decode the incoming chunk. The incremental decoder retains
        # any trailing partial codepoint internally.
        text = self._decoder.decode(chunk)
        return self.feed_text(text)

    def feed_text(self, text: str) -> List[SSEEvent]:
        """Feed already-decoded text and return any newly-completed events.

        Useful for callers whose transport already yields ``str``
        chunks (e.g. ``requests.iter_lines(decode_unicode=True)``).
        Multi-byte boundary safety is the caller's problem in that
        case; this method only handles line buffering.

        Args:
            text: Decoded text from the underlying transport.

        Returns:
            A list of zero or more :class:`SSEEvent` instances that
            became complete as a result of this text.
        """
        if not text:
            return []

        # Combine with any previously-buffered partial line.
        buffer = self._line_remainder + text
        self._line_remainder = ""

        events: List[SSEEvent] = []

        # Walk the buffer character-by-character to honour the spec's
        # treatment of mixed CR / LF / CRLF terminators. ``str.splitlines``
        # also handles these but it does not let us distinguish "buffer
        # ends mid-line" (we keep the remainder) from "buffer ends with
        # a terminator" (no remainder). Manual scanning is clearer.
        i = 0
        line_start = 0
        n = len(buffer)
        while i < n:
            ch = buffer[i]
            if ch == "\r":
                line = buffer[line_start:i]
                event = self._consume_line(line)
                if event is not None:
                    events.append(event)
                # Skip the LF in a CRLF pair.
                if i + 1 < n and buffer[i + 1] == "\n":
                    i += 2
                else:
                    i += 1
                line_start = i
            elif ch == "\n":
                line = buffer[line_start:i]
                event = self._consume_line(line)
                if event is not None:
                    events.append(event)
                i += 1
                line_start = i
            else:
                i += 1

        # Anything not yet terminated is held for the next chunk.
        if line_start < n:
            self._line_remainder = buffer[line_start:]

        return events

    def flush(self) -> List[SSEEvent]:
        """Force-dispatch any buffered partial event.

        Call this at end-of-stream to surface a final event whose
        producer did not append a trailing blank line. The spec is
        ambiguous on this case, but real-world producers (notably
        OpenAI-compatible servers) sometimes close the connection
        immediately after the last ``data:`` line. Without flushing,
        that final event would be lost.

        After ``flush()``, the parser is reset to its initial state
        and may be re-used.

        Returns:
            A list containing zero or one :class:`SSEEvent`.
        """
        events: List[SSEEvent] = []

        # Drain the incremental decoder (no more bytes coming).
        try:
            tail = self._decoder.decode(b"", final=True)
        except UnicodeDecodeError:
            tail = ""
        if tail:
            self._line_remainder += tail

        # Treat any remaining buffered text as a final un-terminated line.
        if self._line_remainder:
            event = self._consume_line(self._line_remainder)
            self._line_remainder = ""
            if event is not None:
                events.append(event)

        # Treat the buffered fields (if any) as one final event even
        # though no blank line arrived.
        final = self._dispatch()
        if final is not None:
            events.append(final)

        # Reset for reuse.
        self._byte_buffer.clear()
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        return events

    # -- Iterable adapters --------------------------------------------------

    def parse_stream_sync(self, byte_iter: Iterable[bytes]) -> Iterator[SSEEvent]:
        """Synchronous helper: turn a byte iterable into an event iterator.

        Consumes ``byte_iter`` lazily and yields each completed event.
        At end of iteration, flushes any final partial event.

        Args:
            byte_iter: Any synchronous iterable producing ``bytes`` —
                e.g. ``response.iter_content(chunk_size=...)``.

        Yields:
            :class:`SSEEvent` instances in stream order.
        """
        for chunk in byte_iter:
            for event in self.feed(chunk):
                yield event
        for event in self.flush():
            yield event

    # -- Internal -----------------------------------------------------------

    def _consume_line(self, line: str) -> Optional[SSEEvent]:
        """Process a single physical line.

        Returns a dispatched event if the line was the empty line that
        terminates an event block; otherwise updates the field buffers
        and returns ``None``.
        """
        # Empty line dispatches the event.
        if not line:
            return self._dispatch()

        # Comment line.
        if line.startswith(":"):
            return None

        # Field line: split on the FIRST colon only.
        if ":" in line:
            field, _, value = line.partition(":")
            # Strip exactly one leading space per spec — not all
            # whitespace, just one space.
            if value.startswith(" "):
                value = value[1:]
        else:
            # Per spec, a line with no colon is treated as a field
            # name with empty value.
            field = line
            value = ""

        if field == "data":
            self._data_lines.append(value)
            self._has_field = True
        elif field == "event":
            # ``event: ""`` resets the event name to default per spec.
            self._event_name = value if value else None
            self._has_field = True
        elif field == "id":
            # An id field containing a NUL is ignored per spec.
            if "\x00" not in value:
                self._event_id = value
                self._has_field = True
        elif field == "retry":
            # Only accept ASCII digits per spec; otherwise ignore.
            if value.isdigit():
                self._retry = int(value)
                self._has_field = True
        else:
            # Unknown field — ignored per spec.
            pass

        return None

    def _dispatch(self) -> Optional[SSEEvent]:
        """Build an :class:`SSEEvent` from accumulated fields and reset state.

        Returns ``None`` when the accumulated buffer is empty (the spec
        rule: do not dispatch a "blank" event).
        """
        if not self._has_field and not self._data_lines:
            # Nothing to dispatch — purely whitespace / comments since
            # the last dispatch.
            return None

        # Per spec, multiple ``data:`` lines are joined with ``\n``.
        # An event with NO ``data`` line is still dispatchable if other
        # fields (event/id/retry) were set, so we cannot reset on
        # empty data alone.
        data = "\n".join(self._data_lines)

        event = SSEEvent(
            data=data,
            event=self._event_name,
            id=self._event_id,
            retry=self._retry,
            done=(data == "[DONE]"),
        )

        # Reset per-event state. ``id`` and ``retry`` persist across
        # events on the *consumer* side (the spec's "last event ID"
        # mechanism), but the *parser* must clear its per-block buffers
        # so subsequent events do not inherit stale fields. Consumers
        # that want last-event-id tracking should remember it from the
        # emitted event.
        self._data_lines = []
        self._event_name = None
        self._event_id = None
        self._retry = None
        self._has_field = False

        return event


# ---------------------------------------------------------------------------
# Async streaming helper
# ---------------------------------------------------------------------------


async def parse_stream(byte_iter: AsyncIterator[bytes]) -> AsyncIterator[SSEEvent]:
    """Parse an async byte stream into a stream of :class:`SSEEvent`.

    Wraps :class:`SSEParser` for use with ``httpx.AsyncClient.stream``,
    ``aiohttp.ClientResponse.content``, or any other async iterator of
    ``bytes``. Yields events in arrival order. Flushes any final
    partial event when the upstream iterator is exhausted.

    Args:
        byte_iter: An async iterator of ``bytes`` — typically the body
            of a streaming HTTP response.

    Yields:
        :class:`SSEEvent` instances in stream order.

    Example:
        >>> async with httpx.AsyncClient() as client:
        ...     async with client.stream("POST", url, json=payload) as resp:
        ...         async for event in parse_stream(resp.aiter_bytes()):
        ...             if event.done:
        ...                 break
        ...             handle(event)
    """
    parser = SSEParser()
    async for chunk in byte_iter:
        for event in parser.feed(chunk):
            yield event
    for event in parser.flush():
        yield event
