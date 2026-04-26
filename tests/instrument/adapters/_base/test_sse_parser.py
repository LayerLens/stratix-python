"""Unit tests for the shared Server-Sent Events parser.

Covers:

* W3C SSE spec field handling — ``data``, ``event``, ``id``, ``retry``,
  comments, unknown fields, leading-space stripping.
* Multi-line ``data:`` concatenation per the spec.
* Mixed line terminators (``\\n``, ``\\r``, ``\\r\\n``).
* Partial events at chunk boundaries.
* UTF-8 multi-byte codepoints split across chunks.
* OpenAI / Agentforce ``[DONE]`` sentinel surfaced via ``done`` flag.
* Async :func:`parse_stream` over an async byte iterator.
* Single-block convenience :func:`parse_event`.
* Malformed input tolerance (no exceptions raised).
* Buffer overflow protection.
"""

from __future__ import annotations

import asyncio
from typing import List, AsyncIterator

import pytest

from layerlens.instrument.adapters._base import (
    SSEEvent,
    SSEParser,
    parse_event,
    parse_stream,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _drain(parser: SSEParser, *chunks: bytes) -> List[SSEEvent]:
    """Feed each chunk in order and return all dispatched events."""
    events: List[SSEEvent] = []
    for chunk in chunks:
        events.extend(parser.feed(chunk))
    events.extend(parser.flush())
    return events


async def _async_iter(chunks: List[bytes]) -> AsyncIterator[bytes]:
    """Wrap a list of byte chunks as an async iterator."""
    for chunk in chunks:
        yield chunk


def _collect_async(events_coro: AsyncIterator[SSEEvent]) -> List[SSEEvent]:
    """Synchronously drain an async iterator of events."""

    async def runner() -> List[SSEEvent]:
        out: List[SSEEvent] = []
        async for event in events_coro:
            out.append(event)
        return out

    return asyncio.run(runner())


# ---------------------------------------------------------------------------
# 1. Basic field parsing (W3C spec compliance)
# ---------------------------------------------------------------------------


def test_single_data_line_dispatches_on_blank_line() -> None:
    """A bare ``data:`` line followed by a blank line dispatches one event."""
    parser = SSEParser()
    events = parser.feed(b"data: hello\n\n")
    assert len(events) == 1
    assert events[0].data == "hello"
    assert events[0].event is None
    assert events[0].id is None
    assert events[0].retry is None
    assert events[0].done is False


def test_all_fields_parsed() -> None:
    """``event:``, ``id:``, ``retry:``, and ``data:`` fields all populate."""
    payload = b"event: chat.delta\nid: 42\nretry: 5000\ndata: chunk-1\n\n"
    parser = SSEParser()
    events = parser.feed(payload)
    assert len(events) == 1
    e = events[0]
    assert e.event == "chat.delta"
    assert e.id == "42"
    assert e.retry == 5000
    assert e.data == "chunk-1"


def test_multiple_data_lines_joined_with_newline() -> None:
    """Multiple ``data:`` lines within one block are joined by ``\\n``."""
    payload = b"data: line one\ndata: line two\ndata: line three\n\n"
    events = SSEParser().feed(payload)
    assert len(events) == 1
    assert events[0].data == "line one\nline two\nline three"


def test_comment_lines_ignored() -> None:
    """Lines starting with ``:`` are comments and produce no fields."""
    payload = b": this is a heartbeat\ndata: real\n\n"
    events = SSEParser().feed(payload)
    assert len(events) == 1
    assert events[0].data == "real"


def test_one_leading_space_stripped_only() -> None:
    """Per spec, exactly one leading space is stripped from values."""
    payload = b"data:  two-leading-spaces\n\n"
    events = SSEParser().feed(payload)
    assert len(events) == 1
    # First space stripped, second preserved.
    assert events[0].data == " two-leading-spaces"


def test_no_space_after_colon_is_valid() -> None:
    """``data:value`` (no space) is a valid form per spec."""
    payload = b"data:no-space\n\n"
    events = SSEParser().feed(payload)
    assert events[0].data == "no-space"


def test_field_with_no_colon_treated_as_empty_value() -> None:
    """Per spec, a non-empty line with no colon is a field with empty value."""
    payload = b"data\n\n"
    events = SSEParser().feed(payload)
    # ``data`` field with empty value -> data lines = [""], joined -> ""
    assert len(events) == 1
    assert events[0].data == ""


def test_unknown_field_ignored() -> None:
    """Unknown field names are silently ignored per spec."""
    payload = b"data: ok\nfizz: ignored\nbuzz: ignored\n\n"
    events = SSEParser().feed(payload)
    assert len(events) == 1
    assert events[0].data == "ok"


def test_empty_data_field_dispatches_when_other_fields_present() -> None:
    """An event with an ``event:`` line and no data still dispatches."""
    payload = b"event: ping\n\n"
    events = SSEParser().feed(payload)
    assert len(events) == 1
    assert events[0].event == "ping"
    assert events[0].data == ""


def test_empty_event_resets_to_default() -> None:
    """``event: `` (empty value) clears the event name."""
    payload = b"event: foo\nevent:\ndata: x\n\n"
    events = SSEParser().feed(payload)
    assert len(events) == 1
    assert events[0].event is None
    assert events[0].data == "x"


def test_retry_with_non_digit_ignored() -> None:
    """``retry:`` with non-digit value is ignored per spec."""
    payload = b"retry: not-a-number\ndata: x\n\n"
    events = SSEParser().feed(payload)
    assert events[0].retry is None
    assert events[0].data == "x"


def test_id_with_null_byte_ignored() -> None:
    """``id:`` containing a NUL is ignored per spec."""
    payload = b"id: bad\x00id\ndata: x\n\n"
    events = SSEParser().feed(payload)
    assert events[0].id is None


# ---------------------------------------------------------------------------
# 2. Line terminator handling
# ---------------------------------------------------------------------------


def test_lf_line_terminators() -> None:
    payload = b"data: a\n\ndata: b\n\n"
    events = SSEParser().feed(payload)
    assert [e.data for e in events] == ["a", "b"]


def test_crlf_line_terminators() -> None:
    payload = b"data: a\r\n\r\ndata: b\r\n\r\n"
    events = SSEParser().feed(payload)
    assert [e.data for e in events] == ["a", "b"]


def test_cr_only_line_terminators() -> None:
    payload = b"data: a\r\rdata: b\r\r"
    events = SSEParser().feed(payload)
    assert [e.data for e in events] == ["a", "b"]


def test_mixed_terminators_in_one_stream() -> None:
    """A stream that mixes ``\\n`` and ``\\r\\n`` is parsed correctly."""
    payload = b"data: a\r\n\r\ndata: b\n\n"
    events = SSEParser().feed(payload)
    assert [e.data for e in events] == ["a", "b"]


# ---------------------------------------------------------------------------
# 3. Partial event / chunk boundary handling
# ---------------------------------------------------------------------------


def test_chunk_split_mid_line() -> None:
    """A chunk that ends mid-line preserves the partial line for next chunk."""
    parser = SSEParser()
    a = parser.feed(b"data: hel")
    assert a == []  # no complete line yet
    b = parser.feed(b"lo\n\n")
    assert len(b) == 1
    assert b[0].data == "hello"


def test_chunk_split_mid_field_name() -> None:
    """Chunk boundary inside the field name (before the colon)."""
    parser = SSEParser()
    assert parser.feed(b"da") == []
    assert parser.feed(b"ta: x\n\n") == [SSEEvent(data="x")]


def test_chunk_split_mid_event_block() -> None:
    """Chunk boundary between two field lines of one event."""
    parser = SSEParser()
    assert parser.feed(b"event: tick\n") == []
    out = parser.feed(b"data: 1\n\n")
    assert len(out) == 1
    assert out[0].event == "tick"
    assert out[0].data == "1"


def test_chunk_split_at_blank_line_separator() -> None:
    """Chunk boundary exactly at the blank line that dispatches."""
    parser = SSEParser()
    assert parser.feed(b"data: x\n") == []
    out = parser.feed(b"\n")
    assert len(out) == 1
    assert out[0].data == "x"


def test_many_tiny_chunks() -> None:
    """Byte-by-byte feeding produces the same result as one big feed."""
    payload = b"event: chunk\ndata: streamed\n\n"
    parser = SSEParser()
    events: List[SSEEvent] = []
    for byte in payload:
        events.extend(parser.feed(bytes([byte])))
    assert len(events) == 1
    assert events[0].event == "chunk"
    assert events[0].data == "streamed"


def test_flush_emits_unterminated_final_event() -> None:
    """A final event without a trailing blank line surfaces via flush()."""
    parser = SSEParser()
    assert parser.feed(b"data: tail") == []
    assert parser.flush() == [SSEEvent(data="tail")]


def test_flush_resets_parser_state() -> None:
    """After flush(), the parser may be re-used for a new stream."""
    parser = SSEParser()
    parser.feed(b"data: a\n\n")
    parser.flush()
    out = parser.feed(b"data: b\n\n")
    assert out == [SSEEvent(data="b")]


# ---------------------------------------------------------------------------
# 4. UTF-8 multi-byte boundary handling
# ---------------------------------------------------------------------------


def test_multibyte_codepoint_split_across_chunks() -> None:
    """A 3-byte codepoint split across chunks reassembles correctly."""
    # "★" (U+2605 BLACK STAR) is 3 bytes in UTF-8: e2 98 85
    full = "data: ★\n\n".encode("utf-8")
    # Split right inside the 3-byte sequence.
    midpoint = full.index(b"\xe2") + 2  # leaves the last byte for next chunk
    parser = SSEParser()
    first = parser.feed(full[:midpoint])
    second = parser.feed(full[midpoint:])
    assert first == []  # incomplete codepoint -> nothing dispatched
    assert len(second) == 1
    assert second[0].data == "★"


def test_four_byte_emoji_split_across_chunks() -> None:
    """A 4-byte UTF-8 codepoint (emoji) split across chunks reassembles."""
    # "🌍" (U+1F30D) is 4 bytes in UTF-8: f0 9f 8c 8d
    payload = "data: hi 🌍 there\n\n".encode("utf-8")
    emoji_start = payload.index(b"\xf0")
    parser = SSEParser()
    out: List[SSEEvent] = []
    # Split between every byte of the emoji to maximise stress.
    for i in range(emoji_start, emoji_start + 4):
        out.extend(parser.feed(payload[i : i + 1]))
    out.extend(parser.feed(payload[emoji_start + 4 :]))
    # Also feed the prefix in one shot.
    parser2 = SSEParser()
    out2 = parser2.feed(payload[:emoji_start])
    out2.extend(parser2.feed(payload[emoji_start : emoji_start + 1]))
    out2.extend(parser2.feed(payload[emoji_start + 1 :]))
    # The end-to-end reassembly must produce "hi 🌍 there" intact.
    assert any(e.data == "hi 🌍 there" for e in out2)


def test_invalid_utf8_does_not_raise() -> None:
    """Garbage bytes are replaced (errors='replace'), never raise."""
    parser = SSEParser()
    # Lone 0xFF is invalid UTF-8 anywhere.
    out = parser.feed(b"data: \xff\n\n")
    # Must not raise; data field contains the replacement char.
    assert len(out) == 1
    assert "�" in out[0].data


# ---------------------------------------------------------------------------
# 5. [DONE] sentinel
# ---------------------------------------------------------------------------


def test_done_sentinel_surfaced_on_event() -> None:
    """``data: [DONE]`` produces an event with ``done=True``."""
    parser = SSEParser()
    out = parser.feed(b"data: [DONE]\n\n")
    assert len(out) == 1
    assert out[0].done is True
    assert out[0].data == "[DONE]"


def test_done_sentinel_only_exact_match() -> None:
    """``data: [DONE] more`` is NOT the sentinel."""
    out = SSEParser().feed(b"data: [DONE] more\n\n")
    assert out[0].done is False


def test_done_sentinel_in_stream_does_not_stop_parser() -> None:
    """The parser continues to emit events after [DONE]; consumers decide."""
    parser = SSEParser()
    out = parser.feed(b"data: a\n\ndata: [DONE]\n\ndata: b\n\n")
    assert [e.data for e in out] == ["a", "[DONE]", "b"]
    assert [e.done for e in out] == [False, True, False]


# ---------------------------------------------------------------------------
# 6. Multiple events in one feed
# ---------------------------------------------------------------------------


def test_multiple_events_in_one_chunk() -> None:
    payload = b"data: one\n\ndata: two\n\ndata: three\n\n"
    out = SSEParser().feed(payload)
    assert [e.data for e in out] == ["one", "two", "three"]


def test_per_event_field_state_is_reset() -> None:
    """``event``/``id``/``retry`` from one block do not leak into the next."""
    payload = (
        b"event: first\nid: 1\nretry: 100\ndata: a\n\n"
        b"data: b\n\n"  # no event/id/retry here
    )
    out = SSEParser().feed(payload)
    assert out[0].event == "first"
    assert out[0].id == "1"
    assert out[0].retry == 100
    assert out[1].event is None
    assert out[1].id is None
    assert out[1].retry is None


# ---------------------------------------------------------------------------
# 7. Malformed input tolerance
# ---------------------------------------------------------------------------


def test_pure_blank_lines_dispatch_nothing() -> None:
    """A run of blank lines with no fields between produces no events."""
    out = SSEParser().feed(b"\n\n\n\n")
    assert out == []


def test_only_comments_dispatch_nothing() -> None:
    out = SSEParser().feed(b": one\n: two\n: three\n\n")
    assert out == []


def test_empty_feed_returns_empty() -> None:
    parser = SSEParser()
    assert parser.feed(b"") == []
    assert parser.flush() == []


def test_buffer_overflow_protection() -> None:
    """A line larger than MAX_LINE_BYTES is discarded with a warning."""
    parser = SSEParser()
    # Send 1.5 MiB of bytes with no newline.
    chunk = b"x" * (parser.MAX_LINE_BYTES + 100)
    out = parser.feed(chunk)
    assert out == []
    # After the discard, a fresh well-formed event still parses.
    out2 = parser.feed(b"\ndata: ok\n\n")
    assert any(e.data == "ok" for e in out2)


# ---------------------------------------------------------------------------
# 8. parse_event() convenience
# ---------------------------------------------------------------------------


def test_parse_event_single_block() -> None:
    e = parse_event("event: foo\ndata: bar")
    assert e is not None
    assert e.event == "foo"
    assert e.data == "bar"


def test_parse_event_empty_input_returns_none() -> None:
    assert parse_event("") is None


def test_parse_event_only_comment_returns_none() -> None:
    assert parse_event(": just a heartbeat") is None


def test_parse_event_with_trailing_blank() -> None:
    """Trailing blank line is fine — same single event."""
    e = parse_event("data: x\n\n")
    assert e is not None
    assert e.data == "x"


# ---------------------------------------------------------------------------
# 9. Async parse_stream()
# ---------------------------------------------------------------------------


def test_parse_stream_async_collects_events() -> None:
    chunks = [b"data: a\n\n", b"data: b\n\n", b"data: c\n\n"]
    out = _collect_async(parse_stream(_async_iter(chunks)))
    assert [e.data for e in out] == ["a", "b", "c"]


def test_parse_stream_async_partial_chunks() -> None:
    """Chunks split mid-line are reassembled by the async wrapper."""
    chunks = [b"data:", b" hel", b"lo\n", b"\n"]
    out = _collect_async(parse_stream(_async_iter(chunks)))
    assert len(out) == 1
    assert out[0].data == "hello"


def test_parse_stream_async_flushes_final_event() -> None:
    """A final event without trailing blank surfaces via parse_stream's flush."""
    chunks = [b"data: tail"]
    out = _collect_async(parse_stream(_async_iter(chunks)))
    assert len(out) == 1
    assert out[0].data == "tail"


# ---------------------------------------------------------------------------
# 10. Sync iterable adapter
# ---------------------------------------------------------------------------


def test_parse_stream_sync_collects_events() -> None:
    parser = SSEParser()
    chunks = [b"event: tick\n", b"data: 1\n\n", b"data: 2\n\n"]
    out = list(parser.parse_stream_sync(iter(chunks)))
    assert len(out) == 2
    assert out[0].event == "tick"
    assert out[0].data == "1"
    assert out[1].event is None
    assert out[1].data == "2"


# ---------------------------------------------------------------------------
# 11. Real-world OpenAI-style stream
# ---------------------------------------------------------------------------


def test_openai_style_chat_stream() -> None:
    """Realistic OpenAI chat.completions stream parses end-to-end."""
    payload = (
        b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'
        b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        b"data: [DONE]\n\n"
    )
    out = SSEParser().feed(payload)
    assert len(out) == 4
    assert out[0].data.startswith('{"choices"')
    assert out[3].done is True


# ---------------------------------------------------------------------------
# 12. Real-world Agentforce-style stream
# ---------------------------------------------------------------------------


def test_agentforce_style_text_stream() -> None:
    """Agentforce-style stream (data: + text JSON + [DONE]) parses cleanly."""
    payload = (
        b'data: {"text":"Welcome"}\n\n'
        b'data: {"text":" to"}\n\n'
        b'data: {"text":" Salesforce"}\n\n'
        b"data: [DONE]\n\n"
    )
    out = SSEParser().feed(payload)
    assert len(out) == 4
    assert out[3].done is True
    # Pre-DONE chunks all carry text payloads.
    assert all('"text"' in e.data for e in out[:3])


# ---------------------------------------------------------------------------
# 13. feed_text() path (text already decoded by transport)
# ---------------------------------------------------------------------------


def test_feed_text_round_trip() -> None:
    parser = SSEParser()
    out = parser.feed_text("data: from-text\n\n")
    assert out == [SSEEvent(data="from-text")]


def test_feed_text_partial_chunks() -> None:
    parser = SSEParser()
    assert parser.feed_text("data: ") == []
    assert parser.feed_text("hi\n\n") == [SSEEvent(data="hi")]


# ---------------------------------------------------------------------------
# 14. SSEEvent equality (sanity)
# ---------------------------------------------------------------------------


def test_sse_event_equality() -> None:
    a = SSEEvent(data="x", event="msg")
    b = SSEEvent(data="x", event="msg")
    assert a == b


def test_sse_event_default_values() -> None:
    e = SSEEvent()
    assert e.data == ""
    assert e.event is None
    assert e.id is None
    assert e.retry is None
    assert e.done is False


# ---------------------------------------------------------------------------
# 15. Stress: many chunks, one big stream
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk_size", [1, 3, 7, 16, 64, 256])
def test_chunked_feed_invariant_across_sizes(chunk_size: int) -> None:
    """Same logical stream parsed at any chunk size yields identical events."""
    payload = b"".join(f"data: msg-{i}\n\n".encode("ascii") for i in range(20))

    parser = SSEParser()
    events: List[SSEEvent] = []
    for offset in range(0, len(payload), chunk_size):
        events.extend(parser.feed(payload[offset : offset + chunk_size]))
    events.extend(parser.flush())

    assert len(events) == 20
    assert [e.data for e in events] == [f"msg-{i}" for i in range(20)]


def test_drain_helper_coverage() -> None:
    """Sanity: the in-test ``_drain`` helper used by readers matches feed+flush."""
    out = _drain(SSEParser(), b"data: a\n\ndata: b\n\n")
    assert [e.data for e in out] == ["a", "b"]
