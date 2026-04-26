"""Re-readable wrapper for Bedrock ``invoke_model`` response bodies.

The native ``botocore`` ``StreamingBody`` is single-pass: once consumed,
subsequent ``read()`` calls return ``b""``. The adapter must read the
body to extract tokens / content for telemetry, but the caller's own
code is also entitled to read it. ``RereadableBody`` materialises the
bytes once and exposes the ``StreamingBody`` subset most callers use:
``read``, ``iter_chunks``, ``iter_lines``, ``close``, and
``content_length``.

Ported from ``ateam/stratix/sdk/python/adapters/llm_providers/bedrock_adapter.py``.
"""

from __future__ import annotations

from typing import Iterator, Optional


class RereadableBody:
    """Drop-in replacement for ``botocore.response.StreamingBody``.

    The full body is held in memory as ``bytes`` so the caller can read
    it after the adapter has already consumed it. ``read(None)`` resets
    the position to the start, matching the semantics callers expect
    after we re-wrap the body.
    """

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    def read(self, amt: Optional[int] = None) -> bytes:
        """Read up to ``amt`` bytes, or the full body if ``amt`` is ``None``.

        A full read also resets the cursor — this matches what callers
        of ``StreamingBody`` typically do (one full read per body) and
        ensures a second full read returns the same bytes rather than an
        empty string after the adapter pre-read for telemetry.
        """
        if amt is None:
            self._pos = 0
            return self._data
        result = self._data[self._pos : self._pos + amt]
        self._pos += amt
        return result

    def iter_chunks(self, chunk_size: int = 1024) -> Iterator[bytes]:
        """Yield successive ``chunk_size``-byte slices of the body."""
        for i in range(0, len(self._data), chunk_size):
            yield self._data[i : i + chunk_size]

    def iter_lines(self) -> Iterator[bytes]:
        """Yield non-empty newline-separated lines."""
        for line in self._data.split(b"\n"):
            if line:
                yield line

    def close(self) -> None:
        """No-op (kept for API compatibility with ``StreamingBody``)."""

    @property
    def content_length(self) -> int:
        """Length of the buffered body in bytes."""
        return len(self._data)


# Backward-compat private alias from the original ateam module.
_RereadableBody = RereadableBody
