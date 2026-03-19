"""Deterministic clock for reproducible trace generation.

Uses seed-based PRNG to generate timestamps that are:
- Monotonically increasing within a trace
- Deterministic given the same seed
- Realistic in distribution (based on OTel histogram boundaries)
"""

from __future__ import annotations

import random
import time


class DeterministicClock:
    """Seed-based clock for reproducible timestamp generation.

    When seed is None, uses real wall-clock time with randomized jitter.
    When seed is provided, generates deterministic timestamps from PRNG.
    """

    def __init__(self, seed: int | None = None, base_time_ns: int | None = None):
        self._seed = seed
        self._rng = random.Random(seed)
        self._base_time_ns = base_time_ns or (
            self._rng.randint(
                1_700_000_000_000_000_000,
                1_800_000_000_000_000_000,
            )
            if seed is not None
            else time.time_ns()
        )
        self._cursor_ns = self._base_time_ns

    @property
    def seed(self) -> int | None:
        return self._seed

    @property
    def base_time_ns(self) -> int:
        return self._base_time_ns

    @property
    def cursor_ns(self) -> int:
        return self._cursor_ns

    def now_ns(self) -> int:
        """Return current cursor position without advancing."""
        return self._cursor_ns

    def advance_ms(self, ms: float) -> int:
        """Advance cursor by exact milliseconds, return new position."""
        self._cursor_ns += int(ms * 1_000_000)
        return self._cursor_ns

    def advance_random_ms(self, min_ms: float, max_ms: float) -> int:
        """Advance cursor by random milliseconds in range, return new position."""
        ms = self._rng.uniform(min_ms, max_ms)
        return self.advance_ms(ms)

    def agent_span_duration_ms(self) -> float:
        """Generate realistic agent span duration (1-10s)."""
        return self._rng.uniform(1000.0, 10000.0)

    def llm_span_duration_ms(self) -> float:
        """Generate realistic LLM call duration (200ms-5s).

        Based on OPERATION_DURATION_BOUNDARIES from otel_metrics.py.
        """
        return self._rng.uniform(200.0, 5000.0)

    def tool_span_duration_ms(self) -> float:
        """Generate realistic tool call duration (50ms-2s)."""
        return self._rng.uniform(50.0, 2000.0)

    def eval_span_duration_ms(self) -> float:
        """Generate realistic evaluation span duration (100ms-1s)."""
        return self._rng.uniform(100.0, 1000.0)

    def ttft_ms(self, min_ms: float = 50.0, max_ms: float = 500.0) -> float:
        """Generate time-to-first-token.

        Based on STREAMING_BOUNDARIES from otel_metrics.py.
        """
        return self._rng.uniform(min_ms, max_ms)

    def tpot_ms(self, min_ms: float = 10.0, max_ms: float = 50.0) -> float:
        """Generate time-per-output-token."""
        return self._rng.uniform(min_ms, max_ms)

    def inter_span_gap_ms(self) -> float:
        """Generate gap between sibling spans (1-50ms)."""
        return self._rng.uniform(1.0, 50.0)

    def fork(self) -> DeterministicClock:
        """Create a child clock sharing the PRNG but starting at current cursor.

        Useful for generating sub-traces or conversation turns.
        """
        child_seed = self._rng.randint(0, 2**31)
        return DeterministicClock(seed=child_seed, base_time_ns=self._cursor_ns)

    def random(self) -> float:
        """Return random float in [0, 1) from the deterministic PRNG."""
        return self._rng.random()

    def randint(self, a: int, b: int) -> int:
        """Return random int in [a, b] from the deterministic PRNG."""
        return self._rng.randint(a, b)

    def choice(self, seq: list) -> object:
        """Return random element from sequence."""
        return self._rng.choice(seq)

    def uniform(self, a: float, b: float) -> float:
        """Return random float in [a, b]."""
        return self._rng.uniform(a, b)
