"""Tests for DeterministicClock."""

from layerlens.instrument.simulators.clock import DeterministicClock


class TestDeterministicClock:
    def test_seeded_determinism(self):
        clock1 = DeterministicClock(seed=42)
        clock2 = DeterministicClock(seed=42)
        assert clock1.base_time_ns == clock2.base_time_ns
        assert clock1.now_ns() == clock2.now_ns()

    def test_advance_ms(self):
        clock = DeterministicClock(seed=42)
        start = clock.now_ns()
        result = clock.advance_ms(100.0)
        assert result == start + 100_000_000
        assert clock.now_ns() == result

    def test_advance_random_ms(self):
        clock = DeterministicClock(seed=42)
        start = clock.now_ns()
        result = clock.advance_random_ms(50.0, 100.0)
        delta_ms = (result - start) / 1_000_000
        assert 50.0 <= delta_ms <= 100.0

    def test_advance_random_deterministic(self):
        clock1 = DeterministicClock(seed=99)
        clock2 = DeterministicClock(seed=99)
        r1 = clock1.advance_random_ms(10.0, 50.0)
        r2 = clock2.advance_random_ms(10.0, 50.0)
        assert r1 == r2

    def test_monotonic_advancing(self):
        clock = DeterministicClock(seed=42)
        values = []
        for _ in range(10):
            values.append(clock.now_ns())
            clock.advance_random_ms(1.0, 10.0)
        assert values == sorted(values)

    def test_unseeded_uses_wall_clock(self):
        clock = DeterministicClock(seed=None)
        assert clock.seed is None
        assert clock.base_time_ns > 0

    def test_agent_span_duration(self):
        clock = DeterministicClock(seed=42)
        dur = clock.agent_span_duration_ms()
        assert 1000.0 <= dur <= 10000.0

    def test_llm_span_duration(self):
        clock = DeterministicClock(seed=42)
        dur = clock.llm_span_duration_ms()
        assert 200.0 <= dur <= 5000.0

    def test_tool_span_duration(self):
        clock = DeterministicClock(seed=42)
        dur = clock.tool_span_duration_ms()
        assert 50.0 <= dur <= 2000.0

    def test_eval_span_duration(self):
        clock = DeterministicClock(seed=42)
        dur = clock.eval_span_duration_ms()
        assert 100.0 <= dur <= 1000.0

    def test_ttft(self):
        clock = DeterministicClock(seed=42)
        ttft = clock.ttft_ms(50.0, 500.0)
        assert 50.0 <= ttft <= 500.0

    def test_tpot(self):
        clock = DeterministicClock(seed=42)
        tpot = clock.tpot_ms(10.0, 50.0)
        assert 10.0 <= tpot <= 50.0

    def test_fork(self):
        parent = DeterministicClock(seed=42)
        parent.advance_ms(1000.0)
        child = parent.fork()
        assert child.base_time_ns == parent.cursor_ns
        # Child should be deterministic but different seed
        assert child.seed is not None

    def test_random(self):
        clock = DeterministicClock(seed=42)
        val = clock.random()
        assert 0.0 <= val < 1.0

    def test_randint(self):
        clock = DeterministicClock(seed=42)
        val = clock.randint(1, 10)
        assert 1 <= val <= 10

    def test_choice(self):
        clock = DeterministicClock(seed=42)
        items = ["a", "b", "c"]
        val = clock.choice(items)
        assert val in items

    def test_custom_base_time(self):
        base = 1_700_000_000_000_000_000
        clock = DeterministicClock(seed=42, base_time_ns=base)
        assert clock.base_time_ns == base
        assert clock.now_ns() == base
