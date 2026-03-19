"""Tests for IDGenerator."""

import re

from layerlens.instrument.simulators.identifiers import IDGenerator


class TestIDGenerator:
    def test_trace_id_format(self):
        gen = IDGenerator(seed=42)
        tid = gen.trace_id()
        assert len(tid) == 32
        assert all(c in "0123456789abcdef" for c in tid)

    def test_span_id_format(self):
        gen = IDGenerator(seed=42)
        sid = gen.span_id()
        assert len(sid) == 16
        assert all(c in "0123456789abcdef" for c in sid)

    def test_deterministic(self):
        gen1 = IDGenerator(seed=42)
        gen2 = IDGenerator(seed=42)
        assert gen1.trace_id() == gen2.trace_id()
        assert gen1.span_id() == gen2.span_id()

    def test_different_seeds(self):
        gen1 = IDGenerator(seed=42)
        gen2 = IDGenerator(seed=99)
        assert gen1.trace_id() != gen2.trace_id()

    def test_traceparent(self):
        gen = IDGenerator(seed=42)
        tid = gen.trace_id()
        sid = gen.span_id()
        tp = gen.traceparent(tid, sid)
        assert tp.startswith("00-")
        assert tp.endswith("-01")
        parts = tp.split("-")
        assert len(parts) == 4
        assert parts[1] == tid
        assert parts[2] == sid

    def test_traceparent_unsampled(self):
        gen = IDGenerator(seed=42)
        tp = gen.traceparent("a" * 32, "b" * 16, sampled=False)
        assert tp.endswith("-00")

    def test_salesforce_id(self):
        gen = IDGenerator(seed=42)
        sf_id = gen.salesforce_id()
        assert len(sf_id) == 18
        assert sf_id.isalnum()

    def test_response_id_openai(self):
        gen = IDGenerator(seed=42)
        rid = gen.response_id_openai()
        assert rid.startswith("chatcmpl-")
        assert len(rid) == len("chatcmpl-") + 29

    def test_response_id_anthropic(self):
        gen = IDGenerator(seed=42)
        rid = gen.response_id_anthropic()
        assert rid.startswith("msg_")

    def test_response_id_vertex(self):
        gen = IDGenerator(seed=42)
        rid = gen.response_id_vertex()
        # UUID format
        assert re.match(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$", rid)

    def test_system_fingerprint(self):
        gen = IDGenerator(seed=42)
        fp = gen.system_fingerprint()
        assert fp.startswith("fp_")

    def test_tool_call_id(self):
        gen = IDGenerator(seed=42)
        tcid = gen.tool_call_id()
        assert tcid.startswith("call_")

    def test_session_id(self):
        gen = IDGenerator(seed=42)
        sid = gen.session_id()
        # UUID format
        assert len(sid) == 36
        assert "-" in sid

    def test_run_id(self):
        gen = IDGenerator(seed=42)
        rid = gen.run_id()
        assert rid.startswith("run_")
        assert len(rid) == 12

    def test_langfuse_trace_id(self):
        gen = IDGenerator(seed=42)
        lid = gen.langfuse_trace_id()
        assert len(lid) == 36

    def test_uniqueness(self):
        gen = IDGenerator(seed=42)
        ids = {gen.trace_id() for _ in range(100)}
        assert len(ids) == 100

    def test_unseeded_random(self):
        gen = IDGenerator(seed=None)
        tid = gen.trace_id()
        assert len(tid) == 32
