"""Tests for simulator UI view models.

NOTE: These tests require the server-side stratix.ui package which is not
part of the SDK. They are skipped in the SDK test suite.
"""

import pytest

pytest.skip("Requires server-side stratix.ui package", allow_module_level=True)


class TestPresetCardView:
    def test_to_dict(self):
        card = PresetCardView(
            name="Minimal",
            description="1 trace, no errors",
            trace_count=1,
            features=["Template content"],
        )
        d = card.to_dict()
        assert d["name"] == "Minimal"
        assert d["trace_count"] == 1
        assert "Template content" in d["features"]


class TestRunSummaryView:
    def test_to_dict(self):
        view = RunSummaryView(
            run_id="run_123",
            source="openai",
            scenario="customer_service",
            count=25,
            status="PASS",
            date="2026-03-01T10:00:00",
        )
        d = view.to_dict()
        assert d["run_id"] == "run_123"
        assert d["status"] == "PASS"


class TestSpanView:
    def test_to_dict(self):
        span = SpanView(
            span_id="span_001",
            span_type="llm",
            name="chat gpt-4o",
            depth=1,
            duration_ms=123.4,
            prompt_tokens=250,
            completion_tokens=180,
        )
        d = span.to_dict()
        assert d["span_type"] == "llm"
        assert d["prompt_tokens"] == 250


class TestTraceDetailView:
    def test_to_dict(self):
        view = TraceDetailView(
            trace_id="trace_001",
            source="openai",
            model="gpt-4o",
            scenario="customer_service",
            topic="Shipping_Delay",
            spans=[
                SpanView(
                    span_id="s1", span_type="agent", name="Agent",
                    depth=0, duration_ms=1000.0,
                ),
            ],
            attributes={"gen_ai.system": "openai"},
        )
        d = view.to_dict()
        assert d["trace_id"] == "trace_001"
        assert len(d["spans"]) == 1
        assert d["attributes"]["gen_ai.system"] == "openai"


class TestValidationBadge:
    def test_to_dict(self):
        badge = ValidationBadge(name="OTLP Schema", status="PASS")
        d = badge.to_dict()
        assert d["name"] == "OTLP Schema"
        assert d["status"] == "PASS"


class TestRunProgressView:
    def test_to_dict(self):
        view = RunProgressView(
            run_id="run_001",
            status=RunStatus.GENERATING,
            progress=0.5,
            generated_count=10,
            total_count=20,
            validated_count=10,
            errors_injected=2,
            elapsed_seconds=3.5,
            live_traces=[],
            config_summary={"source": "openai"},
        )
        d = view.to_dict()
        assert d["status"] == "generating"
        assert d["progress"] == 0.5


class TestTraceReviewView:
    def test_to_dict(self):
        view = TraceReviewView(
            run_id="run_001",
            traces=[],
            selected_trace=None,
            validation_results=[
                ValidationBadge(name="OTLP Schema", status="PASS"),
            ],
            filters=FilterState(),
        )
        d = view.to_dict()
        assert d["run_id"] == "run_001"
        assert len(d["validation_results"]) == 1
        assert d["selected_trace"] is None


class TestAuditView:
    def test_to_dict(self):
        view = AuditView(
            total_runs=47,
            total_traces=892,
            pass_rate=0.97,
            sources_used=12,
            heatmap=[[0] * 5 for _ in range(12)],
            run_history=[],
        )
        d = view.to_dict()
        assert d["total_runs"] == 47
        assert d["pass_rate"] == 0.97
        assert len(d["heatmap"]) == 12


class TestConfigDiffView:
    def test_to_dict(self):
        view = ConfigDiffView(
            run_id_a="run_a",
            run_id_b="run_b",
            added={"streaming": True},
            removed={"dry_run": True},
            changed={"count": (10, 25)},
        )
        d = view.to_dict()
        assert d["added"]["streaming"] is True
        assert d["changed"]["count"]["old"] == 10
        assert d["changed"]["count"]["new"] == 25


class TestBuildSimulatorHome:
    def test_empty_runs(self):
        view = build_simulator_home()
        d = view.to_dict()
        assert len(d["presets"]) == 3
        assert len(d["recent_runs"]) == 0
        assert len(d["source_coverage"]) == len(SourceFormat)
        assert len(d["scenario_coverage"]) == len(ScenarioName)

    def test_with_runs(self):
        from layerlens.instrument.simulators.config import SimulatorConfig
        record = RunRecord(
            run_id="run_test",
            config=SimulatorConfig.minimal().model_dump(),
            status="pass",
            trace_count=1,
        )
        view = build_simulator_home(runs=[record])
        d = view.to_dict()
        assert len(d["recent_runs"]) == 1
