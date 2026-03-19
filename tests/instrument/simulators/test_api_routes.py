"""Simulator API route tests.

NOTE: These tests require the server-side stratix.api package which is not
part of the SDK. They are skipped in the SDK test suite.
"""

import json
from urllib.parse import urlparse, parse_qs

import pytest

pytest.skip("Requires server-side stratix.api package", allow_module_level=True)

from layerlens.instrument.simulators.run_store import RunStore  # noqa: E402


@pytest.fixture
def run_store(tmp_path):
    return RunStore(store_dir=str(tmp_path / "runs"))


@pytest.fixture
def app(run_store):
    a = create_app(auth_required=False)
    # Inject the run store so endpoints use our temp dir
    a._simulator_run_store = run_store
    return a


def _call(app, method, path, body=None):
    """Call an API endpoint and return the response."""
    body_bytes = json.dumps(body).encode() if body else None
    parsed = urlparse(path)
    raw_qs = parse_qs(parsed.query)
    query_params = {k: ",".join(v) if len(v) > 1 else v[0] for k, v in raw_qs.items()}
    response = app.handle_request(
        method, parsed.path, {}, body_bytes, query_params
    )
    return response.status, response.body


class TestGetHome:
    def test_returns_presets(self, app):
        status, resp = _call(app, "GET", "/api/v1/simulator/home")
        assert status == 200
        data = resp["data"]
        assert len(data["presets"]) == 3
        names = [p["name"] for p in data["presets"]]
        assert "Minimal" in names
        assert "Standard" in names
        assert "Full" in names

    def test_returns_coverage_maps(self, app):
        status, resp = _call(app, "GET", "/api/v1/simulator/home")
        data = resp["data"]
        # 12 source formats
        assert len(data["source_coverage"]) == 12
        # 5 scenarios
        assert len(data["scenario_coverage"]) == 5

    def test_returns_empty_recent_runs(self, app):
        status, resp = _call(app, "GET", "/api/v1/simulator/home")
        data = resp["data"]
        assert data["recent_runs"] == []


class TestStartGenerate:
    def test_preset_minimal(self, app):
        status, resp = _call(app, "POST", "/api/v1/simulator/generate", {"preset": "minimal"})
        assert status == 200
        data = resp["data"]
        assert data["status"] == "complete"
        assert data["trace_count"] >= 1
        assert data["run_id"].startswith("run_")

    def test_preset_standard(self, app):
        status, resp = _call(app, "POST", "/api/v1/simulator/generate", {"preset": "standard"})
        assert status == 200
        data = resp["data"]
        assert data["status"] == "complete"
        assert data["trace_count"] >= 1

    def test_preset_full(self, app):
        status, resp = _call(app, "POST", "/api/v1/simulator/generate", {"preset": "full"})
        assert status == 200
        data = resp["data"]
        assert data["status"] == "complete"
        assert data["trace_count"] >= 1

    def test_custom_config(self, app):
        status, resp = _call(app, "POST", "/api/v1/simulator/generate", {
            "source_format": "openai",
            "scenario": "customer_service",
            "count": 3,
            "seed": 42,
        })
        assert status == 200
        data = resp["data"]
        assert data["status"] == "complete"
        assert data["trace_count"] == 3

    def test_invalid_preset(self, app):
        status, resp = _call(app, "POST", "/api/v1/simulator/generate", {"preset": "invalid"})
        assert status == 400
        assert "error" in resp

    def test_missing_body(self, app):
        status, resp = _call(app, "POST", "/api/v1/simulator/generate")
        assert status == 400

    def test_run_stored_after_generate(self, app, run_store):
        status, resp = _call(app, "POST", "/api/v1/simulator/generate", {"preset": "minimal"})
        run_id = resp["data"]["run_id"]
        record = run_store.get(run_id)
        assert record is not None
        assert record.status == "complete"
        assert record.trace_count >= 1


class TestListRuns:
    def test_empty(self, app):
        status, resp = _call(app, "GET", "/api/v1/simulator/runs")
        assert status == 200
        assert resp["data"]["runs"] == []
        assert resp["data"]["pagination"]["total"] == 0

    def test_after_generate(self, app):
        _call(app, "POST", "/api/v1/simulator/generate", {"preset": "minimal"})
        status, resp = _call(app, "GET", "/api/v1/simulator/runs")
        assert status == 200
        assert len(resp["data"]["runs"]) == 1

    def test_pagination(self, app):
        for _ in range(3):
            _call(app, "POST", "/api/v1/simulator/generate", {"preset": "minimal"})
        # limit=2
        status, resp = _call(app, "GET", "/api/v1/simulator/runs?limit=2")
        assert status == 200
        assert len(resp["data"]["runs"]) == 2
        assert resp["data"]["pagination"]["has_more"] is True


class TestGetRun:
    def test_valid_run(self, app):
        _, gen_resp = _call(app, "POST", "/api/v1/simulator/generate", {"preset": "minimal"})
        run_id = gen_resp["data"]["run_id"]
        status, resp = _call(app, "GET", f"/api/v1/simulator/runs/{run_id}")
        assert status == 200
        data = resp["data"]
        assert data["run_id"] == run_id
        assert data["status"] == "complete"
        assert data["progress"] == 1.0

    def test_invalid_run(self, app):
        status, resp = _call(app, "GET", "/api/v1/simulator/runs/nonexistent")
        assert status == 404


class TestGetRunTraces:
    def test_traces_returned(self, app):
        _, gen_resp = _call(app, "POST", "/api/v1/simulator/generate", {"preset": "minimal"})
        run_id = gen_resp["data"]["run_id"]
        status, resp = _call(app, "GET", f"/api/v1/simulator/runs/{run_id}/traces")
        assert status == 200
        data = resp["data"]
        assert data["run_id"] == run_id
        assert data["count"] >= 1
        assert isinstance(data["traces"], list)

    def test_invalid_run(self, app):
        status, resp = _call(app, "GET", "/api/v1/simulator/runs/nonexistent/traces")
        assert status == 404


class TestGetAudit:
    def test_empty_audit(self, app):
        status, resp = _call(app, "GET", "/api/v1/simulator/audit")
        assert status == 200
        data = resp["data"]
        assert data["total_runs"] == 0
        assert data["total_traces"] == 0
        assert isinstance(data["heatmap"], list)
        assert len(data["source_labels"]) == 12
        assert len(data["scenario_labels"]) == 5

    def test_audit_after_runs(self, app):
        _call(app, "POST", "/api/v1/simulator/generate", {"preset": "minimal"})
        _call(app, "POST", "/api/v1/simulator/generate", {"preset": "standard"})
        status, resp = _call(app, "GET", "/api/v1/simulator/audit")
        assert status == 200
        data = resp["data"]
        assert data["total_runs"] == 2
        assert data["total_traces"] >= 2


class TestListSources:
    def test_returns_12_sources(self, app):
        status, resp = _call(app, "GET", "/api/v1/simulator/sources")
        assert status == 200
        data = resp["data"]
        assert data["count"] == 12
        values = [s["value"] for s in data["sources"]]
        assert "openai" in values
        assert "anthropic" in values
        assert "generic_otel" in values


class TestListScenarios:
    def test_returns_5_scenarios(self, app):
        status, resp = _call(app, "GET", "/api/v1/simulator/scenarios")
        assert status == 200
        data = resp["data"]
        assert data["count"] == 5
        values = [s["value"] for s in data["scenarios"]]
        assert "customer_service" in values
        assert "sales" in values


class TestValidateRun:
    def test_validate_complete_run(self, app):
        _, gen_resp = _call(app, "POST", "/api/v1/simulator/generate", {"preset": "minimal"})
        run_id = gen_resp["data"]["run_id"]
        status, resp = _call(app, "POST", "/api/v1/simulator/validate", {"run_id": run_id})
        assert status == 200
        data = resp["data"]
        assert data["run_id"] == run_id
        assert data["validation_status"] in ("pass", "warn", "fail")
        assert len(data["checks"]) == 6

    def test_validate_missing_run_id(self, app):
        status, resp = _call(app, "POST", "/api/v1/simulator/validate", {})
        assert status == 400

    def test_validate_nonexistent_run(self, app):
        status, resp = _call(app, "POST", "/api/v1/simulator/validate", {"run_id": "nope"})
        assert status == 404


class TestAutoIngestion:
    """Verify that generate auto-ingests traces (like an external source)."""

    def test_generate_returns_ingestion_stats(self, app):
        status, resp = _call(app, "POST", "/api/v1/simulator/generate", {"preset": "minimal"})
        assert status == 200
        data = resp["data"]
        # Without a pipeline configured, ingested_traces should be 0
        assert "ingested_traces" in data
        assert "ingested_events" in data
        assert isinstance(data["ingested_traces"], int)

    def test_generate_with_pipeline_ingests(self, app, run_store):
        """When pipeline is present, traces are auto-ingested."""
        # Set up a mock pipeline that records ingest calls
        ingest_calls = []

        class MockPipeline:
            def ingest(self, events, tenant_id="default"):
                ingest_calls.append({"events": events, "tenant_id": tenant_id})
                from types import SimpleNamespace
                return SimpleNamespace(
                    accepted_count=len(events), rejected_count=0,
                    errors=[], trace_ids=[], processing_time_ms=0.0,
                )

        app.pipeline = MockPipeline()
        status, resp = _call(app, "POST", "/api/v1/simulator/generate", {"preset": "minimal"})
        assert status == 200
        data = resp["data"]
        # Pipeline should have been called — at least 1 trace ingested
        assert data["ingested_traces"] >= 1
        assert data["ingested_events"] >= 1
        assert len(ingest_calls) >= 1
        # Verify tenant_id passed as "simulator"
        assert all(c["tenant_id"] == "simulator" for c in ingest_calls)
        app.pipeline = None  # clean up

    def test_re_ingest_endpoint(self, app, run_store):
        """POST /simulator/ingest re-ingests a completed run."""
        ingest_calls = []

        class MockPipeline:
            def ingest(self, events, tenant_id="default"):
                ingest_calls.append({"events": events, "tenant_id": tenant_id})
                from types import SimpleNamespace
                return SimpleNamespace(
                    accepted_count=len(events), rejected_count=0,
                    errors=[], trace_ids=[], processing_time_ms=0.0,
                )

        # First generate without pipeline
        _call(app, "POST", "/api/v1/simulator/generate", {"preset": "minimal"})
        # Now attach pipeline and re-ingest
        app.pipeline = MockPipeline()
        runs_resp = _call(app, "GET", "/api/v1/simulator/runs")
        run_id = runs_resp[1]["data"]["runs"][0]["run_id"]
        status, resp = _call(app, "POST", "/api/v1/simulator/ingest", {"run_id": run_id})
        assert status == 200
        assert resp["data"]["ingested_traces"] >= 1
        assert resp["data"]["ingested_events"] >= 1
        assert len(ingest_calls) >= 1
        app.pipeline = None

    def test_re_ingest_no_pipeline_returns_503(self, app):
        _call(app, "POST", "/api/v1/simulator/generate", {"preset": "minimal"})
        runs_resp = _call(app, "GET", "/api/v1/simulator/runs")
        run_id = runs_resp[1]["data"]["runs"][0]["run_id"]
        status, resp = _call(app, "POST", "/api/v1/simulator/ingest", {"run_id": run_id})
        assert status == 503

    def test_re_ingest_missing_run(self, app):
        status, resp = _call(app, "POST", "/api/v1/simulator/ingest", {"run_id": "nope"})
        assert status == 404


class TestHomeAfterRuns:
    def test_recent_runs_populated(self, app):
        _call(app, "POST", "/api/v1/simulator/generate", {"preset": "minimal"})
        status, resp = _call(app, "GET", "/api/v1/simulator/home")
        data = resp["data"]
        assert len(data["recent_runs"]) == 1
        # Coverage should show at least one source and scenario covered
        covered_sources = [k for k, v in data["source_coverage"].items() if v]
        covered_scenarios = [k for k, v in data["scenario_coverage"].items() if v]
        assert len(covered_sources) >= 1
        assert len(covered_scenarios) >= 1
