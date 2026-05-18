"""Tests for the benchmark importer (CSV, JSON, JSONL, HELM, schema mapping)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from layerlens.benchmarks import ImportResult, BenchmarkImporter


@pytest.fixture
def importer():
    return BenchmarkImporter(imported_by="tests")


class TestCSV:
    def test_import_csv(self, tmp_path: Path, importer: BenchmarkImporter):
        path = tmp_path / "bench.csv"
        path.write_text("question,answer\nq1,a1\nq2,a2\n")
        result = importer.import_csv(str(path))

        assert result.success is True
        assert result.records_imported == 2
        assert result.records[0] == {"question": "q1", "answer": "a1"}
        assert result.metadata is not None
        assert result.metadata.source == "csv"
        assert result.metadata.record_count == 2

    def test_csv_with_schema_mapping(self, tmp_path: Path, importer: BenchmarkImporter):
        path = tmp_path / "bench.csv"
        path.write_text("question,answer\nq1,a1\n")
        result = importer.import_csv(
            str(path),
            schema_mapping={"question": "prompt", "answer": "expected_output"},
        )
        assert result.records[0] == {"prompt": "q1", "expected_output": "a1"}
        assert result.metadata.schema_mapping == {"question": "prompt", "answer": "expected_output"}

    def test_missing_file_returns_failure(self, importer: BenchmarkImporter):
        result = importer.import_csv("/no/such/file.csv")
        assert result.success is False
        assert any("Could not read CSV" in e for e in result.errors)


class TestJSON:
    def test_jsonl(self, tmp_path: Path, importer: BenchmarkImporter):
        path = tmp_path / "bench.jsonl"
        path.write_text('{"prompt": "p1", "answer": "a1"}\n{"prompt": "p2", "answer": "a2"}\n')
        result = importer.import_json(str(path))
        assert result.records_imported == 2
        assert result.records[0]["prompt"] == "p1"
        assert result.metadata.source == "json"

    def test_json_array(self, tmp_path: Path, importer: BenchmarkImporter):
        path = tmp_path / "bench.json"
        path.write_text(json.dumps([{"a": 1}, {"a": 2}]))
        result = importer.import_json(str(path))
        assert result.records_imported == 2

    def test_json_with_records_wrapper(self, tmp_path: Path, importer: BenchmarkImporter):
        path = tmp_path / "bench.json"
        path.write_text(json.dumps({"records": [{"x": 1}], "metadata": "ignored"}))
        result = importer.import_json(str(path))
        assert result.records_imported == 1
        assert result.records[0] == {"x": 1}

    def test_invalid_json_returns_failure(self, tmp_path: Path, importer: BenchmarkImporter):
        path = tmp_path / "bench.json"
        path.write_text("not json")
        result = importer.import_json(str(path))
        assert result.success is False

    def test_json_array_root_with_schema_mapping(self, tmp_path: Path, importer: BenchmarkImporter):
        path = tmp_path / "bench.json"
        path.write_text(json.dumps([{"question": "q1"}, {"question": "q2"}]))
        result = importer.import_json(str(path), schema_mapping={"question": "prompt"})
        assert all("prompt" in r for r in result.records)


class TestHELM:
    def test_helm_instances_key(self, tmp_path: Path, importer: BenchmarkImporter):
        path = tmp_path / "helm.json"
        path.write_text(
            json.dumps(
                {
                    "name": "mmlu-stem",
                    "instances": [
                        {"input": "q1", "references": ["r1"]},
                        {"input": "q2", "references": ["r2"]},
                    ],
                }
            )
        )
        result = importer.import_helm(str(path))
        assert result.records_imported == 2
        assert result.metadata.source == "helm"
        assert result.metadata.name == "mmlu-stem"

    def test_helm_records_list(self, tmp_path: Path, importer: BenchmarkImporter):
        path = tmp_path / "helm.json"
        path.write_text(json.dumps([{"x": 1}, {"x": 2}]))
        result = importer.import_helm(str(path))
        assert result.records_imported == 2

    def test_helm_unreadable_returns_failure(self, importer: BenchmarkImporter):
        result = importer.import_helm("/no/such/helm.json")
        assert result.success is False


class TestSchemaMapping:
    def test_apply_renames_keys(self):
        out = BenchmarkImporter._apply_schema_mapping(
            {"q": "what?", "a": "answer"}, {"q": "prompt", "a": "expected_output"}
        )
        assert out == {"prompt": "what?", "expected_output": "answer"}

    def test_apply_with_no_mapping_is_identity(self):
        record = {"foo": "bar"}
        assert BenchmarkImporter._apply_schema_mapping(record, None) == record

    def test_apply_ignores_unmapped_source_keys(self):
        out = BenchmarkImporter._apply_schema_mapping({"foo": 1}, {"missing": "prompt"})
        assert out == {"foo": 1}


class TestMetadata:
    def test_imported_tracks_metadata(self, tmp_path: Path, importer: BenchmarkImporter):
        path = tmp_path / "b.csv"
        path.write_text("a,b\n1,2\n")
        result = importer.import_csv(str(path))
        assert result.success
        assert len(importer.imported) == 1
        assert importer.imported[0].benchmark_id == result.benchmark_id

    def test_result_to_dict_is_json_serializable(self, tmp_path: Path, importer: BenchmarkImporter):
        path = tmp_path / "b.csv"
        path.write_text("a\n1\n")
        result = importer.import_csv(str(path))
        # round-trip through JSON
        json.dumps(result.to_dict())

    def test_imported_by_propagates(self, tmp_path: Path):
        importer = BenchmarkImporter(imported_by="alice@example.com")
        path = tmp_path / "b.csv"
        path.write_text("a\n1\n")
        result = importer.import_csv(str(path))
        assert result.metadata.imported_by == "alice@example.com"


class TestHuggingFaceMissingDep:
    def test_returns_friendly_error_when_datasets_not_installed(self, importer: BenchmarkImporter, monkeypatch):
        # Force the import inside import_huggingface to fail.
        import builtins

        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "datasets":
                raise ImportError("forced")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        result: ImportResult = importer.import_huggingface("squad")
        assert result.success is False
        assert any("datasets" in e and "not installed" in e for e in result.errors)
