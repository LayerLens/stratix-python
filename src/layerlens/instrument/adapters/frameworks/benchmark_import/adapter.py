"""
STRATIX Benchmark Import Adapter (ADP-074)

Imports external benchmark datasets from:
- HuggingFace Datasets (via ``datasets`` library with streaming)
- HELM (Holistic Evaluation of Language Models) JSON results
- Custom sources: CSV, JSON, Parquet files

Features:
- Automatic schema detection and mapping to Stratix benchmark format
- Versioned tracking with source, version, and import timestamp
- Comparison of external benchmark scores with internal evaluations
"""

from __future__ import annotations

import csv
import json
import time
import uuid
import logging
from typing import Any, Optional
from pathlib import Path
from datetime import datetime, timezone

# Python 3.11+ exposes ``datetime.UTC``; we alias to ``timezone.utc`` for 3.8+ compat.
UTC = timezone.utc

from pydantic import Field, BaseModel

logger = logging.getLogger(__name__)


class BenchmarkMetadata(BaseModel):
    """Metadata for an imported benchmark."""

    benchmark_id: str = Field(default_factory=lambda: f"bench-{uuid.uuid4().hex[:12]}")
    name: str = Field(description="Benchmark name")
    source: str = Field(description="Import source (huggingface, helm, csv, json, parquet)")
    source_identifier: str = Field(
        default="", description="Source-specific ID (e.g., HF dataset name)"
    )
    version: str = Field(default="1.0.0", description="Benchmark version")
    record_count: int = Field(default=0, description="Number of records imported")
    schema_mapping: dict[str, str] = Field(
        default_factory=dict, description="Field mapping applied"
    )
    imported_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
    )
    imported_by: str = Field(default="", description="User who triggered the import")
    tags: list[str] = Field(default_factory=list)


class ImportResult(BaseModel):
    """Result of a benchmark import operation."""

    success: bool = Field(default=True)
    benchmark_id: str = Field(default="")
    records_imported: int = Field(default=0)
    records_skipped: int = Field(default=0)
    duration_ms: float = Field(default=0.0)
    errors: list[str] = Field(default_factory=list)
    # Use Optional[...] (not `X | None`) so Pydantic 2 can resolve the field
    # annotation under Python 3.9 — `from __future__ import annotations` does
    # not help here because Pydantic eagerly evaluates the forward ref.
    metadata: Optional[BenchmarkMetadata] = Field(default=None)


class BenchmarkImportAdapter:
    """
    Imports external benchmark datasets into Stratix evaluation spaces.

    Usage::

        adapter = BenchmarkImportAdapter()

        # Import from HuggingFace
        result = adapter.import_huggingface("squad", split="validation")

        # Import from HELM results
        result = adapter.import_helm("/path/to/helm_results.json")

        # Import from CSV
        result = adapter.import_csv("/path/to/benchmark.csv", schema_mapping={
            "question": "prompt",
            "answer": "expected_output",
        })
    """

    def __init__(self, store: Any | None = None) -> None:
        """
        Args:
            store: Optional storage backend for persisting imported benchmarks.
                   If None, benchmarks are returned in-memory only.
        """
        self._store = store
        self._benchmarks: dict[str, BenchmarkMetadata] = {}

    # -- HuggingFace Datasets ----------------------------------------------

    def import_huggingface(
        self,
        dataset_name: str,
        split: str = "test",
        subset: str | None = None,
        schema_mapping: dict[str, str] | None = None,
        max_records: int | None = None,
        tags: list[str] | None = None,
    ) -> ImportResult:
        """Import a benchmark from HuggingFace Datasets.

        Args:
            dataset_name: HuggingFace dataset identifier (e.g., "squad", "mmlu").
            split: Dataset split to import (default: "test").
            subset: Optional dataset subset/config.
            schema_mapping: Optional field mapping override.
            max_records: Maximum number of records to import.
            tags: Optional tags for categorization.

        Returns:
            ImportResult with import statistics and metadata.
        """
        start = time.monotonic()
        errors: list[str] = []
        records: list[dict[str, Any]] = []

        try:
            import datasets as hf_datasets  # type: ignore[import-not-found,unused-ignore]

            load_kwargs: dict[str, Any] = {"path": dataset_name, "split": split, "streaming": True}
            if subset:
                load_kwargs["name"] = subset

            ds = hf_datasets.load_dataset(**load_kwargs)

            count = 0
            for record in ds:
                if max_records and count >= max_records:
                    break
                mapped = self._apply_schema_mapping(dict(record), schema_mapping)
                records.append(mapped)
                count += 1  # noqa: SIM113

        except ImportError:
            errors.append("'datasets' library not installed. Run: pip install datasets")
            return ImportResult(success=False, errors=errors)
        except Exception as exc:
            errors.append(f"HuggingFace import failed: {exc}")
            return ImportResult(success=False, errors=errors)

        elapsed_ms = (time.monotonic() - start) * 1000

        metadata = BenchmarkMetadata(
            name=dataset_name,
            source="huggingface",
            source_identifier=f"{dataset_name}/{subset or 'default'}/{split}",
            record_count=len(records),
            schema_mapping=schema_mapping or {},
            tags=tags or ["huggingface"],
        )

        self._benchmarks[metadata.benchmark_id] = metadata
        self._persist(metadata, records)

        return ImportResult(
            success=True,
            benchmark_id=metadata.benchmark_id,
            records_imported=len(records),
            duration_ms=round(elapsed_ms, 2),
            metadata=metadata,
        )

    # -- HELM Results ------------------------------------------------------

    def import_helm(
        self,
        path: str,
        schema_mapping: dict[str, str] | None = None,
        tags: list[str] | None = None,
    ) -> ImportResult:
        """Import HELM benchmark results from a JSON file.

        Args:
            path: Path to HELM results JSON file.
            schema_mapping: Optional field mapping override.
            tags: Optional tags.

        Returns:
            ImportResult with import statistics.
        """
        start = time.monotonic()
        errors: list[str] = []
        records: list[dict[str, Any]] = []

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            # HELM format: list of scenario results with instances
            scenarios = (
                data if isinstance(data, list) else data.get("results", data.get("scenarios", []))
            )
            if isinstance(scenarios, dict):
                scenarios = [scenarios]

            for scenario in scenarios:
                instances = scenario.get("instances", scenario.get("results", []))
                if isinstance(instances, list):
                    for inst in instances:
                        mapped = self._apply_schema_mapping(dict(inst), schema_mapping)
                        mapped.setdefault("scenario", scenario.get("scenario", ""))
                        mapped.setdefault("model", scenario.get("model", ""))
                        records.append(mapped)
                else:
                    mapped = self._apply_schema_mapping(dict(scenario), schema_mapping)
                    records.append(mapped)

        except FileNotFoundError:
            errors.append(f"File not found: {path}")
            return ImportResult(success=False, errors=errors)
        except json.JSONDecodeError as exc:
            errors.append(f"Invalid JSON: {exc}")
            return ImportResult(success=False, errors=errors)
        except Exception as exc:
            errors.append(f"HELM import failed: {exc}")
            return ImportResult(success=False, errors=errors)

        elapsed_ms = (time.monotonic() - start) * 1000

        metadata = BenchmarkMetadata(
            name=Path(path).stem,
            source="helm",
            source_identifier=path,
            record_count=len(records),
            schema_mapping=schema_mapping or {},
            tags=tags or ["helm"],
        )

        self._benchmarks[metadata.benchmark_id] = metadata
        self._persist(metadata, records)

        return ImportResult(
            success=True,
            benchmark_id=metadata.benchmark_id,
            records_imported=len(records),
            duration_ms=round(elapsed_ms, 2),
            metadata=metadata,
        )

    # -- CSV / JSON / Parquet ----------------------------------------------

    def import_csv(
        self,
        path: str,
        schema_mapping: dict[str, str] | None = None,
        delimiter: str = ",",
        max_records: int | None = None,
        tags: list[str] | None = None,
    ) -> ImportResult:
        """Import a benchmark from a CSV file."""
        start = time.monotonic()
        errors: list[str] = []
        records: list[dict[str, Any]] = []

        try:
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                for i, row in enumerate(reader):
                    if max_records and i >= max_records:
                        break
                    mapped = self._apply_schema_mapping(dict(row), schema_mapping)
                    records.append(mapped)
        except Exception as exc:
            errors.append(f"CSV import failed: {exc}")
            return ImportResult(success=False, errors=errors)

        elapsed_ms = (time.monotonic() - start) * 1000

        metadata = BenchmarkMetadata(
            name=Path(path).stem,
            source="csv",
            source_identifier=path,
            record_count=len(records),
            schema_mapping=schema_mapping or {},
            tags=tags or ["csv"],
        )

        self._benchmarks[metadata.benchmark_id] = metadata
        self._persist(metadata, records)

        return ImportResult(
            success=True,
            benchmark_id=metadata.benchmark_id,
            records_imported=len(records),
            duration_ms=round(elapsed_ms, 2),
            metadata=metadata,
        )

    def import_json(
        self,
        path: str,
        schema_mapping: dict[str, str] | None = None,
        records_key: str | None = None,
        max_records: int | None = None,
        tags: list[str] | None = None,
    ) -> ImportResult:
        """Import a benchmark from a JSON file (array or object with records key)."""
        start = time.monotonic()
        errors: list[str] = []
        records: list[dict[str, Any]] = []

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)

            items = data
            if isinstance(data, dict):
                items = data.get(records_key or "records", data.get("data", []))
            if not isinstance(items, list):
                items = [items]

            for i, item in enumerate(items):
                if max_records and i >= max_records:
                    break
                mapped = self._apply_schema_mapping(dict(item), schema_mapping)
                records.append(mapped)
        except Exception as exc:
            errors.append(f"JSON import failed: {exc}")
            return ImportResult(success=False, errors=errors)

        elapsed_ms = (time.monotonic() - start) * 1000

        metadata = BenchmarkMetadata(
            name=Path(path).stem,
            source="json",
            source_identifier=path,
            record_count=len(records),
            schema_mapping=schema_mapping or {},
            tags=tags or ["json"],
        )

        self._benchmarks[metadata.benchmark_id] = metadata
        self._persist(metadata, records)

        return ImportResult(
            success=True,
            benchmark_id=metadata.benchmark_id,
            records_imported=len(records),
            duration_ms=round(elapsed_ms, 2),
            metadata=metadata,
        )

    def import_parquet(
        self,
        path: str,
        schema_mapping: dict[str, str] | None = None,
        max_records: int | None = None,
        tags: list[str] | None = None,
    ) -> ImportResult:
        """Import a benchmark from a Parquet file."""
        start = time.monotonic()
        errors: list[str] = []
        records: list[dict[str, Any]] = []

        try:
            import pyarrow.parquet as pq  # type: ignore[import-untyped,unused-ignore]

            table = pq.read_table(path)  # type: ignore[no-untyped-call,unused-ignore]
            df_dicts = table.to_pydict()

            # Convert columnar to row-based
            keys = list(df_dicts.keys())
            num_rows = len(df_dicts[keys[0]]) if keys else 0

            for i in range(min(num_rows, max_records or num_rows)):
                row = {k: df_dicts[k][i] for k in keys}
                mapped = self._apply_schema_mapping(row, schema_mapping)
                records.append(mapped)

        except ImportError:
            errors.append("'pyarrow' library not installed. Run: pip install pyarrow")
            return ImportResult(success=False, errors=errors)
        except Exception as exc:
            errors.append(f"Parquet import failed: {exc}")
            return ImportResult(success=False, errors=errors)

        elapsed_ms = (time.monotonic() - start) * 1000

        metadata = BenchmarkMetadata(
            name=Path(path).stem,
            source="parquet",
            source_identifier=path,
            record_count=len(records),
            schema_mapping=schema_mapping or {},
            tags=tags or ["parquet"],
        )

        self._benchmarks[metadata.benchmark_id] = metadata
        self._persist(metadata, records)

        return ImportResult(
            success=True,
            benchmark_id=metadata.benchmark_id,
            records_imported=len(records),
            duration_ms=round(elapsed_ms, 2),
            metadata=metadata,
        )

    # -- Query -------------------------------------------------------------

    def list_benchmarks(self) -> list[BenchmarkMetadata]:
        """Return metadata for all imported benchmarks."""
        return list(self._benchmarks.values())

    def get_benchmark(self, benchmark_id: str) -> BenchmarkMetadata | None:
        """Return metadata for a specific benchmark."""
        return self._benchmarks.get(benchmark_id)

    # -- Internal ----------------------------------------------------------

    @staticmethod
    def _apply_schema_mapping(
        record: dict[str, Any],
        mapping: dict[str, str] | None,
    ) -> dict[str, Any]:
        """Apply field name mapping to a record."""
        if not mapping:
            return record
        result: dict[str, Any] = {}
        for src_key, value in record.items():
            dst_key = mapping.get(src_key, src_key)
            result[dst_key] = value
        return result

    def _persist(self, metadata: BenchmarkMetadata, records: list[dict[str, Any]]) -> None:
        """Persist benchmark metadata and records to the store."""
        if self._store is None:
            return
        try:
            self._store.insert_row("benchmarks", metadata.model_dump())
            for record in records:
                record["benchmark_id"] = metadata.benchmark_id
                self._store.insert_row("benchmark_records", record)
        except Exception:
            logger.debug("Failed to persist benchmark %s", metadata.benchmark_id, exc_info=True)
