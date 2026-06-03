"""Import external benchmark datasets into the layerlens benchmark format.

Supported sources:

* **HuggingFace Datasets** — via the optional ``datasets`` package.
* **HELM results** — read the JSON file produced by Stanford HELM.
* **CSV / JSON / JSONL** — local files with optional schema-mapping.

A ``schema_mapping`` dict renames source fields to the layerlens canonical
field names (typically ``prompt`` / ``expected_output`` / ``metadata``).
Records that don't match the mapping pass through unchanged.

This module is intentionally NOT an instrumentation adapter — it converts
external data formats to layerlens benchmark records. Ateam's analogue
lives under ``stratix.sdk.python.adapters.benchmark_import`` for legacy
reasons (see ateam's own docstring noting the inconsistency); we ship it
in its own subpackage instead.
"""

from __future__ import annotations

import csv
import json
import time
import uuid
import logging
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime, timezone
from dataclasses import field, asdict, dataclass

log = logging.getLogger(__name__)


@dataclass
class BenchmarkMetadata:
    """Descriptive metadata for an imported benchmark."""

    name: str
    source: str
    benchmark_id: str = field(default_factory=lambda: f"bench-{uuid.uuid4().hex[:12]}")
    source_identifier: str = ""
    version: str = "1.0.0"
    record_count: int = 0
    schema_mapping: Dict[str, str] = field(default_factory=dict)
    imported_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    imported_by: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ImportResult:
    """Result of one import call."""

    success: bool = True
    benchmark_id: str = ""
    records_imported: int = 0
    records_skipped: int = 0
    duration_ms: float = 0.0
    errors: List[str] = field(default_factory=list)
    metadata: Optional[BenchmarkMetadata] = None
    records: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "benchmark_id": self.benchmark_id,
            "records_imported": self.records_imported,
            "records_skipped": self.records_skipped,
            "duration_ms": self.duration_ms,
            "errors": list(self.errors),
            "metadata": self.metadata.to_dict() if self.metadata else None,
        }


class BenchmarkImporter:
    """Convert external benchmark datasets into layerlens records.

    Each ``import_*`` method returns an :class:`ImportResult` carrying the
    metadata, the parsed records, and any errors. Callers can then post
    the records to ``client.benchmarks.create`` (or persist them however
    they wish).

    Usage::

        importer = BenchmarkImporter()
        result = importer.import_huggingface("squad", split="validation")
        if result.success:
            for record in result.records:
                ...
    """

    def __init__(self, imported_by: str = "") -> None:
        self._imported_by = imported_by
        self._benchmarks: Dict[str, BenchmarkMetadata] = {}

    # ------------------------------------------------------------------
    # HuggingFace
    # ------------------------------------------------------------------

    def import_huggingface(
        self,
        dataset_name: str,
        *,
        split: str = "test",
        subset: Optional[str] = None,
        schema_mapping: Optional[Dict[str, str]] = None,
        max_records: Optional[int] = None,
        tags: Optional[List[str]] = None,
    ) -> ImportResult:
        """Import a benchmark from HuggingFace Datasets (streaming)."""
        start = time.monotonic()
        records: List[Dict[str, Any]] = []
        errors: List[str] = []

        try:
            import datasets as hf_datasets  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]
        except ImportError:
            return ImportResult(
                success=False,
                errors=["'datasets' library not installed. Run: pip install datasets"],
            )

        try:
            load_kwargs: Dict[str, Any] = {
                "path": dataset_name,
                "split": split,
                "streaming": True,
            }
            if subset:
                load_kwargs["name"] = subset
            ds = hf_datasets.load_dataset(**load_kwargs)
            for record in ds:
                if max_records is not None and len(records) >= max_records:
                    break
                records.append(self._apply_schema_mapping(dict(record), schema_mapping))
        except Exception as exc:
            errors.append(f"HuggingFace import failed: {exc}")
            return ImportResult(success=False, errors=errors)

        metadata = BenchmarkMetadata(
            name=dataset_name,
            source="huggingface",
            source_identifier=f"{dataset_name}/{subset or 'default'}/{split}",
            record_count=len(records),
            schema_mapping=schema_mapping or {},
            tags=list(tags or []) or ["huggingface"],
            imported_by=self._imported_by,
        )
        return self._finalize(metadata, records, errors, start)

    # ------------------------------------------------------------------
    # HELM
    # ------------------------------------------------------------------

    def import_helm(
        self,
        path: str,
        *,
        schema_mapping: Optional[Dict[str, str]] = None,
        tags: Optional[List[str]] = None,
    ) -> ImportResult:
        """Import a HELM-style JSON results file."""
        start = time.monotonic()
        try:
            with open(path) as fh:
                blob = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            return ImportResult(success=False, errors=[f"Could not read HELM file: {exc}"])

        raw_records = self._extract_helm_records(blob)
        records = [self._apply_schema_mapping(r, schema_mapping) for r in raw_records]

        metadata = BenchmarkMetadata(
            name=(blob.get("name") or path.split("/")[-1] if isinstance(blob, dict) else path),
            source="helm",
            source_identifier=path,
            record_count=len(records),
            schema_mapping=schema_mapping or {},
            tags=list(tags or []) or ["helm"],
            imported_by=self._imported_by,
        )
        return self._finalize(metadata, records, [], start)

    # ------------------------------------------------------------------
    # CSV / JSON / JSONL
    # ------------------------------------------------------------------

    def import_csv(
        self,
        path: str,
        *,
        schema_mapping: Optional[Dict[str, str]] = None,
        tags: Optional[List[str]] = None,
        delimiter: str = ",",
    ) -> ImportResult:
        """Import a CSV file into benchmark records."""
        start = time.monotonic()
        records: List[Dict[str, Any]] = []
        try:
            with open(path, newline="") as fh:
                reader = csv.DictReader(fh, delimiter=delimiter)
                for row in reader:
                    records.append(self._apply_schema_mapping(dict(row), schema_mapping))
        except OSError as exc:
            return ImportResult(success=False, errors=[f"Could not read CSV: {exc}"])

        metadata = BenchmarkMetadata(
            name=path.split("/")[-1],
            source="csv",
            source_identifier=path,
            record_count=len(records),
            schema_mapping=schema_mapping or {},
            tags=list(tags or []) or ["csv"],
            imported_by=self._imported_by,
        )
        return self._finalize(metadata, records, [], start)

    def import_json(
        self,
        path: str,
        *,
        schema_mapping: Optional[Dict[str, str]] = None,
        tags: Optional[List[str]] = None,
    ) -> ImportResult:
        """Import a JSON or JSONL file. JSON arrays-of-objects are flattened."""
        start = time.monotonic()
        records: List[Dict[str, Any]] = []
        try:
            with open(path) as fh:
                text = fh.read()
        except OSError as exc:
            return ImportResult(success=False, errors=[f"Could not read JSON: {exc}"])

        # Try JSONL only when the file has multiple non-empty lines;
        # a single-line file is treated as JSON below (so we don't misread
        # ``{"records": [...]}`` as a JSONL stream containing one wrapper).
        non_empty_lines = [line for line in text.splitlines() if line.strip()]
        if len(non_empty_lines) > 1:
            try:
                jsonl = [json.loads(line) for line in non_empty_lines]
                if all(isinstance(r, dict) for r in jsonl):
                    records = [self._apply_schema_mapping(r, schema_mapping) for r in jsonl]
            except json.JSONDecodeError:
                records = []

        if not records:
            try:
                blob = json.loads(text)
            except json.JSONDecodeError as exc:
                return ImportResult(success=False, errors=[f"Invalid JSON: {exc}"])
            if isinstance(blob, list):
                records = [self._apply_schema_mapping(r, schema_mapping) for r in blob if isinstance(r, dict)]
            elif isinstance(blob, dict) and isinstance(blob.get("records"), list):
                records = [
                    self._apply_schema_mapping(r, schema_mapping) for r in blob["records"] if isinstance(r, dict)
                ]
            else:
                return ImportResult(
                    success=False,
                    errors=["JSON must be an array of objects, a JSONL stream, or {records: [...]}."],
                )

        metadata = BenchmarkMetadata(
            name=path.split("/")[-1],
            source="json",
            source_identifier=path,
            record_count=len(records),
            schema_mapping=schema_mapping or {},
            tags=list(tags or []) or ["json"],
            imported_by=self._imported_by,
        )
        return self._finalize(metadata, records, [], start)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_schema_mapping(record: Dict[str, Any], mapping: Optional[Dict[str, str]]) -> Dict[str, Any]:
        if not mapping:
            return record
        out = dict(record)
        for src, dst in mapping.items():
            if src in record and src != dst:
                out[dst] = record[src]
                out.pop(src, None)
        return out

    @staticmethod
    def _extract_helm_records(blob: Any) -> List[Dict[str, Any]]:
        """Tolerate the several shapes HELM result files come in."""
        if isinstance(blob, list):
            return [r for r in blob if isinstance(r, dict)]
        if isinstance(blob, dict):
            for key in ("instances", "predictions", "records", "data"):
                value = blob.get(key)
                if isinstance(value, list):
                    return [r for r in value if isinstance(r, dict)]
        return []

    def _finalize(
        self,
        metadata: BenchmarkMetadata,
        records: List[Dict[str, Any]],
        errors: List[str],
        start_monotonic: float,
    ) -> ImportResult:
        duration_ms = (time.monotonic() - start_monotonic) * 1000
        metadata.record_count = len(records)
        self._benchmarks[metadata.benchmark_id] = metadata
        return ImportResult(
            success=not errors,
            benchmark_id=metadata.benchmark_id,
            records_imported=len(records),
            records_skipped=0,
            duration_ms=round(duration_ms, 2),
            errors=errors,
            metadata=metadata,
            records=records,
        )

    @property
    def imported(self) -> Tuple[BenchmarkMetadata, ...]:
        return tuple(self._benchmarks.values())
