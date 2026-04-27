"""Sample: import a tiny CSV benchmark with the LayerLens adapter.

Writes a small CSV to a tempfile, then runs ``BenchmarkImportAdapter.import_csv``
and prints the resulting ``ImportResult``. This adapter is a data importer
(not a runtime trace adapter) so it does not require any LLM credentials.

Run::

    pip install 'layerlens[benchmark-import]'
    python -m samples.instrument.benchmark_import.main
"""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

from layerlens.instrument.adapters.frameworks.benchmark_import import (
    BenchmarkImportAdapter,
)


def _write_sample_csv(path: Path) -> None:
    rows = [
        {"question": "What is 2 + 2?", "answer": "4", "category": "math"},
        {"question": "Capital of France?", "answer": "Paris", "category": "geo"},
        {"question": "Largest planet?", "answer": "Jupiter", "category": "science"},
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer", "category"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    adapter = BenchmarkImportAdapter()

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "sample_benchmark.csv"
        _write_sample_csv(csv_path)

        result = adapter.import_csv(
            path=str(csv_path),
            schema_mapping={
                "question": "prompt",
                "answer": "expected_output",
                "category": "category",
            },
            tags=["sample", "qa"],
        )

        if not result.success:
            print(f"Import failed: {result.errors}", file=sys.stderr)
            return 1

        print(f"Benchmark id: {result.benchmark_id}")
        print(f"Records imported: {result.records_imported}")
        print(f"Duration: {result.duration_ms:.2f} ms")
        if result.metadata is not None:
            print(f"Tags: {', '.join(result.metadata.tags)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
