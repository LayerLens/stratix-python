"""
Evaluation Filtering & Sorting -- LayerLens Python SDK Sample
=============================================================

Demonstrates the full range of filtering, sorting, and pagination
options available on ``client.evaluations.get_many()``:

  - Sort by accuracy, submission date, or average duration.
  - Filter by status, model IDs, or benchmark IDs.
  - Combine filters with pagination.

Prerequisites
-------------
* ``pip install layerlens --index-url https://sdk.layerlens.ai/package``
* Set ``LAYERLENS_STRATIX_API_KEY`` environment variable
* At least one completed evaluation in the project

Usage
-----
::

    export LAYERLENS_STRATIX_API_KEY=your-api-key
    python evaluation_filtering.py
"""

from __future__ import annotations

from layerlens import Stratix
from layerlens.models import EvaluationStatus


def main() -> None:
    client = Stratix()

    # ── Sort by accuracy (highest first) ──────────────────────────────
    response = client.evaluations.get_many(
        sort_by="accuracy",
        order="desc",
        page_size=10,
    )
    if response:
        print(f"Top {len(response.evaluations)} evaluations by accuracy:")
        for e in response.evaluations:
            print(f"  - {e.id}: accuracy={e.accuracy:.2f}%, status={e.status.value}")

    # ── Sort by submission date (newest first) ────────────────────────
    response = client.evaluations.get_many(
        sort_by="submittedAt",
        order="desc",
        page_size=5,
    )
    if response:
        print(f"\nLatest {len(response.evaluations)} evaluations:")
        for e in response.evaluations:
            print(f"  - {e.id}: submitted_at={e.submitted_at}")

    # ── Sort by average duration (fastest first) ──────────────────────
    response = client.evaluations.get_many(
        sort_by="averageDuration",
        order="asc",
        page_size=5,
    )
    if response:
        print(f"\nFastest {len(response.evaluations)} evaluations:")
        for e in response.evaluations:
            print(f"  - {e.id}: avg_duration={e.average_duration}ms")

    # ── Filter by status (only successful) ────────────────────────────
    response = client.evaluations.get_many(
        status=EvaluationStatus.SUCCESS,
        sort_by="accuracy",
        order="desc",
    )
    if response:
        print(f"\nSuccessful evaluations: {response.pagination.total_count}")

    # ── Filter by specific model IDs ──────────────────────────────────
    # Replace with actual model IDs from your project
    response = client.evaluations.get_many(
        model_ids=["your-model-id"],
        sort_by="accuracy",
        order="desc",
    )
    if response:
        print(f"\nEvaluations for specified model: {response.pagination.total_count}")

    # ── Filter by specific benchmark IDs ──────────────────────────────
    # Replace with actual benchmark IDs from your project
    response = client.evaluations.get_many(
        benchmark_ids=["your-benchmark-id"],
        sort_by="submittedAt",
        order="desc",
    )
    if response:
        print(f"\nEvaluations for specified benchmark: {response.pagination.total_count}")

    # ── Combine sorting, filtering, and pagination ────────────────────
    response = client.evaluations.get_many(
        status=EvaluationStatus.SUCCESS,
        sort_by="accuracy",
        order="desc",
        page=1,
        page_size=20,
    )
    if response:
        print(f"\nPage 1 of successful evaluations (sorted by accuracy):")
        print(f"  Total: {response.pagination.total_count}")
        print(f"  Pages: {response.pagination.total_pages}")
        for e in response.evaluations:
            print(f"  - {e.id}: accuracy={e.accuracy:.2f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
