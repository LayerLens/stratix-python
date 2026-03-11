#!/usr/bin/env python3

import asyncio

from layerlens import AsyncStratix
from layerlens.models import EvaluationStatus


async def main():
    # Construct async client (requires API key)
    client = AsyncStratix()

    # --- Get evaluations sorted by accuracy (highest first)
    response = await client.evaluations.get_many(
        sort_by="accuracy",
        order="desc",
        page_size=10,
    )
    if response:
        print(f"Top {len(response.evaluations)} evaluations by accuracy:")
        for evaluation in response.evaluations:
            print(f"  - {evaluation.id}: accuracy={evaluation.accuracy:.2f}%, status={evaluation.status.value}")

    # --- Get evaluations sorted by submission date (newest first)
    response = await client.evaluations.get_many(
        sort_by="submittedAt",
        order="desc",
        page_size=5,
    )
    if response:
        print(f"\nLatest {len(response.evaluations)} evaluations:")
        for evaluation in response.evaluations:
            print(f"  - {evaluation.id}: submitted_at={evaluation.submitted_at}")

    # --- Get evaluations sorted by average duration (fastest first)
    response = await client.evaluations.get_many(
        sort_by="averageDuration",
        order="asc",
        page_size=5,
    )
    if response:
        print(f"\nFastest {len(response.evaluations)} evaluations:")
        for evaluation in response.evaluations:
            print(f"  - {evaluation.id}: avg_duration={evaluation.average_duration}ms")

    # --- Filter by status (only successful evaluations)
    response = await client.evaluations.get_many(
        status=EvaluationStatus.SUCCESS,
        sort_by="accuracy",
        order="desc",
    )
    if response:
        print(f"\nSuccessful evaluations: {response.pagination.total_count}")

    # --- Filter by specific model IDs
    # Replace with actual model IDs from your organization
    response = await client.evaluations.get_many(
        model_ids=["your-model-id"],
        sort_by="accuracy",
        order="desc",
    )
    if response:
        print(f"\nEvaluations for specified model: {response.pagination.total_count}")

    # --- Filter by specific benchmark IDs
    # Replace with actual benchmark IDs from your organization
    response = await client.evaluations.get_many(
        benchmark_ids=["your-benchmark-id"],
        sort_by="submittedAt",
        order="desc",
    )
    if response:
        print(f"\nEvaluations for specified benchmark: {response.pagination.total_count}")

    # --- Combine sorting, filtering, and pagination
    response = await client.evaluations.get_many(
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
        for evaluation in response.evaluations:
            print(f"  - {evaluation.id}: accuracy={evaluation.accuracy:.2f}%")


if __name__ == "__main__":
    asyncio.run(main())
