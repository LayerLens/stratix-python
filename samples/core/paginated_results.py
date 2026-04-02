"""
Paginated Results -- LayerLens Python SDK Sample
================================================

Demonstrates two approaches for fetching evaluation results:

  1. **Manual pagination** -- iterate page-by-page using
     ``client.results.get(evaluation=, page=, page_size=)``
     with full control over each request.

  2. **Automatic pagination** -- fetch all results at once using
     ``client.results.get_all(evaluation=)`` which handles
     pagination internally.

Prerequisites
-------------
* ``pip install layerlens --index-url https://sdk.layerlens.ai/package``
* Set ``LAYERLENS_STRATIX_API_KEY`` environment variable
* At least one completed evaluation in the project

Usage
-----
::

    export LAYERLENS_STRATIX_API_KEY=your-api-key
    python paginated_results.py
"""

from __future__ import annotations

from layerlens import Stratix


def main() -> None:
    client = Stratix()

    # ── Get a completed evaluation to work with ──────────────────────
    models = client.models.get()
    benchmarks = client.benchmarks.get()

    if not models or not benchmarks:
        print("No models or benchmarks available. Add them first.")
        return

    model = models[0]
    benchmark = benchmarks[0]

    print(f"Using model: {model.name}")
    print(f"Using benchmark: {benchmark.name}")

    # Create an evaluation and wait for it
    evaluation = client.evaluations.create(model=model, benchmark=benchmark)
    if not evaluation:
        print("Failed to create evaluation")
        return

    print(f"Created evaluation {evaluation.id}, waiting for completion...")
    evaluation = client.evaluations.wait_for_completion(
        evaluation,
        interval_seconds=10,
        timeout_seconds=600,
    )
    print(f"Evaluation {evaluation.id} finished with status={evaluation.status}")

    if not evaluation.is_success:
        print("Evaluation did not succeed, no results to show.")
        return

    # ── Approach 1: Manual page-by-page iteration ─────────────────────
    print("\n" + "=" * 60)
    print("MANUAL PAGINATION")
    print("=" * 60)

    all_results = []
    page = 1
    page_size = 50

    while True:
        print(f"Fetching page {page} (page_size={page_size})...")
        results_data = client.results.get(
            evaluation=evaluation,
            page=page,
            page_size=page_size,
        )

        if not results_data or not results_data.results:
            print("No more results to fetch")
            break

        all_results.extend(results_data.results)

        # Show progress on first page
        if page == 1:
            total_count = results_data.pagination.total_count
            total_pages = results_data.pagination.total_pages
            print(f"Total results: {total_count:,}")
            print(f"Total pages: {total_pages}")

        print(f"Page {page}: Retrieved {len(results_data.results)} results (running total: {len(all_results):,})")

        # Check if we have reached the last page
        if page >= results_data.pagination.total_pages:
            print("Reached last page")
            break

        page += 1

    print(f"\nManual pagination complete: {len(all_results):,} results collected")

    if all_results:
        correct = sum(1 for r in all_results if r.score > 0.5)
        accuracy = correct / len(all_results)
        avg_score = sum(r.score for r in all_results) / len(all_results)
        print(f"Accuracy: {accuracy:.1%} ({correct:,}/{len(all_results):,})")
        print(f"Average score: {avg_score:.3f}")

        print(f"\nFirst 3 results:")
        for i, result in enumerate(all_results[:3], 1):
            print(f"  {i}. Score: {result.score:.3f}, Subset: {result.subset}")
            print(f"     Prompt: {result.prompt[:100]}...")
            print(f"     Response: {result.result[:100]}...")

    # ── Alternative: get_by_id (using evaluation_id instead of object) ─
    print("\n" + "=" * 60)
    print("ALTERNATIVE: results.get_by_id(evaluation_id=...)")
    print("=" * 60)

    try:
        results_data = client.results.get_by_id(
            evaluation_id=evaluation.id,
            page=1,
            page_size=10,
        )
        if results_data and results_data.results:
            print(
                f"get_by_id returned {len(results_data.results)} results (total: {results_data.pagination.total_count})"
            )
        else:
            print("get_by_id returned no results")
    except Exception as exc:
        print(f"results.get_by_id() not available: {exc}")

    # ── Approach 2: Automatic get_all ─────────────────────────────────
    print("\n" + "=" * 60)
    print("AUTOMATIC PAGINATION (get_all)")
    print("=" * 60)

    all_results_auto = client.results.get_all(evaluation=evaluation)
    print(f"Retrieved {len(all_results_auto)} results in one call")

    if all_results_auto:
        avg_score = sum(r.score for r in all_results_auto) / len(all_results_auto)
        print(f"Average score: {avg_score:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
