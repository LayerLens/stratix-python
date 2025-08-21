#!/usr/bin/env -S poetry run python

import asyncio

from atlas import AsyncAtlas


async def main():
    # Construct async client
    client = AsyncAtlas()

    # --- Models
    models = await client.models.get()
    print(f"Found {len(models)} models")

    # --- Benchmarks
    benchmarks = await client.benchmarks.get()
    print(f"Found {len(benchmarks)} benchmarks")
    
    # --- Create evaluation
    evaluation = await client.evaluations.create(
        model=models[0],
        benchmark=benchmarks[0],
    )
    print(f"Created evaluation {evaluation.id}, status={evaluation.status}")

    # --- Wait for completion
    evaluation = await client.evaluations.wait_for_completion(
        evaluation,
        interval_seconds=10,
        # Keep in mind that the evaluation will take a while to complete, so you may want to increase the timeout
        # or grab the evaluation id and check the status later
        timeout_seconds=600,  # 10 minutes
    )
    print(f"Evaluation {evaluation.id} finished with status={evaluation.status}")

    # --- Results with pagination
    if evaluation.is_success:
        print("Fetching all results with pagination...")
        
        all_results = []
        page = 1
        page_size = 50
        
        while True:
            print(f"Fetching page {page} (page size: {page_size})...")
            
            # Get results for current page
            results_data = await client.results.get_by_id(
                evaluation_id=evaluation.id,
                page=page,
                page_size=page_size
            )
            
            if not results_data or not results_data.results:
                print("No more results to fetch")
                break
            
            # Add current page results to our collection
            all_results.extend(results_data.results)
            
            # Show progress
            if page == 1:
                total_count = results_data.pagination.total_count
                total_pages = results_data.pagination.total_pages
                print(f"Total results: {total_count:,}")
                print(f"Total pages: {total_pages}")
            
            print(f"Page {page}: Retrieved {len(results_data.results)} results")
            print(f"Running total: {len(all_results):,} results")
            
            # Check if we've reached the last page
            if page >= results_data.pagination.total_pages:
                print("Reached last page")
                break
            
            page += 1
        
        # Summary of all results
        print(f"\n=== PAGINATION COMPLETE ===")
        print(f"Total results collected: {len(all_results):,}")
        
        if all_results:
            # Calculate some basic statistics
            correct_answers = sum(1 for r in all_results if r.score > 0.5)
            accuracy = correct_answers / len(all_results)
            avg_score = sum(r.score for r in all_results) / len(all_results)
            
            print(f"Overall accuracy: {accuracy:.1%} ({correct_answers:,}/{len(all_results):,})")
            print(f"Average score: {avg_score:.3f}")
            
            # Show a few example results
            print(f"\nFirst 3 results:")
            for i, result in enumerate(all_results[:3], 1):
                print(f"  {i}. Score: {result.score:.3f}, Subset: {result.subset}")
                print(f"     Prompt: {result.prompt[:100]}...")
                print(f"     Response: {result.result[:100]}...")
                print()
        
    else:
        print("Evaluation did not succeed, no results to show.")


if __name__ == "__main__":
    asyncio.run(main())
