#!/usr/bin/env -S poetry run python

from layerlens import Stratix
from layerlens.models import EvaluationStatus


def main():
    # Construct client (API key from env or inline)
    client = Stratix()

    # --- Get successful evaluations to find a comparable pair
    response = client.evaluations.get_many(
        status=EvaluationStatus.SUCCESS,
        sort_by="accuracy",
        order="desc",
        page_size=100,
    )

    if not response or len(response.evaluations) < 2:
        print("Need at least 2 successful evaluations to compare, exiting")
        return

    # Find two evaluations on the same benchmark
    eval_1 = None
    eval_2 = None
    for i, e1 in enumerate(response.evaluations):
        for e2 in response.evaluations[i + 1 :]:
            if e1.benchmark_id == e2.benchmark_id and e1.id != e2.id:
                eval_1 = e1
                eval_2 = e2
                break
        if eval_1:
            break

    if not eval_1 or not eval_2:
        print("No two evaluations share the same benchmark, exiting")
        return

    print(f"Comparing evaluations on the same benchmark ({eval_1.benchmark_id}):")
    print(f"  Evaluation 1: {eval_1.id} (accuracy={eval_1.accuracy:.2f}%)")
    print(f"  Evaluation 2: {eval_2.id} (accuracy={eval_2.accuracy:.2f}%)")

    # --- Get comparison results
    comparison = client.public.comparisons.compare(
        evaluation_id_1=eval_1.id,
        evaluation_id_2=eval_2.id,
        page=1,
        page_size=10,
    )

    if comparison:
        print(f"\n=== Comparison Summary ===")
        print(f"Evaluation 1: {comparison.correct_count_1}/{comparison.total_results_1} correct")
        print(f"Evaluation 2: {comparison.correct_count_2}/{comparison.total_results_2} correct")
        print(f"Total compared: {comparison.total_count}")

        # --- Show individual results
        if comparison.results:
            print(f"\nFirst {len(comparison.results)} results:")
            for result in comparison.results:
                score_indicator_1 = "✓" if result.score1 and result.score1 > 0.5 else "✗"
                score_indicator_2 = "✓" if result.score2 and result.score2 > 0.5 else "✗"
                print(f"  Prompt: {result.prompt[:80]}...")
                print(f"    Model 1: {score_indicator_1} (score={result.score1})")
                print(f"    Model 2: {score_indicator_2} (score={result.score2})")
                print()

    # --- Filter by outcome: where only model 1 fails
    comparison = client.public.comparisons.compare(
        evaluation_id_1=eval_1.id,
        evaluation_id_2=eval_2.id,
        outcome_filter="reference_fails",
    )

    if comparison:
        print(f"\n=== Where Model 1 Fails but Model 2 Succeeds ===")
        print(f"Found {comparison.total_count} such cases")

    # --- Filter by outcome: where both models fail
    comparison = client.public.comparisons.compare(
        evaluation_id_1=eval_1.id,
        evaluation_id_2=eval_2.id,
        outcome_filter="both_fail",
    )

    if comparison:
        print(f"\n=== Where Both Models Fail ===")
        print(f"Found {comparison.total_count} such cases")


if __name__ == "__main__":
    main()
