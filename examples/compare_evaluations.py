#!/usr/bin/env python3

from layerlens import PublicClient


def main():
    # Construct public client (API key from LAYERLENS_STRATIX_API_KEY env var or inline)
    client = PublicClient()

    # --- Compare two models on a benchmark using compare_models
    # Just provide the benchmark and two model IDs - the SDK automatically
    # finds the most recent successful evaluation for each model.
    benchmark_id = "682bddc1e014f9fa440f8a91"  # AIME 2025
    model_id_1 = "699f9761e014f9c3072b0513"  # Qwen3.5 27B
    model_id_2 = "699f9761e014f9c3072b0512"  # Qwen3.5 122B A10B

    print(f"Comparing models on benchmark {benchmark_id}...")
    comparison = client.comparisons.compare_models(
        benchmark_id=benchmark_id,
        model_id_1=model_id_1,
        model_id_2=model_id_2,
        page=1,
        page_size=10,
    )

    if comparison:
        print(f"\n=== Comparison Summary ===")
        print(f"Model 1: {comparison.correct_count_1}/{comparison.total_results_1} correct")
        print(f"Model 2: {comparison.correct_count_2}/{comparison.total_results_2} correct")
        print(f"Total compared: {comparison.total_count}")

        if comparison.results:
            print(f"\nFirst {len(comparison.results)} results:")
            for result in comparison.results:
                s1 = "Y" if result.score1 and result.score1 > 0.5 else "N"
                s2 = "Y" if result.score2 and result.score2 > 0.5 else "N"
                print(f"  Prompt: {result.prompt[:80]}...")
                print(f"    Model 1: {s1} (score={result.score1})")
                print(f"    Model 2: {s2} (score={result.score2})")
                print()

    # --- Filter: where model 1 fails but model 2 succeeds
    comparison = client.comparisons.compare_models(
        benchmark_id=benchmark_id,
        model_id_1=model_id_1,
        model_id_2=model_id_2,
        outcome_filter="reference_fails",
    )

    if comparison:
        print(f"\n=== Where Model 1 Fails but Model 2 Succeeds ===")
        print(f"Found {comparison.total_count} such cases")

    # --- You can also compare using evaluation IDs directly
    comparison = client.comparisons.compare(
        evaluation_id_1="699f9938a03d70bf6607081f",  # Qwen3.5 27B on AIME 2025
        evaluation_id_2="699f991ca782d00ebd666ba1",  # Qwen3.5 122B A10B on AIME 2025
        page=1,
        page_size=5,
    )

    if comparison:
        print(f"\n=== Direct Comparison by Evaluation IDs ===")
        print(f"Model 1: {comparison.correct_count_1}/{comparison.total_results_1} correct")
        print(f"Model 2: {comparison.correct_count_2}/{comparison.total_results_2} correct")


if __name__ == "__main__":
    main()
