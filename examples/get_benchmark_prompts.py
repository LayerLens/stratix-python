#!/usr/bin/env python3
"""Fetch prompts from a benchmark (custom or public)."""

from layerlens import Stratix


def main():
    client = Stratix()

    # Find a benchmark with prompts
    benchmarks = client.benchmarks.get()
    benchmark = next((b for b in benchmarks if b.prompt_count and b.prompt_count > 0), None)
    if benchmark is None:
        print("No benchmarks with prompts found.")
        return

    print(f"Benchmark: {benchmark.name} ({benchmark.key})")
    print(f"Total prompts: {benchmark.prompt_count}\n")

    # --- Get a single page of prompts
    page = client.benchmarks.get_prompts(benchmark.id, page=1, page_size=5)
    if page:
        print(f"Page 1 ({len(page.prompts)} of {page.count}):")
        for p in page.prompts:
            inp = str(p.input)[:80]
            print(f"  [{p.id}] {inp}")

    # --- Get all prompts (auto-paginated)
    all_prompts = client.benchmarks.get_all_prompts(benchmark.id)
    print(f"\nAll prompts fetched: {len(all_prompts)}")

    # --- Search and sort
    results = client.benchmarks.get_prompts(
        benchmark.id,
        search_field="truth",
        search_value="the",
        sort_by="id",
        sort_order="asc",
        page_size=3,
    )
    if results:
        print(f"\nSearch results ({results.count} matches):")
        for p in results.prompts:
            print(f"  [{p.id}] truth: {p.truth[:60]}")


if __name__ == "__main__":
    main()
