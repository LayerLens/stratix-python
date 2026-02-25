#!/usr/bin/env -S poetry run python

from layerlens import PublicClient


def main():
    # Construct public client (API key from env or inline)
    client = PublicClient()

    # --- Browse all public benchmarks
    response = client.benchmarks.get(page=1, page_size=10)
    print(f"Found {response.total_count} public benchmarks (showing first {len(response.datasets)})")
    for benchmark in response.datasets:
        print(f"  - {benchmark.name} (prompts={benchmark.prompt_count}, language={benchmark.language})")

    # --- Filter by language
    response = client.benchmarks.get(languages=["English"])
    print(f"\nFound {response.total_count} English benchmarks")

    # --- Filter by category
    if response.categories:
        print(f"\nAvailable categories: {response.categories}")

    # --- Search by name
    response = client.benchmarks.get(query="mmlu")
    print(f"\nFound {response.total_count} benchmarks matching 'mmlu'")
    for benchmark in response.datasets:
        print(f"  - {benchmark.name}: {benchmark.description[:80] if benchmark.description else 'N/A'}...")

    # --- Get benchmark prompts (content download)
    if response.datasets:
        benchmark = response.datasets[0]
        print(f"\nFetching prompts for '{benchmark.name}' (id={benchmark.id})...")

        prompts_response = client.benchmarks.get_prompts(
            benchmark.id,
            page=1,
            page_size=5,
        )

        if prompts_response:
            print(f"Total prompts: {prompts_response.data.count}")
            print(f"Showing first {len(prompts_response.data.prompts)} prompts:")
            for prompt in prompts_response.data.prompts:
                input_preview = str(prompt.input)[:80]
                truth_preview = prompt.truth[:50] if prompt.truth else "N/A"
                print(f"  - Input: {input_preview}...")
                print(f"    Truth: {truth_preview}")
                print()

    # --- Get all prompts (auto-paginates)
    if response.datasets:
        benchmark = response.datasets[0]
        print(f"Fetching ALL prompts for '{benchmark.name}'...")
        all_prompts = client.benchmarks.get_all_prompts(benchmark.id)
        print(f"Retrieved {len(all_prompts)} total prompts")


if __name__ == "__main__":
    main()
