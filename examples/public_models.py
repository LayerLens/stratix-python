#!/usr/bin/env -S poetry run python

from layerlens import PublicClient


def main():
    # Construct public client (API key from env or inline)
    client = PublicClient()

    # --- Browse all public models (first page)
    response = client.models.get(page=1, page_size=10)
    print(f"Found {response.total_count} public models (showing first {len(response.models)})")
    for model in response.models:
        print(f"  - {model.name} ({model.company})")

    # --- Search models by query
    response = client.models.get(query="gpt")
    print(f"\nFound {response.total_count} models matching 'gpt'")
    for model in response.models:
        print(f"  - {model.name}")

    # --- Filter by company
    companies = ["OpenAI", "Anthropic"]
    response = client.models.get(companies=companies)
    print(f"\nFound {response.total_count} models from {companies}")
    for model in response.models:
        print(f"  - {model.name} ({model.company})")

    # --- Filter by region
    response = client.models.get(regions=["usa"])
    print(f"\nFound {response.total_count} models in region 'usa'")

    # --- Filter by category
    response = client.models.get(categories=["open-source"])
    print(f"\nFound {response.total_count} open-source models")

    # --- Sort by release date (newest first)
    response = client.models.get(sort_by="releasedAt", order="desc", page_size=5)
    print(f"\nNewest 5 models:")
    for model in response.models:
        print(f"  - {model.name} (released_at={model.released_at})")

    # --- Include deprecated models
    response = client.models.get(include_deprecated=True)
    print(f"\nTotal models (including deprecated): {response.total_count}")

    # --- Discover available filter values
    response = client.models.get(page=1, page_size=1)
    print(f"\nAvailable filter values:")
    print(f"  Categories: {response.categories}")
    print(f"  Companies:  {response.companies}")
    print(f"  Regions:    {response.regions}")
    print(f"  Licenses:   {response.licenses}")
    print(f"  Sizes:      {response.sizes}")


if __name__ == "__main__":
    main()
