#!/usr/bin/env -S poetry run python

import asyncio

from layerlens import AsyncStratix


async def main():
    # Construct async client
    client = AsyncStratix()

    # --- Get models by name
    model_name = "gpt-4o"
    models = await client.models.get(name=model_name)
    print(f"Found {len(models)} models with name {model_name}")
    print(models)

    # --- Get models by company
    company_names = ["openai", "anthropic"]
    models = await client.models.get(companies=company_names)
    print(f"Found {len(models)} models with companies {company_names}")
    print(models)

    # --- Get models by region
    region_names = ["usa"]
    models = await client.models.get(regions=region_names)
    print(f"Found {len(models)} models with regions {region_names}")
    print(models)

    # --- Get models by type
    model_type = "public"
    models = await client.models.get(type=model_type)
    print(f"Found {len(models)} models with type {model_type}")
    print(models)


if __name__ == "__main__":
    asyncio.run(main())
