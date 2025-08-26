#!/usr/bin/env -S poetry run python

import asyncio

from atlas import AsyncAtlas


async def main():
    # Construct async client
    client = AsyncAtlas()

    # --- Models
    models = await client.models.get(type="public",name="gpt-4o")
    print(f"Models found: {models}")

    if not models:
        print("gpt-4o not found, exiting")
        return

    model = models[0]
    # --- Benchmarks
    benchmarks = await client.benchmarks.get(type="public", name="simpleQA")

    if not benchmarks:
        print("SimpleQA benchmark not found, exiting")
        return

    benchmark = benchmarks[0]

    # --- Create evaluation
    evaluation = await client.evaluations.create(
        model=model,
        benchmark=benchmark,
    )


if __name__ == "__main__":
    asyncio.run(main())
