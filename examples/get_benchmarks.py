#!/usr/bin/env python3

import asyncio

from layerlens import AsyncStratix


async def main():
    # Construct async client
    client = AsyncStratix()

    # --- Get benchmarks by name
    benchmark_name = "mmlu"
    benchmarks = await client.benchmarks.get(name=benchmark_name)
    print(f"Found {len(benchmarks)} benchmarks with name {benchmark_name}")
    print(benchmarks)

    # --- Get benchmarks by type
    benchmark_type = "public"
    benchmarks = await client.benchmarks.get(type=benchmark_type)
    print(f"Found {len(benchmarks)} benchmarks with type {benchmark_type}")
    print(benchmarks)


if __name__ == "__main__":
    asyncio.run(main())
