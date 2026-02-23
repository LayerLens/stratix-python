#!/usr/bin/env -S poetry run python

import asyncio

from layerlens import AsyncStratix


async def main():
    # Construct async client
    client = AsyncStratix()

    # --- Get evaluation by id
    evaluation_id = "eval_123"
    evaluation = await client.evaluations.get(evaluation_id)
    print(f"Found evaluation {evaluation.id}")
    print(evaluation)


if __name__ == "__main__":
    asyncio.run(main())
