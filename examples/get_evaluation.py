#!/usr/bin/env python3

import asyncio

from layerlens import AsyncStratix


async def main():
    # Construct async client
    client = AsyncStratix()

    # --- Get evaluation by id
    evaluation_id = "699f1426c1212b2d9c78e947"
    evaluation = await client.evaluations.get_by_id(evaluation_id)
    print(f"Found evaluation {evaluation.id}")
    print(evaluation)


if __name__ == "__main__":
    asyncio.run(main())
