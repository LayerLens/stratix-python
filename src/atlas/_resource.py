from __future__ import annotations

import time
import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._client import Atlas, AsyncAtlas


class SyncAPIResource:
    _client: Atlas

    def __init__(self, client: Atlas) -> None:
        self._client = client
        self._get = client.get_cast
        self._post = client.post_cast

    def _sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class AsyncAPIResource:
    _client: AsyncAtlas

    def __init__(self, client: AsyncAtlas) -> None:
        self._client = client
        self._get = client.get_cast
        self._post = client.post_cast

    async def _sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)
