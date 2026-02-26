from __future__ import annotations

import time
import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._client import Stratix, AsyncStratix
    from ._public_client import PublicClient, AsyncPublicClient


class SyncAPIResource:
    _client: Stratix

    def __init__(self, client: Stratix) -> None:
        self._client = client
        self._get = client.get_cast
        self._post = client.post_cast
        self._patch = client.patch_cast
        self._delete = client.delete_cast

    def _sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class AsyncAPIResource:
    _client: AsyncStratix

    def __init__(self, client: AsyncStratix) -> None:
        self._client = client
        self._get = client.get_cast
        self._post = client.post_cast
        self._patch = client.patch_cast
        self._delete = client.delete_cast

    async def _sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class SyncPublicAPIResource:
    _client: PublicClient

    def __init__(self, client: PublicClient) -> None:
        self._client = client
        self._get = client.get_cast

    def _sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class AsyncPublicAPIResource:
    _client: AsyncPublicClient

    def __init__(self, client: AsyncPublicClient) -> None:
        self._client = client
        self._get = client.get_cast

    async def _sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)
