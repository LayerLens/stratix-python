from __future__ import annotations

import logging
from typing import Any, Dict, List, Union, Optional

import httpx

from ..._resource import SyncAPIResource, AsyncAPIResource
from ..._constants import DEFAULT_TIMEOUT

log: logging.Logger = logging.getLogger(__name__)


def _unwrap(resp: Any) -> Any:
    if isinstance(resp, dict) and "data" in resp and "status" in resp:
        return resp["data"]
    return resp


class SigningKeys(SyncAPIResource):
    def _base_url(self) -> str:
        org_id = self._client.organization_id
        if not org_id:
            raise ValueError("Client has no organization_id configured")
        return f"/organizations/{org_id}/signing-keys"

    def get_active(
        self,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[Dict[str, Any]]:
        """Fetch the active signing key (key_id, name, secret).

        Returns None if no active signing key exists (404).
        """
        try:
            resp = self._get(
                f"{self._base_url()}/active",
                timeout=timeout,
                cast_to=dict,
            )
            data = _unwrap(resp)
            return data if isinstance(data, dict) else None
        except Exception:
            log.debug("No active signing key found", exc_info=True)
            return None

    def create(
        self,
        *,
        name: str = "default",
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[Dict[str, Any]]:
        """Create a new signing key for the organization.

        Returns the key data (key_id, name, secret) or None on failure.
        """
        try:
            resp = self._post(
                self._base_url(),
                body={"name": name},
                timeout=timeout,
                cast_to=dict,
            )
            data = _unwrap(resp)
            return data if isinstance(data, dict) else None
        except Exception:
            log.debug("Failed to create signing key", exc_info=True)
            return None

    def list(
        self,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[Union[List[Dict[str, Any]], Dict[str, Any]]]:
        """List signing key metadata (no secrets)."""
        try:
            resp = self._get(self._base_url(), timeout=timeout, cast_to=dict)
            data = _unwrap(resp)
            if isinstance(data, (dict, list)):
                return data
            return None
        except Exception:
            log.debug("Failed to list signing keys", exc_info=True)
            return None


class AsyncSigningKeys(AsyncAPIResource):
    def _base_url(self) -> str:
        org_id = self._client.organization_id
        if not org_id:
            raise ValueError("Client has no organization_id configured")
        return f"/organizations/{org_id}/signing-keys"

    async def get_active(
        self,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[Dict[str, Any]]:
        """Fetch the active signing key (key_id, name, secret).

        Returns None if no active signing key exists (404).
        """
        try:
            resp = await self._get(
                f"{self._base_url()}/active",
                timeout=timeout,
                cast_to=dict,
            )
            data = _unwrap(resp)
            return data if isinstance(data, dict) else None
        except Exception:
            log.debug("No active signing key found", exc_info=True)
            return None

    async def create(
        self,
        *,
        name: str = "default",
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[Dict[str, Any]]:
        """Create a new signing key for the organization.

        Returns the key data (key_id, name, secret) or None on failure.
        """
        try:
            resp = await self._post(
                self._base_url(),
                body={"name": name},
                timeout=timeout,
                cast_to=dict,
            )
            data = _unwrap(resp)
            return data if isinstance(data, dict) else None
        except Exception:
            log.debug("Failed to create signing key", exc_info=True)
            return None

    async def list(
        self,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[Union[List[Dict[str, Any]], Dict[str, Any]]]:
        """List signing key metadata (no secrets)."""
        try:
            resp = await self._get(self._base_url(), timeout=timeout, cast_to=dict)
            data = _unwrap(resp)
            if isinstance(data, (dict, list)):
                return data
            return None
        except Exception:
            log.debug("Failed to list signing keys", exc_info=True)
            return None
