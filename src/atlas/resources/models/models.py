from __future__ import annotations

from typing import List, Literal, Optional

import httpx

from ...models import Model, CustomModel, PublicModel, ModelsResponse
from ..._resource import SyncAPIResource, AsyncAPIResource
from ..._constants import DEFAULT_TIMEOUT


class Models(SyncAPIResource):
    def get(
        self,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
        type: Literal["custom", "public"] | None = None,
        name: Optional[str] = None,
        key: Optional[str] = None,
        companies: Optional[List[str]] = None,
        regions: Optional[List[str]] = None,
        licenses: Optional[List[str]] = None,
    ) -> Optional[List[Model]]:
        base_url = f"/organizations/{self._client.organization_id}/projects/{self._client.project_id}/models"

        def fetch(model_type: str) -> ModelsResponse | None:
            params = {"type": model_type}
            if name:
                params["name"] = name
            if key:
                params["key"] = key
            if companies:
                params["companies"] = ",".join(companies)
            if regions:
                params["regions"] = ",".join(regions)
            if licenses:
                params["licenses"] = ",".join(licenses)

            resp = self._get(
                base_url,
                params=params,
                timeout=timeout,
                cast_to=ModelsResponse,
            )
            return resp if isinstance(resp, ModelsResponse) else None

        models: List[Model] = []

        def cast_model(m: Model, model_type: str) -> Model:
            if model_type == "custom":
                return CustomModel(**m.model_dump())
            elif model_type == "public":
                return PublicModel(**m.model_dump())
            return m  # fallback, just base class

        if type is None:  # fetch both
            for t in ["custom", "public"]:
                resp = fetch(t)
                if resp:
                    models.extend([cast_model(m, t) for m in resp.data.models])
        else:  # fetch only one type
            resp = fetch(type)
            if resp:
                models.extend([cast_model(m, type) for m in resp.data.models])

        return models

    def get_by_id(self, id: str, *, timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT) -> Optional[Model]:
        base_url = f"/organizations/{self._client.organization_id}/projects/{self._client.project_id}/models/{id}"

        resp = self._get(
            base_url,
            timeout=timeout,
            cast_to=dict,
        )

        if not isinstance(resp, dict):
            return None

        model = resp.get("data")
        if not isinstance(model, dict):
            return None

        # Detect type dynamically: presence of "organization_id" means custom
        if "organization_id" in model:
            return CustomModel(**model)
        else:
            return PublicModel(**model)

    def get_by_key(
        self,
        key: str,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[Model]:
        """Fetch a single model by its unique key."""
        models = self.get(timeout=timeout, key=key)

        if not models:
            return None

        for model in models:
            if model.key == key:
                return model
        return None


class AsyncModels(AsyncAPIResource):
    async def get(
        self,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
        type: Literal["custom", "public"] | None = None,
        name: Optional[str] = None,
        key: Optional[str] = None,
        companies: Optional[List[str]] = None,
        regions: Optional[List[str]] = None,
        licenses: Optional[List[str]] = None,
    ) -> Optional[List[Model]]:
        base_url = f"/organizations/{self._client.organization_id}/projects/{self._client.project_id}/models"

        async def fetch(model_type: str) -> ModelsResponse | None:
            params = {"type": model_type}
            if name:
                params["name"] = name
            if key:
                params["key"] = key
            if companies:
                params["companies"] = ",".join(companies)
            if regions:
                params["regions"] = ",".join(regions)
            if licenses:
                params["licenses"] = ",".join(licenses)

            resp = await self._get(
                base_url,
                params=params,
                timeout=timeout,
                cast_to=ModelsResponse,
            )
            return resp if isinstance(resp, ModelsResponse) else None

        models: List[Model] = []

        def cast_model(m: Model, model_type: str) -> Model:
            if model_type == "custom":
                return CustomModel(**m.model_dump())
            elif model_type == "public":
                return PublicModel(**m.model_dump())
            return m  # fallback, just base class

        if type is None:  # fetch both
            for t in ["custom", "public"]:
                resp = await fetch(t)
                if resp:
                    models.extend([cast_model(m, t) for m in resp.data.models])
        else:  # fetch only one type
            resp = await fetch(type)
            if resp:
                models.extend([cast_model(m, type) for m in resp.data.models])

        return models

    async def get_by_id(self, id: str, *, timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT) -> Optional[Model]:
        base_url = f"/organizations/{self._client.organization_id}/projects/{self._client.project_id}/models/{id}"

        resp = await self._get(
            base_url,
            timeout=timeout,
            cast_to=dict,
        )

        if not isinstance(resp, dict):
            return None

        model = resp.get("data")
        if not isinstance(model, dict):
            return None

        # Detect type dynamically: presence of "organization_id" means custom
        if "organization_id" in model:
            return CustomModel(**model)
        else:
            return PublicModel(**model)

    async def get_by_key(
        self,
        key: str,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[Model]:
        """Fetch a single model by its unique key."""
        models = await self.get(timeout=timeout, key=key)

        if not models:
            return None

        for model in models:
            if model.key == key:
                return model
        return None
