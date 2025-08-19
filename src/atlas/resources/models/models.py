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
        companies: Optional[List[str]] = None,
        regions: Optional[List[str]] = None,
        licenses: Optional[List[str]] = None,
    ) -> List[Model] | None:
        base_url = f"/organizations/{self._client.organization_id}/projects/{self._client.project_id}/models"

        def fetch(model_type: str) -> ModelsResponse | None:
            params = {"type": model_type}
            if name:
                params["query"] = name
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


class AsyncModels(AsyncAPIResource):
    async def get(
        self,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
        type: Literal["custom", "public"] | None = None,
        name: Optional[str] = None,
        companies: Optional[List[str]] = None,
        regions: Optional[List[str]] = None,
        licenses: Optional[List[str]] = None,
    ) -> List[Model] | None:
        base_url = f"/organizations/{self._client.organization_id}/projects/{self._client.project_id}/models"

        async def fetch(model_type: str) -> ModelsResponse | None:
            params = {"type": model_type}
            if name:
                params["query"] = name
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
