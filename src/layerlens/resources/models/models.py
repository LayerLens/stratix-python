from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

import httpx

from ...models import Model, CustomModel, PublicModel, ModelsResponse, CreateModelResponse


def _exclude_custom_models(
    models: List[Model],
    *,
    categories: Optional[List[str]] = None,
    companies: Optional[List[str]] = None,
    regions: Optional[List[str]] = None,
    licenses: Optional[List[str]] = None,
) -> List[Model]:
    """Exclude custom models when filtering by fields they don't have.

    The API correctly filters public models and custom models by name/key,
    but custom models don't have categories/companies/regions/licenses fields,
    so they must be excluded from results when those filters are active.
    """
    if categories:
        cat_set = {c.lower() for c in categories}

        def matches_category(m: Model) -> bool:
            if not isinstance(m, PublicModel):
                return False
            arch = (m.architecture_type or "").lower()
            for cat in cat_set:
                if cat == "open-source" and m.open_weights:
                    return True
                if cat == "closed-source" and not m.open_weights and arch:
                    return True
                if arch and cat == arch:
                    return True
            return False

        models = [m for m in models if matches_category(m)]

    if companies:
        comp_set = {c.lower() for c in companies}
        models = [m for m in models if isinstance(m, PublicModel) and m.company and m.company.lower() in comp_set]

    if regions:
        reg_set = {r.lower() for r in regions}
        models = [m for m in models if isinstance(m, PublicModel) and m.region and m.region.lower() in reg_set]

    if licenses:
        lic_set = {l.lower() for l in licenses}
        models = [m for m in models if isinstance(m, PublicModel) and m.license and m.license.lower() in lic_set]

    return models


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
        categories: Optional[List[str]] = None,
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
            if categories:
                params["categories"] = ",".join(categories)
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

        models = _exclude_custom_models(
            models,
            categories=categories,
            companies=companies,
            regions=regions,
            licenses=licenses,
        )

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

    def add(
        self,
        *model_ids: str,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> bool:
        """Add models to the project by their IDs."""
        # Only fetch public (platform) models — custom models are managed
        # separately and must not be included in the project patch payload.
        current = self.get(timeout=timeout, type="public") or []
        current_ids = [str(m.id) for m in current]
        new_ids = list(dict.fromkeys(current_ids + list(model_ids)))
        return self._patch_project_models(new_ids, timeout)

    def remove(
        self,
        *model_ids: str,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> bool:
        """Remove models from the project by their IDs."""
        # Only fetch public (platform) models — custom models are managed
        # separately and must not be included in the project patch payload.
        current = self.get(timeout=timeout, type="public") or []
        remove_set = set(model_ids)
        new_ids = [str(m.id) for m in current if str(m.id) not in remove_set]
        return self._patch_project_models(new_ids, timeout)

    def _patch_project_models(
        self,
        model_ids: List[str],
        timeout: float | httpx.Timeout | None,
    ) -> bool:
        url = f"/organizations/{self._client.organization_id}/projects/{self._client.project_id}"
        resp = self._patch(
            url,
            body={"models": model_ids},
            timeout=timeout,
            cast_to=dict,
        )
        if isinstance(resp, dict):
            data = resp.get("data", resp)
            if isinstance(data, dict) and "id" in data:
                return True
        return False

    def create_custom(
        self,
        *,
        name: str,
        key: str,
        description: str,
        api_url: str,
        max_tokens: int,
        api_key: Optional[str] = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[CreateModelResponse]:
        """Create a custom model backed by an OpenAI-compatible API.

        Args:
            name: Model name (max 256 characters).
            key: Unique model key, lowercase alphanumeric with dots/hyphens/slashes (max 256 characters).
            description: Model description (max 500 characters).
            api_url: Base URL of the OpenAI-compatible API endpoint.
            max_tokens: Maximum number of tokens the model supports.
            api_key: Optional API key for the model provider.
            timeout: Request timeout override.

        Returns:
            CreateModelResponse with model_id, or None on failure.
        """
        base = f"/organizations/{self._client.organization_id}/projects/{self._client.project_id}"
        body: Dict[str, Any] = {
            "name": name,
            "key": key,
            "description": description,
            "api_url": api_url,
            "max_tokens": max_tokens,
        }
        if api_key is not None:
            body["api_key"] = api_key

        resp = self._post(
            f"{base}/custom-models",
            body=body,
            timeout=timeout,
            cast_to=dict,
        )
        if isinstance(resp, dict) and "data" in resp and "status" in resp:
            resp = resp["data"]
        if isinstance(resp, dict) and "model_id" in resp:
            return CreateModelResponse(**resp)
        return None


class AsyncModels(AsyncAPIResource):
    async def get(
        self,
        *,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
        type: Literal["custom", "public"] | None = None,
        name: Optional[str] = None,
        key: Optional[str] = None,
        categories: Optional[List[str]] = None,
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
            if categories:
                params["categories"] = ",".join(categories)
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

        models = _exclude_custom_models(
            models,
            categories=categories,
            companies=companies,
            regions=regions,
            licenses=licenses,
        )

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

    async def add(
        self,
        *model_ids: str,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> bool:
        """Add models to the project by their IDs."""
        # Only fetch public (platform) models — custom models are managed
        # separately and must not be included in the project patch payload.
        current = await self.get(timeout=timeout, type="public") or []
        current_ids = [str(m.id) for m in current]
        new_ids = list(dict.fromkeys(current_ids + list(model_ids)))
        return await self._patch_project_models(new_ids, timeout)

    async def remove(
        self,
        *model_ids: str,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> bool:
        """Remove models from the project by their IDs."""
        # Only fetch public (platform) models — custom models are managed
        # separately and must not be included in the project patch payload.
        current = await self.get(timeout=timeout, type="public") or []
        remove_set = set(model_ids)
        new_ids = [str(m.id) for m in current if str(m.id) not in remove_set]
        return await self._patch_project_models(new_ids, timeout)

    async def _patch_project_models(
        self,
        model_ids: List[str],
        timeout: float | httpx.Timeout | None,
    ) -> bool:
        url = f"/organizations/{self._client.organization_id}/projects/{self._client.project_id}"
        resp = await self._patch(
            url,
            body={"models": model_ids},
            timeout=timeout,
            cast_to=dict,
        )
        if isinstance(resp, dict):
            data = resp.get("data", resp)
            if isinstance(data, dict) and "id" in data:
                return True
        return False

    async def create_custom(
        self,
        *,
        name: str,
        key: str,
        description: str,
        api_url: str,
        max_tokens: int,
        api_key: Optional[str] = None,
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> Optional[CreateModelResponse]:
        """Create a custom model backed by an OpenAI-compatible API.

        Args:
            name: Model name (max 256 characters).
            key: Unique model key, lowercase alphanumeric with dots/hyphens/slashes (max 256 characters).
            description: Model description (max 500 characters).
            api_url: Base URL of the OpenAI-compatible API endpoint.
            max_tokens: Maximum number of tokens the model supports.
            api_key: Optional API key for the model provider.
            timeout: Request timeout override.

        Returns:
            CreateModelResponse with model_id, or None on failure.
        """
        base = f"/organizations/{self._client.organization_id}/projects/{self._client.project_id}"
        body: Dict[str, Any] = {
            "name": name,
            "key": key,
            "description": description,
            "api_url": api_url,
            "max_tokens": max_tokens,
        }
        if api_key is not None:
            body["api_key"] = api_key

        resp = await self._post(
            f"{base}/custom-models",
            body=body,
            timeout=timeout,
            cast_to=dict,
        )
        if isinstance(resp, dict) and "data" in resp and "status" in resp:
            resp = resp["data"]
        if isinstance(resp, dict) and "model_id" in resp:
            return CreateModelResponse(**resp)
        return None
