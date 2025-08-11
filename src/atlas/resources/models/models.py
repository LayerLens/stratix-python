from __future__ import annotations

from typing import List, Literal, Optional

import httpx

from ...models import Model, Models as ModelsResponse
from ..._resource import SyncAPIResource
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

        if type is None:  # fetch both
            for t in ["custom", "public"]:
                resp = fetch(t)
                if resp:
                    models.extend(resp.models)
        else:  # fetch only one type
            resp = fetch(type)
            if resp:
                models.extend(resp.models)

        return models
