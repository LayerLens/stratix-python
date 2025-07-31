from __future__ import annotations

from typing import List, Union, Literal

import httpx

from ..._models import Model, Models as ModelsData, CustomModel
from ..._resource import SyncAPIResource
from ..._constants import DEFAULT_TIMEOUT


class Models(SyncAPIResource):
    def get(
        self,
        *,
        type: Literal["public"] | Literal["custom"],
        timeout: float | httpx.Timeout | None = DEFAULT_TIMEOUT,
    ) -> List[Union[Model | CustomModel]] | None:
        models = self._get(
            f"/organizations/{self._client.organization_id}/projects/{self._client.project_id}/models",
            params={
                "type": type,
            },
            timeout=timeout,
            cast_to=ModelsData,
        )
        if isinstance(models, ModelsData):
            return models.models
        return None
