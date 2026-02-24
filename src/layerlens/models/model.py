from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class Model(BaseModel):
    id: str
    key: str
    name: str
    description: str


class CustomModel(Model):
    max_tokens: Optional[int] = None
    api_url: Optional[str] = None
    disabled: Optional[bool] = None

    @property
    def type(self) -> str:
        return "custom"


class PublicModel(Model):
    company: Optional[str] = None
    released_at: Optional[int] = None
    parameters: Optional[float] = None
    modality: Optional[str] = None
    context_length: Optional[int] = None
    architecture_type: Optional[str] = None
    license: Optional[str] = None
    open_weights: Optional[bool] = None
    region: Optional[str] = None
    deprecated: Optional[bool] = None

    @property
    def type(self) -> str:
        return "public"
