from __future__ import annotations

from pydantic import BaseModel


class Model(BaseModel):
    id: str
    key: str
    name: str
    description: str


class CustomModel(Model):
    max_tokens: int
    api_url: str
    disabled: bool


class PublicModel(Model):
    company: str
    released_at: int
    parameters: float
    modality: str
    context_length: int
    architecture_type: str
    license: str
    open_weights: bool
    region: str
    deprecated: bool
