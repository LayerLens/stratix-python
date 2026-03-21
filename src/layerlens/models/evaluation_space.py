from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class EvaluationSpaceModelFilter(BaseModel):
    ids: List[str] = []
    vendors: List[str] = []
    regions: List[str] = []


class EvaluationSpaceDatasetFilter(BaseModel):
    ids: List[str] = []
    categories: List[str] = []
    languages: List[str] = []


class EvaluationSpaceFilters(BaseModel):
    model_filters: Optional[EvaluationSpaceModelFilter] = None
    dataset_filters: Optional[EvaluationSpaceDatasetFilter] = None
    providers: List[str] = []


class EvaluationSpace(BaseModel):
    id: str
    organization_id: Optional[str] = None
    project_id: Optional[str] = None
    name: str
    description: Optional[str] = None
    filters: Optional[EvaluationSpaceFilters] = None
    owner: Optional[str] = None
    visibility: Optional[str] = None
    is_featured: bool = False
    is_partner: bool = False
    partner_name: Optional[str] = None
    created_at: Optional[str] = None
    image_path: Optional[str] = None
    weight: int = 0
    slug: Optional[str] = None
    models_count: int = 0
    benchmarks_count: int = 0
    evaluations_count: int = 0
