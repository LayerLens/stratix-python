from __future__ import annotations

from typing import Any, Dict, List, Union, Optional

from pydantic import BaseModel, ConfigDict


class PublicModelDetail(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    key: str
    name: str
    description: Optional[str] = None
    company: Optional[str] = None
    created_at: Optional[Union[int, str]] = None
    released_at: Optional[Union[int, str]] = None
    parameters: Optional[float] = None
    modality: Optional[str] = None
    context_length: Optional[int] = None
    architecture_type: Optional[str] = None
    license: Optional[str] = None
    open_weights: Optional[bool] = None
    region: Optional[str] = None
    key_takeaways: Optional[List[str]] = None
    deprecated: Optional[bool] = None
    cost_per_input_token: Optional[str] = None
    cost_per_output_token: Optional[str] = None


class PublicModelsListResponse(BaseModel):
    models: List[PublicModelDetail]
    categories: List[str] = []
    companies: List[str] = []
    regions: List[str] = []
    licenses: List[str] = []
    sizes: List[str] = []
    count: int
    total_count: int


class PublicBenchmarkDetail(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    key: str
    name: str
    description: Optional[str] = None
    created_at: Optional[Union[int, str]] = None
    prompt_count: Optional[int] = None
    language: Optional[str] = None
    categories: Optional[List[str]] = None
    characteristics: Optional[List[str]] = None
    deprecated: Optional[bool] = None
    is_public: Optional[bool] = None


class PublicBenchmarksListResponse(BaseModel):
    datasets: List[PublicBenchmarkDetail]
    categories: List[str] = []
    languages: List[str] = []
    count: int
    total_count: int


class BenchmarkPrompt(BaseModel):
    id: str
    input: Union[str, List[Dict[str, Any]], Dict[str, Any]]
    truth: str


class BenchmarkPromptsData(BaseModel):
    prompts: List[BenchmarkPrompt]
    count: int


class BenchmarkPromptsResponse(BaseModel):
    status: str
    data: BenchmarkPromptsData


class ComparisonResult(BaseModel):
    result_id_1: Optional[int] = None
    result_id_2: Optional[int] = None
    prompt: str
    truth: str
    result1: Optional[str] = None
    score1: Optional[float] = None
    result2: Optional[str] = None
    score2: Optional[float] = None


class ComparisonResponse(BaseModel):
    results: Optional[List[ComparisonResult]] = None
    total_count: int
    correct_count_1: int
    total_results_1: int
    correct_count_2: int
    total_results_2: int
