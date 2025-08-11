from __future__ import annotations

from typing import List, Optional

from pydantic import Field, BaseModel


class Benchmark(BaseModel):
    id: str
    key: str
    name: str


class CustomBenchmark(Benchmark):
    description: str
    system_prompt: Optional[str]
    prompt_count: int
    version_count: int
    regex_pattern: Optional[str]
    llm_judge_model_id: str
    custom_instructions: str
    scoring_metric: Optional[str]
    metrics: List[str]
    files: List[str]
    disabled: bool


class PublicBenchmark(Benchmark):
    description: str = Field(..., alias="full_description")
    language: str
    prompt_count: int
    deprecated: bool
