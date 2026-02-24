from __future__ import annotations

from typing import List, Optional

from pydantic import Field, BaseModel


class Benchmark(BaseModel):
    id: str
    key: str
    name: str


class CustomBenchmark(Benchmark):
    description: Optional[str] = None
    system_prompt: Optional[str] = None
    prompt_count: Optional[int] = None
    version_count: Optional[int] = None
    regex_pattern: Optional[str] = None
    llm_judge_model_id: Optional[str] = None
    custom_instructions: Optional[str] = None
    scoring_metric: Optional[str] = None
    metrics: Optional[List[str]] = None
    files: Optional[List[str]] = None
    disabled: Optional[bool] = None

    @property
    def type(self) -> str:
        return "custom"


class PublicBenchmark(Benchmark):
    description: Optional[str] = Field(None, alias="full_description")
    language: Optional[str] = None
    prompt_count: Optional[int] = None
    deprecated: Optional[bool] = None

    @property
    def type(self) -> str:
        return "public"
