"""
Normalized Token Usage

Provides a common data structure for token usage across all LLM providers.
Each provider adapter constructs this from its own response format.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class NormalizedTokenUsage(BaseModel):
    """Normalized token usage across all LLM providers."""

    prompt_tokens: int = Field(default=0, description="Input tokens (prompt, system, context)")
    completion_tokens: int = Field(default=0, description="Output tokens (response, generation)")
    total_tokens: int = Field(default=0, description="prompt_tokens + completion_tokens")
    cached_tokens: int | None = Field(
        default=None,
        description="Cached prompt tokens (OpenAI cached, Anthropic cache_read)",
    )
    reasoning_tokens: int | None = Field(
        default=None,
        description="Reasoning tokens (o1/o3 reasoning, Claude extended thinking)",
    )

    @model_validator(mode="after")
    def _auto_total(self) -> "NormalizedTokenUsage":
        """Auto-compute total_tokens if not explicitly provided."""
        if self.total_tokens == 0 and (self.prompt_tokens or self.completion_tokens):
            self.total_tokens = self.prompt_tokens + self.completion_tokens
        return self

    def compute_total(self) -> NormalizedTokenUsage:
        """Return a copy with total_tokens computed from prompt + completion."""
        return self.model_copy(
            update={"total_tokens": self.prompt_tokens + self.completion_tokens},
        )
