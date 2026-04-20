from __future__ import annotations

from typing import Optional

from pydantic import Field, BaseModel, model_validator


class NormalizedTokenUsage(BaseModel):
    """Normalized token usage across all LLM providers."""

    prompt_tokens: int = Field(default=0, description="Input tokens (prompt, system, context)")
    completion_tokens: int = Field(default=0, description="Output tokens (response, generation)")
    total_tokens: int = Field(default=0, description="prompt_tokens + completion_tokens")
    # NOTE: Pydantic evaluates field annotations at model-build time, so we use
    # ``Optional[int]`` here — PEP 604 ``int | None`` breaks on Python 3.9 even
    # with ``from __future__ import annotations``.
    cached_tokens: Optional[int] = Field(
        default=None,
        description="Cached prompt tokens (OpenAI prompt cache, Anthropic cache_read)",
    )
    cache_creation_tokens: Optional[int] = Field(
        default=None,
        description="Tokens written to cache on this call (Anthropic cache_creation_input_tokens)",
    )
    reasoning_tokens: Optional[int] = Field(
        default=None,
        description="Reasoning tokens (o1/o3) or extended thinking tokens (Claude)",
    )
    thinking_tokens: Optional[int] = Field(
        default=None,
        description="Extended thinking tokens — alias surfaced for Anthropic thinking blocks",
    )

    @model_validator(mode="after")
    def _auto_total(self) -> NormalizedTokenUsage:
        if self.total_tokens == 0 and (self.prompt_tokens or self.completion_tokens):
            self.total_tokens = self.prompt_tokens + self.completion_tokens
        return self

    def as_event_dict(self) -> dict[str, int]:
        """Render as a flat int dict suitable for emission in cost.record events.

        Skips ``None`` optional fields so we don't pollute downstream telemetry.
        """
        out: dict[str, int] = {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }
        for key in ("cached_tokens", "cache_creation_tokens", "reasoning_tokens", "thinking_tokens"):
            val = getattr(self, key)
            if val is not None:
                out[key] = val
        return out
