"""Normalized Token Usage.

Provides a common data structure for token usage across all LLM
providers. Each provider adapter constructs this from its own response
format.

Ported from ``ateam/stratix/sdk/python/adapters/llm_providers/token_usage.py``.

The source uses Pydantic v2's ``model_validator`` and ``model_copy``,
which do not exist in Pydantic v1. The ``stratix-python`` SDK pins
``pydantic>=1.9.0, <3``, so this port avoids both v2-only features:

* The auto-total behavior is implemented as :meth:`with_auto_total`
  classmethod and :meth:`compute_total` instance method that construct
  fresh instances rather than relying on a validator hook.
* Callers in this codebase always pass an explicit ``total_tokens``,
  so the auto-compute is purely a defensive convenience for external
  callers.
"""

from __future__ import annotations

from typing import Optional

from layerlens._compat.pydantic import Field, BaseModel


class NormalizedTokenUsage(BaseModel):
    """Normalized token usage across all LLM providers."""

    prompt_tokens: int = Field(default=0, description="Input tokens (prompt, system, context)")
    completion_tokens: int = Field(default=0, description="Output tokens (response, generation)")
    total_tokens: int = Field(default=0, description="prompt_tokens + completion_tokens")
    cached_tokens: Optional[int] = Field(
        default=None,
        description="Cached prompt tokens (OpenAI cached, Anthropic cache_read)",
    )
    reasoning_tokens: Optional[int] = Field(
        default=None,
        description="Reasoning tokens (o1/o3 reasoning, Claude extended thinking)",
    )

    @classmethod
    def with_auto_total(
        cls,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        cached_tokens: Optional[int] = None,
        reasoning_tokens: Optional[int] = None,
    ) -> "NormalizedTokenUsage":
        """Construct a usage record, auto-computing ``total_tokens`` when zero.

        Use this constructor when the provider response does not include
        an explicit total. Callers that already have a total should
        instantiate :class:`NormalizedTokenUsage` directly.
        """
        if total_tokens == 0 and (prompt_tokens or completion_tokens):
            total_tokens = prompt_tokens + completion_tokens
        return cls(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cached_tokens=cached_tokens,
            reasoning_tokens=reasoning_tokens,
        )

    def compute_total(self) -> "NormalizedTokenUsage":
        """Return a fresh instance with ``total_tokens`` computed from prompt + completion.

        Constructs a new instance rather than calling Pydantic v2's
        ``model_copy(update=...)`` so the code runs under v1 and v2.
        """
        return type(self)(
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.prompt_tokens + self.completion_tokens,
            cached_tokens=self.cached_tokens,
            reasoning_tokens=self.reasoning_tokens,
        )
