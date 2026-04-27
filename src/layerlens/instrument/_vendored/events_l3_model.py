"""Vendored snapshot of ``stratix.core.events.l3_model``.

Source: ``A:/github/layerlens/ateam/stratix/core/events/l3_model.py``
Source SHA: 7359c0e38d74e02aa1b27c34daef7a958abbd002

Compatibility shims applied for Python 3.9 + Pydantic 2:
- PEP-604 union syntax (``X | None``) on Pydantic field annotations
  rewritten as ``Optional[X]``.

Updates require re-vendoring — see ``__init__.py`` for the workflow.
"""

# STRATIX Layer 3 Events - Model Metadata
#
# {
#     "event_type": "model.invoke",
#     "layer": "L3",
#     "model": {
#         "provider": "string",
#         "name": "string",
#         "version": "string",
#         "parameters": { "temperature": 0.2 }
#     }
# }

from __future__ import annotations

from typing import Any, Optional

from pydantic import Field, BaseModel


class ModelInfo(BaseModel):
    """Model information for L3 events."""

    provider: str = Field(description="Model provider (e.g., 'openai', 'anthropic')")
    name: str = Field(description="Model name (e.g., 'gpt-4', 'claude-3-opus')")
    version: str = Field(description="Model version or checkpoint (or 'unavailable')")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Model parameters (temperature, max_tokens, etc.)"
    )


class ModelInvokeEvent(BaseModel):
    """Layer 3 Event: Model Invoke.

    Represents an LLM model invocation.

    NORMATIVE:
    - Must be emitted for every LLM invocation
    - One model.invoke per request (no hidden provider calls)
    - Tool version required (or explicitly 'unavailable')
    """

    event_type: str = Field(default="model.invoke", description="Event type identifier")
    layer: str = Field(default="L3", description="Layer identifier")
    model: ModelInfo = Field(description="Model information")
    prompt_tokens: Optional[int] = Field(default=None, description="Number of prompt tokens")
    completion_tokens: Optional[int] = Field(
        default=None, description="Number of completion tokens"
    )
    total_tokens: Optional[int] = Field(default=None, description="Total number of tokens")
    latency_ms: Optional[float] = Field(default=None, description="Latency in milliseconds")
    input_messages: Optional[list[dict[str, str]]] = Field(
        default=None, description="Input messages sent to the model (opt-in via capture_content)"
    )
    output_message: Optional[dict[str, str]] = Field(
        default=None, description="Output message from the model (opt-in via capture_content)"
    )

    @classmethod
    def create(
        cls,
        provider: str,
        name: str,
        version: str = "unavailable",
        parameters: Optional[dict[str, Any]] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        latency_ms: Optional[float] = None,
        input_messages: Optional[list[dict[str, str]]] = None,
        output_message: Optional[dict[str, str]] = None,
    ) -> ModelInvokeEvent:
        """Create a model invoke event."""
        return cls(
            model=ModelInfo(
                provider=provider,
                name=name,
                version=version,
                parameters=parameters or {},
            ),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            input_messages=input_messages,
            output_message=output_message,
        )


__all__ = [
    "ModelInfo",
    "ModelInvokeEvent",
]
