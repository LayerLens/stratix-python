"""
STRATIX Layer 3 Events - Model Metadata

From Step 1 specification:
{
    "event_type": "model.invoke",
    "layer": "L3",
    "model": {
        "provider": "string",
        "name": "string",
        "version": "string",
        "parameters": { "temperature": 0.2 }
    }
}
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """Model information for L3 events."""
    provider: str = Field(
        description="Model provider (e.g., 'openai', 'anthropic')"
    )
    name: str = Field(
        description="Model name (e.g., 'gpt-4', 'claude-3-opus')"
    )
    version: str = Field(
        description="Model version or checkpoint (or 'unavailable')"
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Model parameters (temperature, max_tokens, etc.)"
    )


class ModelInvokeEvent(BaseModel):
    """
    Layer 3 Event: Model Invoke

    Represents an LLM model invocation.

    NORMATIVE:
    - Must be emitted for every LLM invocation
    - One model.invoke per request (no hidden provider calls)
    - Tool version required (or explicitly 'unavailable')
    """
    event_type: str = Field(
        default="model.invoke",
        description="Event type identifier"
    )
    layer: str = Field(
        default="L3",
        description="Layer identifier"
    )
    model: ModelInfo = Field(
        description="Model information"
    )
    prompt_tokens: int | None = Field(
        default=None,
        description="Number of prompt tokens"
    )
    completion_tokens: int | None = Field(
        default=None,
        description="Number of completion tokens"
    )
    total_tokens: int | None = Field(
        default=None,
        description="Total number of tokens"
    )
    latency_ms: float | None = Field(
        default=None,
        description="Latency in milliseconds"
    )
    input_messages: list[dict[str, str]] | None = Field(
        default=None,
        description="Input messages sent to the model (opt-in via capture_content)"
    )
    output_message: dict[str, str] | None = Field(
        default=None,
        description="Output message from the model (opt-in via capture_content)"
    )

    @classmethod
    def create(
        cls,
        provider: str,
        name: str,
        version: str = "unavailable",
        parameters: dict[str, Any] | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        latency_ms: float | None = None,
        input_messages: list[dict[str, str]] | None = None,
        output_message: dict[str, str] | None = None,
    ) -> ModelInvokeEvent:
        """
        Create a model invoke event.

        Args:
            provider: Model provider
            name: Model name
            version: Model version (default: 'unavailable')
            parameters: Model parameters
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            total_tokens: Total tokens
            latency_ms: Latency in milliseconds
            input_messages: Input messages sent to the model
            output_message: Output message from the model

        Returns:
            ModelInvokeEvent instance
        """
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
