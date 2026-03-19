"""Simulator configuration models.

Pydantic-based configuration following CaptureConfig preset pattern
from stratix/sdk/python/adapters/capture.py.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class SourceFormat(str, Enum):
    """Supported ingestion source formats (12 sources)."""

    GENERIC_OTEL = "generic_otel"
    AGENTFORCE_OTLP = "agentforce_otlp"
    AGENTFORCE_SOQL = "agentforce_soql"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    BEDROCK = "bedrock"
    GOOGLE_VERTEX = "google_vertex"
    OLLAMA = "ollama"
    LITELLM = "litellm"
    LANGFUSE = "langfuse"
    JSONL = "jsonl"


class OutputFormat(str, Enum):
    """Output wire formats."""

    OTLP_JSON = "otlp_json"
    LANGFUSE_JSON = "langfuse_json"
    STRATIX_NATIVE = "stratix_native"


class ScenarioName(str, Enum):
    """Available scenario types."""

    CUSTOMER_SERVICE = "customer_service"
    SALES = "sales"
    ORDER_MANAGEMENT = "order_management"
    KNOWLEDGE_FAQ = "knowledge_faq"
    IT_HELPDESK = "it_helpdesk"


class ContentTier(str, Enum):
    """Content generation tier."""

    SEED = "seed"
    TEMPLATE = "template"
    LLM = "llm"


class ContentConfig(BaseModel):
    """Content generation configuration."""

    tier: ContentTier = ContentTier.TEMPLATE
    seed_data_path: str | None = None
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str | None = None
    llm_cache_enabled: bool = True
    llm_cache_path: str | None = None


class ConversationConfig(BaseModel):
    """Multi-turn conversation configuration."""

    enabled: bool = False
    turns_min: int = Field(default=2, ge=1, le=20)
    turns_max: int = Field(default=5, ge=1, le=20)

    @field_validator("turns_max")
    @classmethod
    def max_gte_min(cls, v: int, info: Any) -> int:
        min_val = info.data.get("turns_min", 2)
        if v < min_val:
            raise ValueError(f"turns_max ({v}) must be >= turns_min ({min_val})")
        return v


class StreamingConfig(BaseModel):
    """Streaming behavior configuration."""

    enabled: bool = False
    ttft_ms_min: float = Field(default=50.0, ge=0.0)
    ttft_ms_max: float = Field(default=500.0, ge=0.0)
    tpot_ms_min: float = Field(default=10.0, ge=0.0)
    tpot_ms_max: float = Field(default=50.0, ge=0.0)
    chunks_min: int = Field(default=5, ge=1)
    chunks_max: int = Field(default=50, ge=1)

    @field_validator("ttft_ms_max")
    @classmethod
    def ttft_max_gte_min(cls, v: float, info: Any) -> float:
        min_val = info.data.get("ttft_ms_min", 50.0)
        if v < min_val:
            raise ValueError(f"ttft_ms_max ({v}) must be >= ttft_ms_min ({min_val})")
        return v

    @field_validator("tpot_ms_max")
    @classmethod
    def tpot_max_gte_min(cls, v: float, info: Any) -> float:
        min_val = info.data.get("tpot_ms_min", 10.0)
        if v < min_val:
            raise ValueError(f"tpot_ms_max ({v}) must be >= tpot_ms_min ({min_val})")
        return v

    @field_validator("chunks_max")
    @classmethod
    def chunks_max_gte_min(cls, v: int, info: Any) -> int:
        min_val = info.data.get("chunks_min", 5)
        if v < min_val:
            raise ValueError(f"chunks_max ({v}) must be >= chunks_min ({min_val})")
        return v


class ErrorConfig(BaseModel):
    """Error injection configuration."""

    enabled: bool = False
    rate_limit_probability: float = Field(default=0.05, ge=0.0, le=1.0)
    timeout_probability: float = Field(default=0.03, ge=0.0, le=1.0)
    auth_failure_probability: float = Field(default=0.01, ge=0.0, le=1.0)
    content_filter_probability: float = Field(default=0.02, ge=0.0, le=1.0)
    server_error_probability: float = Field(default=0.02, ge=0.0, le=1.0)


class SimulatorConfig(BaseModel):
    """Main simulator configuration.

    Follows CaptureConfig preset pattern with minimal/standard/full factories.
    """

    source_format: SourceFormat = SourceFormat.GENERIC_OTEL
    output_format: OutputFormat = OutputFormat.OTLP_JSON
    scenario: ScenarioName = ScenarioName.CUSTOMER_SERVICE
    seed: int | None = None
    count: int = Field(default=1, ge=1)
    include_content: bool = False
    streaming: StreamingConfig = Field(default_factory=StreamingConfig)
    errors: ErrorConfig = Field(default_factory=ErrorConfig)
    conversation: ConversationConfig = Field(default_factory=ConversationConfig)
    content: ContentConfig = Field(default_factory=ContentConfig)
    dry_run: bool = False
    output_path: str | None = None

    @classmethod
    def minimal(cls) -> SimulatorConfig:
        """1 trace, template content, no errors — lightweight testing."""
        return cls(
            count=1,
            content=ContentConfig(tier=ContentTier.TEMPLATE),
            errors=ErrorConfig(enabled=False),
            streaming=StreamingConfig(enabled=False),
            conversation=ConversationConfig(enabled=False),
        )

    @classmethod
    def standard(cls) -> SimulatorConfig:
        """10 traces, conversations, 5% errors — recommended."""
        return cls(
            count=10,
            content=ContentConfig(tier=ContentTier.TEMPLATE),
            conversation=ConversationConfig(enabled=True, turns_min=2, turns_max=5),
            errors=ErrorConfig(enabled=True, rate_limit_probability=0.05),
            streaming=StreamingConfig(enabled=False),
        )

    @classmethod
    def full(cls) -> SimulatorConfig:
        """100 traces, all features enabled — comprehensive testing."""
        return cls(
            count=100,
            include_content=True,
            content=ContentConfig(tier=ContentTier.TEMPLATE),
            conversation=ConversationConfig(enabled=True, turns_min=2, turns_max=5),
            errors=ErrorConfig(
                enabled=True,
                rate_limit_probability=0.05,
                timeout_probability=0.03,
                auth_failure_probability=0.01,
                content_filter_probability=0.02,
                server_error_probability=0.02,
            ),
            streaming=StreamingConfig(enabled=True),
        )

    @classmethod
    def from_yaml(cls, path: str) -> SimulatorConfig:
        """Load configuration from a YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)

        config_data = data.get("simulator", data)
        # Map YAML nested keys to flat config
        if "conversation" in config_data and isinstance(config_data["conversation"], dict):
            conv = config_data["conversation"]
            if "turns_range" in conv:
                turns = conv.pop("turns_range")
                conv["turns_min"] = turns[0]
                conv["turns_max"] = turns[1]
        if "streaming" in config_data and isinstance(config_data["streaming"], dict):
            streaming = config_data["streaming"]
            if "ttft_ms_range" in streaming:
                r = streaming.pop("ttft_ms_range")
                streaming["ttft_ms_min"] = r[0]
                streaming["ttft_ms_max"] = r[1]
            if "tpot_ms_range" in streaming:
                r = streaming.pop("tpot_ms_range")
                streaming["tpot_ms_min"] = r[0]
                streaming["tpot_ms_max"] = r[1]
        if "content" in config_data and isinstance(config_data["content"], dict):
            content = config_data["content"]
            if "seed_data_path" not in content:
                env_path = os.environ.get("STRATIX_SIMULATOR_SEED_DATA_PATH")
                if env_path:
                    content["seed_data_path"] = env_path

        return cls(**config_data)

    def to_yaml(self, path: str | None = None) -> str:
        """Serialize configuration to YAML string, optionally writing to file."""
        import yaml

        data = {"simulator": self.model_dump(mode="json")}
        yaml_str = yaml.dump(data, default_flow_style=False, sort_keys=False)
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                f.write(yaml_str)
        return yaml_str
