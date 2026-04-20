"""Pluggable synthetic-trace providers.

Providers own the heavy lifting — an LLM-backed provider would call an
actual model, while :class:`StochasticProvider` (included here) samples
realistic-looking events from numeric distributions. New providers
register themselves via :class:`ProviderRegistry`.
"""

from __future__ import annotations

import abc
import uuid
import random
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field, BaseModel

from .templates import TEMPLATE_LIBRARY
from ..models.trace import Trace


class ProviderCapability(str, Enum):
    LLM_TRACES = "llm_traces"
    AGENT_TRACES = "agent_traces"
    MULTI_AGENT_TRACES = "multi_agent_traces"
    RAG_TRACES = "rag_traces"
    TOOL_CALL_TRACES = "tool_call_traces"
    OTEL_SPANS = "otel_spans"


class ProviderTier(str, Enum):
    LOCAL = "local"
    HOSTED = "hosted"
    ENTERPRISE = "enterprise"


class ProviderInfo(BaseModel):
    id: str
    description: str = ""
    tier: ProviderTier = ProviderTier.LOCAL
    capabilities: List[ProviderCapability] = Field(default_factory=list)
    ecu_per_trace: float = 0.0
    max_batch_size: int = 1000


class GenerationResult(BaseModel):
    job_id: str
    provider_id: str
    template_id: str
    traces: List[Trace] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    total_ecu: float = 0.0


class SyntheticProvider(abc.ABC):
    """Provider interface. Subclasses implement :meth:`generate`."""

    info: ProviderInfo

    def validate_parameters(self, template_id: str, parameters: Dict[str, Any]) -> List[str]:
        template = TEMPLATE_LIBRARY.get(template_id)
        if template is None:
            return [f"unknown template: {template_id}"]
        errors: List[str] = []
        for p in template.parameters:
            if p.required and p.name not in parameters:
                errors.append(f"missing required parameter: {p.name}")
            if p.choices and p.name in parameters:
                if parameters[p.name] not in p.choices:
                    errors.append(f"parameter {p.name}={parameters[p.name]!r} not in {p.choices}")
        return errors

    def estimate_cost(self, template_id: str, count: int) -> float:  # noqa: ARG002
        return self.info.ecu_per_trace * count

    @abc.abstractmethod
    def generate(
        self,
        *,
        template_id: str,
        parameters: Dict[str, Any],
        count: int,
        project_id: Optional[str] = None,
        organization_id: Optional[str] = None,
    ) -> GenerationResult: ...


class StochasticProvider(SyntheticProvider):
    """Offline provider — no external calls, deterministic with a seed."""

    info = ProviderInfo(
        id="stochastic",
        description="Numeric distributions, no model calls.",
        tier=ProviderTier.LOCAL,
        capabilities=[
            ProviderCapability.LLM_TRACES,
            ProviderCapability.AGENT_TRACES,
            ProviderCapability.MULTI_AGENT_TRACES,
            ProviderCapability.RAG_TRACES,
            ProviderCapability.TOOL_CALL_TRACES,
        ],
        ecu_per_trace=0.0,
        max_batch_size=10_000,
    )

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    def generate(
        self,
        *,
        template_id: str,
        parameters: Dict[str, Any],
        count: int,
        project_id: Optional[str] = None,
        organization_id: Optional[str] = None,
    ) -> GenerationResult:
        job_id = f"gen_{uuid.uuid4().hex[:12]}"
        template = TEMPLATE_LIBRARY.get(template_id)
        if template is None:
            return GenerationResult(
                job_id=job_id,
                provider_id=self.info.id,
                template_id=template_id,
                errors=[f"unknown template: {template_id}"],
            )

        traces: List[Trace] = []
        for i in range(count):
            events = self._events_for_category(template.category.value, parameters)
            trace = Trace(
                id=f"synth_{uuid.uuid4().hex[:16]}",
                organization_id=organization_id or "synthetic",
                project_id=project_id or "synthetic",
                created_at="synthetic",
                filename=f"{template_id}.{i}.json",
                data={
                    "template_id": template_id,
                    "synthetic": True,
                    "events": events,
                    "latency_ms": self._rng.uniform(50, 3000),
                    "output": f"synthetic output {i}",
                },
            )
            traces.append(trace)

        return GenerationResult(
            job_id=job_id,
            provider_id=self.info.id,
            template_id=template_id,
            traces=traces,
            total_ecu=self.estimate_cost(template_id, count),
        )

    def _events_for_category(self, category: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        model = parameters.get("model", "gpt-4o-mini")
        prompt_tokens = max(1, int(self._rng.gauss(parameters.get("prompt_tokens_avg", 300), 80)))
        completion_tokens = max(1, int(self._rng.gauss(parameters.get("completion_tokens_avg", 120), 40)))

        events: List[Dict[str, Any]] = []
        if category in ("rag",):
            events.append(
                {
                    "type": "retrieval",
                    "top_k": parameters.get("top_k", 5),
                    "doc_ids": [f"doc_{j}" for j in range(parameters.get("top_k", 5))],
                }
            )
        if category in ("tool-calling", "agent", "multi-agent"):
            tool_count = self._rng.randint(1, max(1, parameters.get("tools_per_run_max", 3)))
            for j in range(tool_count):
                events.append(
                    {
                        "type": "tool.call",
                        "tool_name": f"tool_{j}",
                        "latency_ms": self._rng.uniform(20, 500),
                    }
                )
        if category == "multi-agent":
            for j in range(max(1, parameters.get("agents", 2)) - 1):
                events.append({"type": "agent.handoff", "from": f"agent_{j}", "to": f"agent_{j + 1}"})
        events.append(
            {
                "type": "model.invoke",
                "model": model,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
        )
        return events


class ProviderRegistry:
    """Singleton registry. Use :meth:`instance` to access."""

    _instance: Optional["ProviderRegistry"] = None

    def __init__(self) -> None:
        self._providers: Dict[str, SyntheticProvider] = {}
        self.register(StochasticProvider())

    @classmethod
    def instance(cls) -> "ProviderRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, provider: SyntheticProvider) -> None:
        self._providers[provider.info.id] = provider

    def get(self, provider_id: Optional[str]) -> Optional[SyntheticProvider]:
        if provider_id is None:
            return None
        return self._providers.get(provider_id)

    def auto_select(self, capabilities: List[ProviderCapability]) -> Optional[SyntheticProvider]:
        for provider in self._providers.values():
            if all(c in provider.info.capabilities for c in capabilities):
                return provider
        return None

    def list(self) -> List[ProviderInfo]:
        return [p.info for p in self._providers.values()]
