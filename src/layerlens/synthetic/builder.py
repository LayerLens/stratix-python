"""High-level orchestrator for synthetic trace generation."""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from .providers import (
    GenerationResult,
    ProviderRegistry,
    SyntheticProvider,
    ProviderCapability,
)
from .templates import TEMPLATE_LIBRARY, TraceCategory, TraceTemplate

_CAPABILITY_FOR_CATEGORY: Dict[TraceCategory, ProviderCapability] = {
    TraceCategory.LLM: ProviderCapability.LLM_TRACES,
    TraceCategory.AGENT: ProviderCapability.AGENT_TRACES,
    TraceCategory.MULTI_AGENT: ProviderCapability.MULTI_AGENT_TRACES,
    TraceCategory.RAG: ProviderCapability.RAG_TRACES,
    TraceCategory.TOOL_CALLING: ProviderCapability.TOOL_CALL_TRACES,
    TraceCategory.OTEL: ProviderCapability.OTEL_SPANS,
}


class SyntheticDataBuilder:
    """Resolve templates/providers, validate parameters, generate traces."""

    def __init__(self, registry: Optional[ProviderRegistry] = None) -> None:
        self._registry = registry or ProviderRegistry.instance()

    def list_templates(self, category: Optional[str] = None) -> List[TraceTemplate]:
        templates = list(TEMPLATE_LIBRARY.values())
        if category:
            templates = [t for t in templates if t.category.value == category]
        return templates

    def get_template(self, template_id: str) -> Optional[TraceTemplate]:
        return TEMPLATE_LIBRARY.get(template_id)

    def estimate_cost(self, template_id: str, count: int, provider_id: Optional[str] = None) -> Dict[str, Any]:
        template = TEMPLATE_LIBRARY.get(template_id)
        if template is None:
            raise ValueError(f"unknown template: {template_id}")
        provider = self._resolve_provider(template, provider_id)
        if provider is None:
            raise ValueError(f"no provider available for template {template_id} (hint={template.provider_hint})")
        return {
            "template_id": template_id,
            "provider_id": provider.info.id,
            "count": count,
            "ecu_per_trace": provider.info.ecu_per_trace,
            "total_ecu": provider.estimate_cost(template_id, count),
            "provider_tier": provider.info.tier.value,
        }

    def generate(
        self,
        template_id: str,
        count: int,
        *,
        parameters: Optional[Dict[str, Any]] = None,
        provider_id: Optional[str] = None,
        project_id: Optional[str] = None,
        organization_id: Optional[str] = None,
    ) -> GenerationResult:
        template = TEMPLATE_LIBRARY.get(template_id)
        if template is None:
            return GenerationResult(
                job_id=f"gen_{uuid.uuid4().hex[:12]}",
                provider_id=provider_id or "unknown",
                template_id=template_id,
                errors=[f"unknown template: {template_id}"],
            )

        provider = self._resolve_provider(template, provider_id)
        if provider is None:
            return GenerationResult(
                job_id=f"gen_{uuid.uuid4().hex[:12]}",
                provider_id=provider_id or "unknown",
                template_id=template_id,
                errors=[f"no provider registered for {provider_id or template.provider_hint}"],
            )

        merged = {**template.defaults, **(parameters or {})}
        errors = provider.validate_parameters(template_id, merged)
        if errors:
            return GenerationResult(
                job_id=f"gen_{uuid.uuid4().hex[:12]}",
                provider_id=provider.info.id,
                template_id=template_id,
                errors=errors,
            )

        bounded = max(template.min_traces, min(count, template.max_traces, provider.info.max_batch_size))
        return provider.generate(
            template_id=template_id,
            parameters=merged,
            count=bounded,
            project_id=project_id,
            organization_id=organization_id,
        )

    def _resolve_provider(self, template: TraceTemplate, provider_id: Optional[str]) -> Optional[SyntheticProvider]:
        if provider_id:
            return self._registry.get(provider_id)
        if template.provider_hint:
            hint = self._registry.get(template.provider_hint)
            if hint is not None:
                return hint
        capability = _CAPABILITY_FOR_CATEGORY.get(template.category)
        if capability is not None:
            return self._registry.auto_select([capability])
        return None
