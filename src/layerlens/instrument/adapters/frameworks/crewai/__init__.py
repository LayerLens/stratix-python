"""
STRATIX CrewAI Adapter

Integrates STRATIX tracing with the CrewAI agent framework.

Usage:
    from layerlens.instrument.adapters.frameworks.crewai import (
        CrewAIAdapter,
        LayerLensCrewCallback,
        instrument_crew,
    )

    adapter = CrewAIAdapter(stratix=stratix_instance)
    adapter.connect()
    instrumented_crew = adapter.instrument_crew(my_crew)
    result = instrumented_crew.kickoff()
"""

from __future__ import annotations

from typing import Any

from layerlens.instrument.adapters._base.pydantic_compat import PydanticCompat, requires_pydantic

# Round-2 deliberation item 20: CrewAI >=0.30 pins ``pydantic = "^2"``;
# fail fast under v1.
requires_pydantic(PydanticCompat.V2_ONLY)

from layerlens.instrument.adapters.frameworks.crewai.metadata import AgentMetadataExtractor
from layerlens.instrument.adapters.frameworks.crewai.callbacks import LayerLensCrewCallback
from layerlens.instrument.adapters.frameworks.crewai.lifecycle import CrewAIAdapter
from layerlens.instrument.adapters.frameworks.crewai.delegation import CrewDelegationTracker

# Registry lazy-loading convention
ADAPTER_CLASS = CrewAIAdapter


def instrument_crew(crew: Any, stratix: Any = None, capture_config: dict[str, Any] = None) -> Any:  # type: ignore[assignment]
    """
    Convenience function to instrument a CrewAI crew with STRATIX tracing.

    Args:
        crew: A CrewAI Crew instance
        stratix: STRATIX SDK instance
        capture_config: CaptureConfig to use

    Returns:
        The instrumented crew
    """
    adapter = CrewAIAdapter(stratix=stratix, capture_config=capture_config)  # type: ignore[arg-type]
    adapter.connect()
    return adapter.instrument_crew(crew)


__all__ = [
    "CrewAIAdapter",
    "LayerLensCrewCallback",
    "CrewDelegationTracker",
    "AgentMetadataExtractor",
    "instrument_crew",
    "ADAPTER_CLASS",
]


# Backward-compat aliases for users coming from ateam.
STRATIXCrewCallback = LayerLensCrewCallback  # noqa: N816 - backward-compat alias for ateam users
