"""
STRATIX CrewAI Adapter

Integrates STRATIX tracing with the CrewAI agent framework.

Usage:
    from layerlens.instrument.adapters.crewai import (
        CrewAIAdapter,
        STRATIXCrewCallback,
        instrument_crew,
    )

    adapter = CrewAIAdapter(stratix=stratix_instance)
    adapter.connect()
    instrumented_crew = adapter.instrument_crew(my_crew)
    result = instrumented_crew.kickoff()
"""

from layerlens.instrument.adapters.crewai.lifecycle import CrewAIAdapter
from layerlens.instrument.adapters.crewai.callbacks import STRATIXCrewCallback
from layerlens.instrument.adapters.crewai.delegation import CrewDelegationTracker
from layerlens.instrument.adapters.crewai.metadata import AgentMetadataExtractor

# Registry lazy-loading convention
ADAPTER_CLASS = CrewAIAdapter


def instrument_crew(crew, stratix=None, capture_config=None):
    """
    Convenience function to instrument a CrewAI crew with STRATIX tracing.

    Args:
        crew: A CrewAI Crew instance
        stratix: STRATIX SDK instance
        capture_config: CaptureConfig to use

    Returns:
        The instrumented crew
    """
    adapter = CrewAIAdapter(stratix=stratix, capture_config=capture_config)
    adapter.connect()
    return adapter.instrument_crew(crew)


__all__ = [
    "CrewAIAdapter",
    "STRATIXCrewCallback",
    "CrewDelegationTracker",
    "AgentMetadataExtractor",
    "instrument_crew",
    "ADAPTER_CLASS",
]
