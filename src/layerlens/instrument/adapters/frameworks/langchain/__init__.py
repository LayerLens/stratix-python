"""
STRATIX LangChain Adapter

Integrates STRATIX tracing with LangChain framework using callbacks.

Usage:
    from layerlens.instrument.adapters.frameworks.langchain import (
        LayerLensCallbackHandler,
        instrument_chain,
        instrument_agent,
    )

    # Create callback handler
    handler = LayerLensCallbackHandler(stratix_instance)

    # Use with LangChain components
    llm = ChatOpenAI(callbacks=[handler])
    chain = LLMChain(llm=llm, callbacks=[handler])

    # Or instrument existing chain/agent
    traced_chain = instrument_chain(chain, stratix_instance)
"""

from __future__ import annotations

from layerlens.instrument.adapters._base.pydantic_compat import PydanticCompat, requires_pydantic

# Round-2 deliberation item 20: fail fast under v1 with a clear message
# rather than letting LangChain raise an opaque ImportError mid-callback.
requires_pydantic(PydanticCompat.V2_ONLY)

from layerlens.instrument.adapters.frameworks.langchain.lcel import (
    LCELNode,
    RunnableKind,
    CompositionPosition,
    LCELRunnableTracker,
    fingerprint_lambda,
    detect_runnable_kind,
    parse_composition_tag,
    parse_parallel_branches,
)
from layerlens.instrument.adapters.frameworks.langchain.state import LangChainMemoryAdapter
from layerlens.instrument.adapters.frameworks.langchain.agents import TracedAgent, instrument_agent
from layerlens.instrument.adapters.frameworks.langchain.chains import TracedChain, instrument_chain
from layerlens.instrument.adapters.frameworks.langchain.memory import TracedMemory, wrap_memory
from layerlens.instrument.adapters.frameworks.langchain.callbacks import LayerLensCallbackHandler

# Registry lazy-loading convention
ADAPTER_CLASS = LayerLensCallbackHandler

__all__ = [
    "LayerLensCallbackHandler",
    "LangChainMemoryAdapter",
    "TracedMemory",
    "wrap_memory",
    "instrument_chain",
    "TracedChain",
    "instrument_agent",
    "TracedAgent",
    "ADAPTER_CLASS",
    # LCEL public surface (spec 04b §4).
    "LCELRunnableTracker",
    "LCELNode",
    "RunnableKind",
    "CompositionPosition",
    "detect_runnable_kind",
    "parse_composition_tag",
    "parse_parallel_branches",
    "fingerprint_lambda",
]


# Backward-compat aliases for users coming from ateam.
STRATIXCallbackHandler = LayerLensCallbackHandler  # noqa: N816 - backward-compat alias for ateam users
