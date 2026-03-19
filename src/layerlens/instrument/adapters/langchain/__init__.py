"""
STRATIX LangChain Adapter

Integrates STRATIX tracing with LangChain framework using callbacks.

Usage:
    from layerlens.instrument.adapters.langchain import (
        STRATIXCallbackHandler,
        instrument_chain,
        instrument_agent,
    )

    # Create callback handler
    handler = STRATIXCallbackHandler(stratix_instance)

    # Use with LangChain components
    llm = ChatOpenAI(callbacks=[handler])
    chain = LLMChain(llm=llm, callbacks=[handler])

    # Or instrument existing chain/agent
    traced_chain = instrument_chain(chain, stratix_instance)
"""

from layerlens.instrument.adapters.langchain.callbacks import STRATIXCallbackHandler
from layerlens.instrument.adapters.langchain.state import LangChainMemoryAdapter
from layerlens.instrument.adapters.langchain.memory import TracedMemory, wrap_memory
from layerlens.instrument.adapters.langchain.chains import instrument_chain, TracedChain
from layerlens.instrument.adapters.langchain.agents import instrument_agent, TracedAgent

# Registry lazy-loading convention
ADAPTER_CLASS = STRATIXCallbackHandler

__all__ = [
    "STRATIXCallbackHandler",
    "LangChainMemoryAdapter",
    "TracedMemory",
    "wrap_memory",
    "instrument_chain",
    "TracedChain",
    "instrument_agent",
    "TracedAgent",
    "ADAPTER_CLASS",
]
