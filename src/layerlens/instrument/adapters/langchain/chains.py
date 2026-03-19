"""
STRATIX LangChain Chain Instrumentation

Provides automatic instrumentation for LangChain chains.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from layerlens.instrument.adapters.langchain.callbacks import STRATIXCallbackHandler

if TYPE_CHECKING:
    from layerlens.instrument.adapters._base import BaseAdapter


@dataclass
class ChainExecution:
    """Tracks a single chain execution."""
    chain_type: str
    start_time_ns: int
    end_time_ns: int | None = None
    inputs: dict[str, Any] | None = None
    outputs: dict[str, Any] | None = None
    error: str | None = None


class TracedChain:
    """
    Wrapper around a LangChain chain with STRATIX tracing.

    Automatically injects STRATIXCallbackHandler and tracks
    chain executions.

    Usage:
        from langchain.chains import LLMChain

        chain = LLMChain(llm=llm, prompt=prompt)
        traced_chain = TracedChain(chain, stratix_instance)

        # Use as normal
        result = traced_chain.invoke({"input": "hello"})
    """

    def __init__(
        self,
        chain: Any,
        stratix_instance: Any = None,
        adapter: BaseAdapter | None = None,
    ):
        """
        Initialize the traced chain.

        Args:
            chain: LangChain chain instance
            stratix_instance: STRATIX SDK instance (legacy)
            adapter: BaseAdapter instance (new-style)
        """
        self._chain = chain
        self._stratix = stratix_instance
        self._adapter = adapter
        self._handler = STRATIXCallbackHandler(
            stratix=adapter._stratix if adapter else None,
            capture_config=adapter.capture_config if adapter else None,
            stratix_instance=stratix_instance,
        )
        self._chain_type = type(chain).__name__
        self._executions: list[ChainExecution] = []

    def invoke(
        self,
        input: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Invoke the chain with tracing.

        Args:
            input: Input dictionary
            config: Optional config
            **kwargs: Additional arguments

        Returns:
            Chain output
        """
        execution = ChainExecution(
            chain_type=self._chain_type,
            start_time_ns=time.time_ns(),
            inputs=input,
        )
        self._executions.append(execution)

        # Inject callback handler
        callbacks = kwargs.get("callbacks", [])
        if self._handler not in callbacks:
            callbacks = list(callbacks) + [self._handler]
            kwargs["callbacks"] = callbacks

        try:
            # Execute chain
            result = self._chain.invoke(input, config, **kwargs)

            execution.end_time_ns = time.time_ns()
            execution.outputs = result if isinstance(result, dict) else {"output": result}

            # Emit chain completion event
            self._emit_chain_event(execution)

            return result

        except Exception as e:
            execution.end_time_ns = time.time_ns()
            execution.error = str(e)
            self._emit_chain_event(execution)
            raise

    async def ainvoke(
        self,
        input: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Async invoke the chain with tracing.

        Args:
            input: Input dictionary
            config: Optional config
            **kwargs: Additional arguments

        Returns:
            Chain output
        """
        execution = ChainExecution(
            chain_type=self._chain_type,
            start_time_ns=time.time_ns(),
            inputs=input,
        )
        self._executions.append(execution)

        # Inject callback handler
        callbacks = kwargs.get("callbacks", [])
        if self._handler not in callbacks:
            callbacks = list(callbacks) + [self._handler]
            kwargs["callbacks"] = callbacks

        try:
            result = await self._chain.ainvoke(input, config, **kwargs)

            execution.end_time_ns = time.time_ns()
            execution.outputs = result if isinstance(result, dict) else {"output": result}

            self._emit_chain_event(execution)

            return result

        except Exception as e:
            execution.end_time_ns = time.time_ns()
            execution.error = str(e)
            self._emit_chain_event(execution)
            raise

    def run(self, *args: Any, **kwargs: Any) -> str:
        """
        Run the chain (deprecated LangChain method).

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Chain output string
        """
        # Inject callback
        callbacks = kwargs.get("callbacks", [])
        if self._handler not in callbacks:
            callbacks = list(callbacks) + [self._handler]
            kwargs["callbacks"] = callbacks

        return self._chain.run(*args, **kwargs)

    def _emit_chain_event(self, execution: ChainExecution) -> None:
        """Emit chain execution event."""
        duration_ns = (execution.end_time_ns or 0) - execution.start_time_ns
        payload = {
            "chain_type": execution.chain_type,
            "inputs": execution.inputs,
            "outputs": execution.outputs,
            "duration_ns": duration_ns,
            "error": execution.error,
        }

        # New-style: route through adapter's circuit-breaker path
        if self._adapter is not None:
            self._adapter.emit_dict_event("chain.execution", payload)
            return

        # Legacy
        if self._stratix and hasattr(self._stratix, "emit"):
            self._stratix.emit("chain.execution", payload)

    @property
    def callback_handler(self) -> STRATIXCallbackHandler:
        """Get the callback handler."""
        return self._handler

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to underlying chain."""
        return getattr(self._chain, name)


def instrument_chain(
    chain: Any,
    stratix_instance: Any = None,
    adapter: BaseAdapter | None = None,
) -> TracedChain:
    """
    Instrument a LangChain chain with STRATIX tracing.

    Args:
        chain: LangChain chain instance
        stratix_instance: STRATIX SDK instance
        adapter: BaseAdapter instance (new-style)

    Returns:
        TracedChain wrapper
    """
    return TracedChain(chain, stratix_instance, adapter=adapter)


class ChainTracer:
    """
    Tracer for multiple chain executions.

    Useful for tracking chains in a larger workflow.
    """

    def __init__(self, stratix_instance: Any = None, adapter: BaseAdapter | None = None):
        """
        Initialize the chain tracer.

        Args:
            stratix_instance: STRATIX SDK instance
            adapter: BaseAdapter instance (new-style)
        """
        self._stratix = stratix_instance
        self._adapter = adapter
        self._handler = STRATIXCallbackHandler(
            stratix=adapter._stratix if adapter else None,
            capture_config=adapter.capture_config if adapter else None,
            stratix_instance=stratix_instance,
        )
        self._chains: dict[str, TracedChain] = {}

    def trace(self, chain: Any, name: str | None = None) -> TracedChain:
        """
        Start tracing a chain.

        Args:
            chain: LangChain chain
            name: Optional name for the chain

        Returns:
            TracedChain wrapper
        """
        chain_name = name or type(chain).__name__
        traced = TracedChain(chain, self._stratix, adapter=self._adapter)
        self._chains[chain_name] = traced
        return traced

    def get_events(self, event_type: str | None = None) -> list[dict[str, Any]]:
        """Get all events from the callback handler."""
        return self._handler.get_events(event_type)

    def get_chain(self, name: str) -> TracedChain | None:
        """Get a traced chain by name."""
        return self._chains.get(name)
