"""Tests for STRATIX LangChain Chain Instrumentation."""

import pytest

from layerlens.instrument.adapters.langchain.chains import (
    TracedChain,
    instrument_chain,
    ChainTracer,
    ChainExecution,
)
from layerlens.instrument.adapters.langchain.callbacks import STRATIXCallbackHandler


class MockStratix:
    """Mock STRATIX instance for testing."""

    def __init__(self):
        self.events = []

    def emit(self, event_type: str, payload: dict):
        self.events.append({"type": event_type, "payload": payload})

    def get_events(self, event_type: str = None):
        if event_type:
            return [e for e in self.events if e["type"] == event_type]
        return self.events


class MockChain:
    """Mock LangChain chain."""

    def __init__(self, output=None):
        self._output = output or {"output": "Generated response"}
        self._invocations = []
        self.memory = None
        self.verbose = False

    def invoke(self, input, config=None, **kwargs):
        self._invocations.append({
            "input": input,
            "config": config,
            "kwargs": kwargs,
        })
        return self._output

    async def ainvoke(self, input, config=None, **kwargs):
        self._invocations.append({
            "input": input,
            "config": config,
            "kwargs": kwargs,
        })
        return self._output

    def run(self, *args, **kwargs):
        self._invocations.append({
            "args": args,
            "kwargs": kwargs,
        })
        return "Generated string"


class TestTracedChain:
    """Tests for TracedChain."""

    def test_initialization(self):
        """Test TracedChain initializes correctly."""
        chain = MockChain()
        traced = TracedChain(chain)

        assert traced._chain is chain
        assert traced._chain_type == "MockChain"
        assert isinstance(traced._handler, STRATIXCallbackHandler)

    def test_initialization_with_stratix(self):
        """Test initialization with STRATIX instance."""
        stratix = MockStratix()
        chain = MockChain()
        traced = TracedChain(chain, stratix)

        assert traced._stratix is stratix

    def test_invoke_executes_chain(self):
        """Test invoke executes underlying chain."""
        chain = MockChain()
        traced = TracedChain(chain)

        result = traced.invoke({"input": "test"})

        assert result == {"output": "Generated response"}
        assert len(chain._invocations) == 1
        assert chain._invocations[0]["input"] == {"input": "test"}

    def test_invoke_injects_callback(self):
        """Test invoke injects callback handler."""
        chain = MockChain()
        traced = TracedChain(chain)

        traced.invoke({"input": "test"})

        # Check callback was injected
        kwargs = chain._invocations[0]["kwargs"]
        assert "callbacks" in kwargs
        assert traced._handler in kwargs["callbacks"]

    def test_invoke_emits_chain_event(self):
        """Test invoke emits chain execution event."""
        stratix = MockStratix()
        chain = MockChain()
        traced = TracedChain(chain, stratix)

        traced.invoke({"input": "test"})

        events = stratix.get_events("chain.execution")
        assert len(events) == 1
        assert events[0]["payload"]["chain_type"] == "MockChain"
        assert events[0]["payload"]["inputs"] == {"input": "test"}

    def test_invoke_handles_exception(self):
        """Test invoke handles chain exceptions."""
        stratix = MockStratix()

        class FailingChain:
            def invoke(self, input, config=None, **kwargs):
                raise ValueError("Chain failed")

        traced = TracedChain(FailingChain(), stratix)

        with pytest.raises(ValueError, match="Chain failed"):
            traced.invoke({"input": "test"})

        events = stratix.get_events("chain.execution")
        assert events[0]["payload"]["error"] == "Chain failed"

    def test_invoke_tracks_execution(self):
        """Test invoke tracks execution."""
        chain = MockChain()
        traced = TracedChain(chain)

        traced.invoke({"input": "test1"})
        traced.invoke({"input": "test2"})

        assert len(traced._executions) == 2

    def test_run_method(self):
        """Test run method injects callback."""
        chain = MockChain()
        traced = TracedChain(chain)

        result = traced.run("test input")

        assert result == "Generated string"
        kwargs = chain._invocations[0]["kwargs"]
        assert "callbacks" in kwargs

    def test_attribute_proxying(self):
        """Test attribute access is proxied."""
        chain = MockChain()
        chain.custom_attr = "custom_value"
        traced = TracedChain(chain)

        assert traced.custom_attr == "custom_value"

    def test_callback_handler_property(self):
        """Test callback_handler property."""
        chain = MockChain()
        traced = TracedChain(chain)

        handler = traced.callback_handler

        assert isinstance(handler, STRATIXCallbackHandler)


@pytest.mark.asyncio
class TestTracedChainAsync:
    """Async tests for TracedChain."""

    async def test_ainvoke_executes_chain(self):
        """Test ainvoke executes underlying chain."""
        chain = MockChain()
        traced = TracedChain(chain)

        result = await traced.ainvoke({"input": "test"})

        assert result == {"output": "Generated response"}

    async def test_ainvoke_emits_event(self):
        """Test ainvoke emits chain execution event."""
        stratix = MockStratix()
        chain = MockChain()
        traced = TracedChain(chain, stratix)

        await traced.ainvoke({"input": "test"})

        events = stratix.get_events("chain.execution")
        assert len(events) == 1

    async def test_ainvoke_handles_exception(self):
        """Test ainvoke handles exceptions."""
        stratix = MockStratix()

        class FailingAsyncChain:
            async def ainvoke(self, input, config=None, **kwargs):
                raise ValueError("Async chain failed")

        traced = TracedChain(FailingAsyncChain(), stratix)

        with pytest.raises(ValueError, match="Async chain failed"):
            await traced.ainvoke({"input": "test"})


class TestInstrumentChain:
    """Tests for instrument_chain function."""

    def test_creates_traced_chain(self):
        """Test creates TracedChain instance."""
        chain = MockChain()
        traced = instrument_chain(chain)

        assert isinstance(traced, TracedChain)

    def test_passes_stratix_instance(self):
        """Test passes STRATIX instance."""
        stratix = MockStratix()
        chain = MockChain()
        traced = instrument_chain(chain, stratix)

        assert traced._stratix is stratix


class TestChainTracer:
    """Tests for ChainTracer."""

    def test_initialization(self):
        """Test tracer initializes correctly."""
        tracer = ChainTracer()

        assert tracer._chains == {}

    def test_trace_creates_traced_chain(self):
        """Test trace creates traced chain."""
        tracer = ChainTracer()
        chain = MockChain()

        traced = tracer.trace(chain)

        assert isinstance(traced, TracedChain)

    def test_trace_with_custom_name(self):
        """Test trace with custom name."""
        tracer = ChainTracer()
        chain = MockChain()

        traced = tracer.trace(chain, name="my_chain")

        assert "my_chain" in tracer._chains

    def test_get_chain(self):
        """Test get_chain retrieves traced chain."""
        tracer = ChainTracer()
        chain = MockChain()

        tracer.trace(chain, name="test")
        retrieved = tracer.get_chain("test")

        assert retrieved is not None

    def test_get_chain_not_found(self):
        """Test get_chain returns None for unknown chain."""
        tracer = ChainTracer()

        result = tracer.get_chain("unknown")

        assert result is None

    def test_get_events(self):
        """Test get_events returns events from handler."""
        stratix = MockStratix()
        tracer = ChainTracer(stratix)
        chain = MockChain()

        traced = tracer.trace(chain)
        traced.invoke({"input": "test"})

        # Events should be retrievable via tracer's handler
        events = tracer._handler.get_events()
        assert isinstance(events, list)


class TestChainExecution:
    """Tests for ChainExecution dataclass."""

    def test_execution_creation(self):
        """Test execution creation."""
        execution = ChainExecution(
            chain_type="TestChain",
            start_time_ns=1000,
            inputs={"input": "test"},
        )

        assert execution.chain_type == "TestChain"
        assert execution.start_time_ns == 1000
        assert execution.inputs == {"input": "test"}
        assert execution.end_time_ns is None
        assert execution.error is None

    def test_execution_with_all_fields(self):
        """Test execution with all fields."""
        execution = ChainExecution(
            chain_type="TestChain",
            start_time_ns=1000,
            end_time_ns=2000,
            inputs={"input": "test"},
            outputs={"output": "result"},
            error=None,
        )

        assert execution.end_time_ns == 2000
        assert execution.outputs == {"output": "result"}
