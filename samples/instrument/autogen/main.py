"""Sample: instrument an AutoGen-style two-agent exchange with LayerLens.

This sample is fully self-contained — no network, no API keys, no
``pyautogen`` install required. It demonstrates the
``AutoGenAdapter`` lifecycle by:

1. Defining two minimal duck-typed agents that mimic AutoGen's
   ``ConversableAgent`` interface (``name``, ``send``, ``receive``,
   ``generate_reply``).
2. Connecting them through ``AutoGenAdapter.connect_agents``.
3. Driving a single send / generate_reply / receive round-trip.
4. Routing the resulting events through an in-process
   :class:`EventSink` and printing the captured stream.

To wire this against the real AutoGen runtime, replace the
``_FakeAgent`` instances with ``autogen.AssistantAgent`` /
``autogen.UserProxyAgent`` and provide ``llm_config`` with real
``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY`` credentials.

Run::

    python -m samples.instrument.autogen.main
"""

from __future__ import annotations

from typing import Any, Dict, List

from layerlens.instrument.adapters._base import CaptureConfig
from layerlens.instrument.adapters._base.sinks import EventSink
from layerlens.instrument.adapters.frameworks.autogen import (
    AutoGenAdapter,
    instrument_agents,
)


class _PrintSink(EventSink):
    """Sink that prints each event and keeps an in-memory log."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def send(self, event_type: str, payload: Dict[str, Any], timestamp_ns: int) -> None:
        self.events.append({"event_type": event_type, "payload": payload, "ts_ns": timestamp_ns})
        print(f"  [{event_type}] {sorted(payload.keys())}")

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class _RecordingStratix:
    """LayerLens-shaped client stub that the adapter emits into."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:
        if len(args) == 2 and isinstance(args[0], str):
            self.events.append({"event_type": args[0], "payload": args[1]})


class _FakeAgent:
    """Minimal duck-typed AutoGen ``ConversableAgent`` for offline demos."""

    def __init__(
        self,
        name: str,
        system_message: str = "",
        llm_config: Dict[str, Any] | None = None,
        canned_reply: str = "",
    ) -> None:
        self.name = name
        self.system_message = system_message
        self.llm_config = llm_config
        self.human_input_mode = "NEVER"
        self._canned_reply = canned_reply
        self.received: List[Any] = []

    def send(self, message: Any, recipient: Any, **_kwargs: Any) -> Any:
        recipient.receive(message, sender=self)

    def receive(self, message: Any, sender: Any, **_kwargs: Any) -> Any:
        self.received.append({"from": getattr(sender, "name", "?"), "message": message})

    def generate_reply(self, messages: Any = None, sender: Any = None, **_kwargs: Any) -> Any:
        return self._canned_reply

    def execute_code_blocks(self, code_blocks: Any) -> Any:
        return f"executed {len(code_blocks)} blocks"


def main() -> int:
    print("=== AutoGen adapter sample (mocked LLM) ===")

    stratix = _RecordingStratix()
    sink = _PrintSink()

    adapter = AutoGenAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.add_sink(sink)
    adapter.connect()

    assistant = _FakeAgent(
        name="assistant",
        system_message="You are a concise assistant.",
        llm_config={"model": "gpt-4o-mini", "temperature": 0},
        canned_reply="2 + 2 is 4.",
    )
    user = _FakeAgent(name="user", canned_reply="thanks")

    instrument_agents(assistant, user, stratix=stratix)
    adapter.connect_agents(assistant, user)

    print("-- agent.send round-trip --")
    user.send("What is 2 + 2?", recipient=assistant)

    print("-- generate_reply --")
    reply = assistant.generate_reply(
        messages=[{"role": "user", "content": "What is 2 + 2?"}],
        sender=user,
    )
    print(f"  reply: {reply!r}")

    print("-- execute_code_blocks --")
    assistant.execute_code_blocks([("python", "print('hi')")])

    print("-- conversation lifecycle --")
    adapter.on_conversation_start(initiator=user, message="What is 2 + 2?")
    adapter.on_conversation_end(final_message=reply, termination_reason="auto_reply")

    adapter.disconnect()

    print(f"\nTotal events captured by sink: {len(sink.events)}")
    print(f"Total events captured by stratix client: {len(stratix.events)}")
    print("Event types:")
    for ev in stratix.events:
        print(f"  - {ev['event_type']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
