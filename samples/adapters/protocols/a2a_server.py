"""Sample: A2A adapter — server-side handler registration + client send_task."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from adapters._shared import capture_events  # type: ignore[import-not-found]

from layerlens.instrument.adapters.protocols.a2a import instrument_a2a, uninstrument_a2a


class _FakeA2AClient:
    def __init__(self) -> None:
        self._handlers = []

    def send_task(self, *, agent_id: str, skill: str, payload: dict) -> dict:
        return {"status": "completed", "result": f"{agent_id}/{skill}: {payload}"}

    def get_agent_card(self, agent_id: str) -> dict:
        return {"id": agent_id, "name": "researcher", "skills": ["lookup", "summarize"]}

    def register_handler(self, handler, *, skill: str) -> None:
        self._handlers.append((skill, handler))


def main() -> None:
    client = _FakeA2AClient()
    instrument_a2a(client)
    try:
        with capture_events("a2a"):
            client.get_agent_card("agent-1")
            client.send_task(agent_id="agent-1", skill="summarize", payload={"text": "hi"})
    finally:
        uninstrument_a2a()


if __name__ == "__main__":
    main()
