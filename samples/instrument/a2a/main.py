"""Sample: simulate an A2A protocol exchange with the LayerLens adapter.

Constructs an in-memory A2A scenario — agent-card discovery, task
submission, task completion — without contacting any real A2A server.
The adapter emits ``protocol.agent_card`` + ``protocol.task_submitted`` +
``protocol.task_completed`` events that ship to atlas-app via
``HttpEventSink``.

This sample requires no external services: the protocol events are emitted
purely from the in-process method calls.

Required environment:

* ``LAYERLENS_STRATIX_API_KEY`` — your LayerLens API key (optional).
* ``LAYERLENS_STRATIX_BASE_URL`` — atlas-app base URL (optional).

Run::

    pip install 'layerlens[protocols-a2a]'
    python -m samples.instrument.a2a.main
"""

from __future__ import annotations

import sys

from layerlens.instrument.adapters._base import CaptureConfig
from layerlens.instrument.transport.sink_http import HttpEventSink
from layerlens.instrument.adapters.protocols.a2a import A2AAdapter


def main() -> int:
    sink = HttpEventSink(
        adapter_name="a2a",
        path="/telemetry/spans",
        max_batch=10,
        flush_interval_s=1.0,
    )

    adapter = A2AAdapter(capture_config=CaptureConfig.standard())
    adapter.add_sink(sink)
    adapter.connect()

    try:
        adapter.register_agent_card(
            {
                "name": "research-agent",
                "url": "https://research.example.com/.well-known/agent.json",
                "protocolVersion": "0.2.0",
                "description": "Sample research agent for the LayerLens A2A demo.",
                "skills": [
                    {
                        "id": "search",
                        "name": "Web search",
                        "description": "Search the public web.",
                        "tags": ["web", "search"],
                    },
                ],
                "capabilities": {"streaming": True},
            },
            source="discovery",
        )

        adapter.on_task_submitted(
            task_id="task-001",
            receiver_url="https://research.example.com/a2a",
            task_type="research",
            submitter_agent_id="orchestrator-1",
            message_role="user",
        )

        adapter.on_stream_event(
            task_id="task-001",
            event_type="status",
            data={"status": "in-progress", "progress": 0.5},
        )

        adapter.on_task_completed(
            task_id="task-001",
            final_status="completed",
            artifacts=[
                {"type": "text", "content": "Result: 42"},
            ],
        )

        if hasattr(sink, "stats"):
            stats = sink.stats()
            print(f"Batches sent: {stats.get('batches_sent', 0)}")
        print("Emitted A2A events: agent_card + task_submitted + stream + task_completed")
    except Exception as exc:
        print(f"A2A scenario failed: {exc}", file=sys.stderr)
        return 1
    finally:
        sink.close()
        adapter.disconnect()

    print("Telemetry shipped. Check the LayerLens dashboard adapter health page.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
