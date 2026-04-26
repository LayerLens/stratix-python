"""Sample: simulate an AG-UI SSE event stream with the LayerLens adapter.

Constructs an in-memory AG-UI session — a text message stream, a state
delta, and a tool call — without contacting any real frontend. The
adapter emits ``protocol.stream.event`` + ``agent.state.change`` +
``tool.call`` events that ship to atlas-app via ``HttpEventSink``.

Required environment:

* ``LAYERLENS_STRATIX_API_KEY`` — your LayerLens API key (optional).
* ``LAYERLENS_STRATIX_BASE_URL`` — atlas-app base URL (optional).

Run::

    pip install 'layerlens[protocols-agui]'
    python -m samples.instrument.agui.main
"""

from __future__ import annotations

import sys

from layerlens.instrument.adapters._base import CaptureConfig
from layerlens.instrument.transport.sink_http import HttpEventSink
from layerlens.instrument.adapters.protocols.agui import AGUIAdapter


def main() -> int:
    sink = HttpEventSink(
        adapter_name="agui",
        path="/telemetry/spans",
        max_batch=10,
        flush_interval_s=1.0,
    )

    # Enable l6b_protocol_streams so we see per-chunk text events too.
    config = CaptureConfig.standard()
    config.l6b_protocol_streams = True
    adapter = AGUIAdapter(capture_config=config)
    adapter.add_sink(sink)
    adapter.connect()

    try:
        # 1. Text message stream
        adapter.on_agui_event(
            "TEXT_MESSAGE_START",
            payload={"thread_id": "thread-1", "message_id": "msg-1", "role": "agent"},
        )
        for chunk in ("Hello", " ", "world"):
            adapter.on_agui_event(
                "TEXT_MESSAGE_CONTENT",
                payload={"thread_id": "thread-1", "content": chunk},
            )
        adapter.on_agui_event(
            "TEXT_MESSAGE_END",
            payload={"thread_id": "thread-1"},
        )

        # 2. State delta
        adapter.on_agui_event(
            "STATE_DELTA",
            payload={"thread_id": "thread-1", "patch": {"step": 1}},
        )

        # 3. Tool call
        adapter.on_agui_event(
            "TOOL_CALL_START",
            payload={"tool_call_id": "tc-1", "name": "search"},
        )
        adapter.on_agui_event(
            "TOOL_CALL_END",
            payload={"tool_call_id": "tc-1", "result": "ok"},
        )

        if hasattr(sink, "stats"):
            stats = sink.stats()
            print(f"Batches sent: {stats.get('batches_sent', 0)}")
        print("Emitted AG-UI events: text stream + state delta + tool call")
    except Exception as exc:
        print(f"AG-UI scenario failed: {exc}", file=sys.stderr)
        return 1
    finally:
        sink.close()
        adapter.disconnect()

    print("Telemetry shipped. Check the LayerLens dashboard adapter health page.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
