"""Sample: simulate MCP extension events with the LayerLens adapter.

Walks through the four MCP extensions in-memory: tool call, structured
output validation, elicitation request/response, and async task lifecycle.
The adapter emits ``tool.call`` + ``protocol.structured_output`` +
``protocol.elicitation_*`` + ``protocol.async_task`` events that ship to
atlas-app via ``HttpEventSink``.

This sample requires no real MCP server — the events are emitted from
in-process method calls.

Required environment:

* ``LAYERLENS_STRATIX_API_KEY`` — your LayerLens API key (optional).
* ``LAYERLENS_STRATIX_BASE_URL`` — atlas-app base URL (optional).

Run::

    pip install 'layerlens[protocols-mcp]'
    python -m samples.instrument.mcp.main
"""

from __future__ import annotations

import sys

from layerlens.instrument.adapters._base import CaptureConfig
from layerlens.instrument.transport.sink_http import HttpEventSink
from layerlens.instrument.adapters.protocols.mcp import MCPExtensionsAdapter


def main() -> int:
    sink = HttpEventSink(
        adapter_name="mcp_extensions",
        path="/telemetry/spans",
        max_batch=10,
        flush_interval_s=1.0,
    )

    adapter = MCPExtensionsAdapter(capture_config=CaptureConfig.standard())
    adapter.add_sink(sink)
    adapter.connect()

    try:
        # 1. Tool call
        adapter.on_tool_call(
            tool_name="weather.get",
            input_data={"city": "NYC"},
            output_data={"temp_f": 72, "condition": "sunny"},
            latency_ms=128.4,
        )

        # 2. Structured output (validation passed)
        adapter.on_structured_output(
            tool_name="weather.get",
            output={"temp_f": 72, "condition": "sunny"},
            schema={
                "$id": "weather-result-v1",
                "type": "object",
                "properties": {
                    "temp_f": {"type": "number"},
                    "condition": {"type": "string"},
                },
                "required": ["temp_f", "condition"],
            },
            validation_passed=True,
        )

        # 3. Elicitation
        adapter.on_elicitation_request(
            prompt_id="elic-1",
            prompt="What city?",
            schema={"type": "object", "properties": {"city": {"type": "string"}}},
        )
        adapter.on_elicitation_response(
            prompt_id="elic-1",
            response={"city": "NYC"},
            valid=True,
        )

        # 4. Async task lifecycle
        adapter.on_async_task(
            task_id="async-1",
            status="running",
            tool_name="long_query",
        )
        adapter.on_async_task(
            task_id="async-1",
            status="completed",
            tool_name="long_query",
            duration_ms=1500.0,
        )

        if hasattr(sink, "stats"):
            stats = sink.stats()
            print(f"Batches sent: {stats.get('batches_sent', 0)}")
        print("Emitted MCP events: tool + structured_output + elicitation + async_task")
    except Exception as exc:
        print(f"MCP scenario failed: {exc}", file=sys.stderr)
        return 1
    finally:
        sink.close()
        adapter.disconnect()

    print("Telemetry shipped. Check the LayerLens dashboard adapter health page.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
