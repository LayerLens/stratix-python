"""Sample: instrument the real Cohere client with the LayerLens adapter.

Required env: ``COHERE_API_KEY``. Optional: ``LAYERLENS_STRATIX_API_KEY``,
``LAYERLENS_STRATIX_BASE_URL``.

Run::

    pip install 'layerlens[providers-cohere]'
    python -m samples.instrument.cohere.main
"""

from __future__ import annotations

import os
import sys

from layerlens.instrument.adapters._base import CaptureConfig
from layerlens.instrument.transport.sink_http import HttpEventSink
from layerlens.instrument.adapters.providers.cohere_adapter import CohereAdapter


def main() -> int:
    if not os.environ.get("COHERE_API_KEY"):
        print("COHERE_API_KEY is not set; cannot run sample.", file=sys.stderr)
        return 2

    try:
        import cohere
    except ImportError:
        print(
            "cohere package not installed. Install with:\n"
            "    pip install 'layerlens[providers-cohere]'",
            file=sys.stderr,
        )
        return 2

    sink = HttpEventSink(
        adapter_name="cohere",
        path="/telemetry/spans",
        max_batch=10,
        flush_interval_s=1.0,
    )

    adapter = CohereAdapter(capture_config=CaptureConfig.standard())
    adapter.add_sink(sink)
    adapter.connect()

    client = cohere.Client()
    adapter.connect_client(client)

    try:
        response = client.chat(
            model="command-r-plus",
            message="What is 2 + 2?",
            preamble="You are a concise assistant.",
        )
        print(f"Response: {response.text}")
        billed = getattr(response.meta, "billed_units", None)
        if billed is not None:
            input_tokens = getattr(billed, "input_tokens", 0)
            output_tokens = getattr(billed, "output_tokens", 0)
            print(f"Tokens — input: {input_tokens}, output: {output_tokens}")
    finally:
        sink.close()
        adapter.disconnect()

    print("Telemetry shipped. Check the LayerLens dashboard adapter health page.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
