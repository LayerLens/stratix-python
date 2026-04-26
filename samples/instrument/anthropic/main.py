"""Sample: instrument the real Anthropic client with the LayerLens adapter.

Required environment:

* ``ANTHROPIC_API_KEY`` — your Anthropic API key.
* ``LAYERLENS_STRATIX_API_KEY`` — your LayerLens API key (optional).
* ``LAYERLENS_STRATIX_BASE_URL`` — atlas-app base URL (optional).

Run::

    pip install 'layerlens[providers-anthropic]'
    python -m samples.instrument.anthropic.main
"""

from __future__ import annotations

import os
import sys

from layerlens.instrument.adapters._base import CaptureConfig
from layerlens.instrument.transport.sink_http import HttpEventSink
from layerlens.instrument.adapters.providers.anthropic_adapter import AnthropicAdapter


def main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set; cannot run sample.", file=sys.stderr)
        return 2

    try:
        from anthropic import Anthropic
    except ImportError:
        print(
            "anthropic package not installed. Install with:\n"
            "    pip install 'layerlens[providers-anthropic]'",
            file=sys.stderr,
        )
        return 2

    sink = HttpEventSink(
        adapter_name="anthropic",
        path="/telemetry/spans",
        max_batch=10,
        flush_interval_s=1.0,
    )

    adapter = AnthropicAdapter(capture_config=CaptureConfig.standard())
    adapter.add_sink(sink)
    adapter.connect()

    client = Anthropic()
    adapter.connect_client(client)

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=20,
            system="You are concise.",
            messages=[{"role": "user", "content": "What is 2 + 2?"}],
        )
        text_blocks = [b.text for b in response.content if getattr(b, "type", None) == "text"]
        print(f"Response: {' '.join(text_blocks)}")
        print(
            f"Tokens — input: {response.usage.input_tokens}, "
            f"output: {response.usage.output_tokens}"
        )
    finally:
        sink.close()
        adapter.disconnect()

    print("Telemetry shipped. Check the LayerLens dashboard adapter health page.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
