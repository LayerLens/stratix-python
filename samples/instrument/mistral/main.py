"""Sample: instrument the real Mistral client with the LayerLens adapter.

Required env: ``MISTRAL_API_KEY``. Optional: ``LAYERLENS_STRATIX_API_KEY``,
``LAYERLENS_STRATIX_BASE_URL``.

Run::

    pip install 'layerlens[providers-mistral]'
    python -m samples.instrument.mistral.main
"""

from __future__ import annotations

import os
import sys

from layerlens.instrument.adapters._base import CaptureConfig
from layerlens.instrument.transport.sink_http import HttpEventSink
from layerlens.instrument.adapters.providers.mistral_adapter import MistralAdapter


def main() -> int:
    if not os.environ.get("MISTRAL_API_KEY"):
        print("MISTRAL_API_KEY is not set; cannot run sample.", file=sys.stderr)
        return 2

    try:
        from mistralai import Mistral
    except ImportError:
        print(
            "mistralai package not installed. Install with:\n"
            "    pip install 'layerlens[providers-mistral]'",
            file=sys.stderr,
        )
        return 2

    sink = HttpEventSink(
        adapter_name="mistral",
        path="/telemetry/spans",
        max_batch=10,
        flush_interval_s=1.0,
    )

    adapter = MistralAdapter(capture_config=CaptureConfig.standard())
    adapter.add_sink(sink)
    adapter.connect()

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    adapter.connect_client(client)

    try:
        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": "What is 2 + 2?"},
            ],
            max_tokens=20,
        )
        text = response.choices[0].message.content if response.choices else "(empty)"
        usage = response.usage
        print(f"Response: {text}")
        if usage is not None:
            print(
                f"Tokens — prompt: {usage.prompt_tokens}, "
                f"completion: {usage.completion_tokens}, "
                f"total: {usage.total_tokens}"
            )
    finally:
        sink.close()
        adapter.disconnect()

    print("Telemetry shipped. Check the LayerLens dashboard adapter health page.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
