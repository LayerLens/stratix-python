"""Sample: instrument the real OpenAI client with the LayerLens adapter.

Runs a single chat completion through ``OpenAIAdapter`` with an
``HttpEventSink`` pointed at atlas-app. Every event the adapter emits
(``model.invoke``, ``cost.record``, optional ``tool.call``) is shipped
to the platform's telemetry ingest endpoint.

Required environment:

* ``OPENAI_API_KEY`` — your OpenAI API key.
* ``LAYERLENS_STRATIX_API_KEY`` — your LayerLens API key (optional;
  defaults to anonymous if unset).
* ``LAYERLENS_STRATIX_BASE_URL`` — atlas-app base URL (optional;
  defaults to ``https://api.layerlens.ai/api/v1``).

Run::

    pip install 'layerlens[providers-openai]'
    python -m samples.instrument.openai.main
"""

from __future__ import annotations

import os
import sys

from layerlens.instrument.adapters._base import CaptureConfig
from layerlens.instrument.transport.sink_http import HttpEventSink
from layerlens.instrument.adapters.providers.openai_adapter import OpenAIAdapter


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set; cannot run sample.", file=sys.stderr)
        return 2

    try:
        from openai import OpenAI
    except ImportError:
        print(
            "openai package not installed. Install with:\n"
            "    pip install 'layerlens[providers-openai]'",
            file=sys.stderr,
        )
        return 2

    sink = HttpEventSink(
        adapter_name="openai",
        path="/telemetry/spans",
        max_batch=10,
        flush_interval_s=1.0,
    )

    adapter = OpenAIAdapter(capture_config=CaptureConfig.standard())
    adapter.add_sink(sink)
    adapter.connect()

    client = OpenAI()
    adapter.connect_client(client)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": "What is 2 + 2?"},
            ],
            max_tokens=20,
        )
        choice = response.choices[0].message.content if response.choices else "(empty)"
        usage = response.usage
        print(f"Response: {choice}")
        if usage is not None:
            print(
                f"Tokens — prompt: {usage.prompt_tokens}, completion: "
                f"{usage.completion_tokens}, total: {usage.total_tokens}"
            )
    finally:
        sink.close()
        adapter.disconnect()

    print("Telemetry shipped. Check the LayerLens dashboard adapter health page.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
