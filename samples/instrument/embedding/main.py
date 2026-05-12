"""Sample: instrument an OpenAI embedding call with the LayerLens adapter.

Wraps an OpenAI client with ``EmbeddingAdapter.wrap_openai`` and runs a
single ``embeddings.create`` call. Emits one ``embedding.create`` event
that ships to atlas-app via ``HttpEventSink``.

Required environment:

* ``OPENAI_API_KEY`` — your OpenAI API key.
* ``LAYERLENS_STRATIX_API_KEY`` — your LayerLens API key (optional).
* ``LAYERLENS_STRATIX_BASE_URL`` — atlas-app base URL (optional).

Run::

    pip install 'layerlens[embedding,providers-openai]'
    python -m samples.instrument.embedding.main
"""

from __future__ import annotations

import os
import sys

from layerlens.instrument.adapters._base import CaptureConfig
from layerlens.instrument.transport.sink_http import HttpEventSink
from layerlens.instrument.adapters.frameworks.embedding import EmbeddingAdapter


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set; cannot run sample.", file=sys.stderr)
        return 2

    try:
        from openai import OpenAI
    except ImportError:
        print(
            "openai not installed. Install with:\n"
            "    pip install 'layerlens[embedding,providers-openai]'",
            file=sys.stderr,
        )
        return 2

    sink = HttpEventSink(
        adapter_name="embedding",
        path="/telemetry/spans",
        max_batch=10,
        flush_interval_s=1.0,
    )

    adapter = EmbeddingAdapter(capture_config=CaptureConfig.standard())
    adapter.add_sink(sink)
    adapter.connect()

    client = OpenAI()
    adapter.wrap_openai(client)

    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=["hello world", "the quick brown fox"],
        )
        first = response.data[0].embedding
        print(f"Embeddings: {len(response.data)} vectors of dim {len(first)}")
        if response.usage is not None:
            print(f"Tokens: {response.usage.total_tokens}")
    finally:
        sink.close()
        adapter.disconnect()

    print("Telemetry shipped. Check the LayerLens dashboard adapter health page.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
