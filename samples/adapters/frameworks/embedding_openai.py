"""Sample: OpenAI embeddings instrumented with layerlens (cross-cutting).

The embedding adapter wraps ``client.embeddings.create`` on a real OpenAI
client, so every embedding call emits an ``embedding.create`` event with the
model, input size, and token usage. Single-agent / cross-cutting: it is not an
agent framework, it instruments the embeddings surface of the provider SDK.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from adapters._shared import capture_events  # type: ignore[import-not-found]

from layerlens.instrument.adapters.frameworks.embedding import EmbeddingAdapter


def main() -> None:
    try:
        import openai  # type: ignore[import-not-found]
    except ImportError:
        print("Install: pip install 'layerlens[openai]' openai")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run OpenAI embeddings against the live API.")
        return

    client = openai.OpenAI()

    adapter = EmbeddingAdapter(None)
    adapter.connect(client)
    try:
        with capture_events("embedding_openai"):
            resp = client.embeddings.create(
                model="text-embedding-3-small",
                input="oceans rivers lakes",
            )
            print("embedding dims:", len(resp.data[0].embedding))
    finally:
        adapter.disconnect()


if __name__ == "__main__":
    main()
