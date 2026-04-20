"""Sample: Ollama local adapter. Requires ``ollama serve`` to be running locally."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from adapters._shared import capture_events  # type: ignore[import-not-found]

from layerlens.instrument.adapters.providers.ollama import instrument_ollama, uninstrument_ollama


def main() -> None:
    try:
        import ollama  # type: ignore[import-not-found]
    except ImportError:
        print("Install the Ollama extra: pip install 'layerlens[ollama]'")
        return

    client = ollama.Client()
    instrument_ollama(client, cost_per_second=0.0001)
    try:
        with capture_events("ollama"):
            try:
                resp = client.chat(
                    model=os.environ.get("OLLAMA_MODEL", "llama3.1:8b"),
                    messages=[{"role": "user", "content": "Name a mountain."}],
                )
            except Exception as exc:
                print(f"Ollama unavailable ({type(exc).__name__}): start 'ollama serve' locally.")
                return
            print("reply:", resp["message"]["content"])
    finally:
        uninstrument_ollama()


if __name__ == "__main__":
    main()
