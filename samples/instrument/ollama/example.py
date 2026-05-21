"""Runnable sample: Ollama + LayerLens instrumentation (LAY-3454).

Run with::

    pip install layerlens[providers-ollama]
    # In another shell: ollama serve   (or use OLLAMA_HOST=...)
    python samples/instrument/ollama/example.py

See ``docs/adapters/providers/ollama.md`` for ``ollama serve`` setup and the
optional ``cost_per_second`` knob that attributes compute time as infra cost.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import Mock


def main() -> int:
    layerlens_client = Mock(name="LayerLensClient")
    try:
        from layerlens.instrument.adapters.providers import OllamaProvider
    except ImportError as exc:
        print(f"[skipped] {exc}")
        print("Install Ollama deps with: pip install layerlens[providers-ollama]")
        return 0

    print("OllamaProvider available.")
    try:
        import ollama  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        print(f"[skipped] {exc}")
        print("Install the Ollama Python SDK: pip install layerlens[providers-ollama]")
        return 0

    endpoint = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    print(f"Wiring against Ollama at {endpoint}")
    print("(set OLLAMA_HOST to point at a remote daemon)")
    print()
    print("    client = ollama.Client(host=os.environ.get('OLLAMA_HOST', 'http://localhost:11434'))")
    print("    provider = OllamaProvider(cost_per_second=0.0001)  # optional infra-cost attribution")
    print("    provider.connect(client)")
    print("    response = client.chat(model='llama3', messages=[{'role': 'user', 'content': 'Hi'}])")
    print("    print(response['message']['content'])")
    print()

    # If the daemon isn't reachable we don't crash the sample.
    try:
        client = ollama.Client(host=endpoint)
        provider = OllamaProvider(cost_per_second=float(os.environ.get("OLLAMA_COST_PER_SECOND", "0") or 0) or None)
        provider.connect(client)
        response = client.chat(
            model=os.environ.get("OLLAMA_MODEL", "llama3"),
            messages=[{"role": "user", "content": "Say hello in one word."}],
        )
        print(f"Ollama says: {response['message']['content']}")
    except Exception as exc:  # noqa: BLE001  -- intentional: sample shouldn't hard-fail
        print(f"[ollama call skipped] {exc}")
        print("Is `ollama serve` running and the model pulled?")
    return 0


if __name__ == "__main__":
    sys.exit(main())
