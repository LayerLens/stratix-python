"""Sample: instrument the real Ollama client with the LayerLens adapter.

Runs a single ``chat`` round-trip through ``OllamaAdapter``. Every
event the adapter emits (``model.invoke``, ``cost.record``, optional
``policy.violation`` on errors) is printed to stdout via the bundled
:class:`_StdoutSink`. Swap the sink for ``HttpEventSink`` (lands with
the M2 transport PR) to ship telemetry to atlas-app.

Two execution modes:

1. **Live** — set ``LAYERLENS_OLLAMA_LIVE=1`` and have ``ollama serve``
   running locally with the requested model pulled. The sample first
   pulls the model (no-op if already cached) then runs a chat request.

2. **Mocked** (default) — uses :mod:`respx` to fake the local Ollama
   HTTP endpoints so the sample is runnable in CI / a fresh checkout
   without an Ollama daemon.

Required to run live::

    # 1. Install + start the daemon (Linux/macOS)
    curl -fsSL https://ollama.com/install.sh | sh
    ollama serve &

    # 2. Pull the model you'll call (3.8B parameters, ~2GB)
    ollama pull llama3.2:3b

    # 3. Install the adapter extra and run
    pip install 'layerlens[providers-ollama]'
    LAYERLENS_OLLAMA_LIVE=1 python -m samples.instrument.ollama.main

Run mocked::

    pip install 'layerlens[providers-ollama]' respx
    python -m samples.instrument.ollama.main
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict


class _StdoutSink:
    """Trivial event sink that prints each emitted event to stdout."""

    def __init__(self) -> None:
        self.count = 0

    def emit(self, *args: Any, **_kwargs: Any) -> None:
        if len(args) == 2 and isinstance(args[0], str):
            event_type, payload = args
            self.count += 1
            print(f"[event {self.count:>2}] {event_type}: {_summarise(payload)}")


def _summarise(payload: Dict[str, Any]) -> str:
    """Pretty-print a few key fields from the payload."""
    if not isinstance(payload, dict):
        return repr(payload)
    keys = (
        "provider",
        "model",
        "method",
        "endpoint",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "api_cost_usd",
        "infra_cost_usd",
        "latency_ms",
        "finish_reason",
        "error",
    )
    return " ".join(
        f"{k}={payload[k]!r}" for k in keys if k in payload and payload[k] is not None
    )


def _run_live() -> int:
    """Hit a real ``ollama serve`` daemon."""
    try:
        from ollama import Client, ResponseError
    except ImportError:
        print(
            "ollama package not installed. Install with:\n"
            "    pip install 'layerlens[providers-ollama]'",
            file=sys.stderr,
        )
        return 2

    from layerlens.instrument.adapters._base import CaptureConfig
    from layerlens.instrument.adapters.providers.ollama_adapter import OllamaAdapter

    model = os.environ.get("LAYERLENS_OLLAMA_MODEL", "llama3.2:3b")

    sink = _StdoutSink()
    adapter = OllamaAdapter(
        stratix=sink,
        capture_config=CaptureConfig.standard(),
        # Optional: $0.005 / GPU-second to attribute infra cost.
        cost_per_second=float(os.environ.get("LAYERLENS_OLLAMA_COST_PER_SECOND", "0")) or None,
    )
    adapter.connect()

    client = Client()
    adapter.connect_client(client)

    try:
        # Step 1: pull the model. This is a no-op if the model is
        # already cached locally; otherwise it streams the layers down.
        # (Pull is NOT instrumented — only chat / generate / embeddings
        # are wrapped.)
        print(f"Pulling model {model!r} (no-op if already cached)...")
        try:
            client.pull(model)
        except ResponseError as exc:
            print(f"Pull failed ({exc}); proceeding to chat anyway", file=sys.stderr)

        # Step 2: run a chat. This call IS instrumented.
        print(f"Chatting with {model!r}...")
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": "What is 2 + 2?"},
            ],
        )
        print(f"Response: {response.message.content}")
        if response.eval_count is not None:
            print(
                f"Tokens — prompt: {response.prompt_eval_count}, "
                f"completion: {response.eval_count}"
            )
    finally:
        adapter.disconnect()

    print(f"\nEmitted {sink.count} events.")
    return 0


def _run_mocked() -> int:
    """Run against a respx-mocked Ollama HTTP endpoint."""
    try:
        import httpx
        import respx
        from ollama import Client
    except ImportError as exc:
        print(f"Missing dependency: {exc}", file=sys.stderr)
        print(
            "Install with:\n"
            "    pip install 'layerlens[providers-ollama]' respx",
            file=sys.stderr,
        )
        return 2

    from layerlens.instrument.adapters._base import CaptureConfig
    from layerlens.instrument.adapters.providers.ollama_adapter import OllamaAdapter

    chat_body = {
        "model": "llama3.2:3b",
        "created_at": "2026-04-25T00:00:00Z",
        "message": {"role": "assistant", "content": "2 + 2 = 4"},
        "done": True,
        "done_reason": "stop",
        "total_duration": 350_000_000,
        "load_duration": 0,
        "prompt_eval_count": 18,
        "prompt_eval_duration": 100_000_000,
        "eval_count": 7,
        "eval_duration": 250_000_000,
    }

    sink = _StdoutSink()
    adapter = OllamaAdapter(
        stratix=sink,
        capture_config=CaptureConfig.standard(),
        cost_per_second=0.005,  # Demo: $0.005/sec attributed GPU rental.
    )
    adapter.connect()

    print("Running mocked Ollama chat round-trip...")
    with respx.mock(base_url="http://127.0.0.1:11434") as router:
        # Pull is mocked as a single line of NDJSON; the SDK ignores
        # streaming progress events when no callback is registered.
        router.post("/api/pull").mock(
            return_value=httpx.Response(200, content=b'{"status":"success"}\n')
        )
        router.post("/api/chat").mock(return_value=httpx.Response(200, json=chat_body))

        client = Client()
        adapter.connect_client(client)

        try:
            print("Pulling model 'llama3.2:3b' (mocked)...")
            try:
                client.pull("llama3.2:3b")
            except Exception as exc:  # noqa: BLE001
                print(f"Mocked pull surfaced: {exc}", file=sys.stderr)

            print("Chatting with mocked daemon...")
            response = client.chat(
                model="llama3.2:3b",
                messages=[
                    {"role": "system", "content": "You are a concise assistant."},
                    {"role": "user", "content": "What is 2 + 2?"},
                ],
            )
            print(f"Response: {response.message.content}")
        finally:
            adapter.disconnect()

    print(f"\nEmitted {sink.count} events.")
    return 0


def main() -> int:
    if os.environ.get("LAYERLENS_OLLAMA_LIVE") == "1":
        return _run_live()
    return _run_mocked()


if __name__ == "__main__":
    raise SystemExit(main())
