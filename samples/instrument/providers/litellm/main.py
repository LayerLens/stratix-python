"""Sample: instrument LiteLLM with the LayerLens provider adapter.

LiteLLM is a multi-provider router. The adapter installs a single
callback into ``litellm.callbacks`` and lets LiteLLM dispatch it for
every provider it routes to (OpenAI, Anthropic, Bedrock, Vertex,
Cohere, Ollama, Together, Groq, ...).

The sample is **mocked by default** — it does not require any provider
API key and never reaches the network. Set ``LAYERLENS_LITELLM_LIVE=1``
plus the appropriate vendor key (``OPENAI_API_KEY`` for the default
``openai/gpt-4o-mini`` model) to run a real round-trip through LiteLLM.

Run::

    pip install 'layerlens[providers-litellm]'
    python -m samples.instrument.providers.litellm.main
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import MagicMock

from layerlens.instrument.adapters._base import CaptureConfig
from layerlens.instrument.adapters.providers.litellm import LiteLLMAdapter


class _PrintingStratix:
    """Tiny stratix shim that prints every event the adapter emits.

    Real production usage attaches an ``HttpEventSink`` instead — see
    ``samples/instrument/openai/main.py`` for that pattern. We avoid
    making a network call here so the sample runs anywhere with no
    additional setup.
    """

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:
        if len(args) == 2 and isinstance(args[0], str):
            event_type, payload = args
            self.events.append({"event_type": event_type, "payload": payload})
            print(f"  [{event_type}] provider={payload.get('provider')!r} model={payload.get('model')!r}")


def _mocked_response(model: str) -> Any:
    """Build a LiteLLM-shaped response object for the offline path."""
    message = SimpleNamespace(role="assistant", content="hello from mock", tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop", index=0)
    usage = SimpleNamespace(prompt_tokens=12, completion_tokens=8, total_tokens=20)
    return SimpleNamespace(
        id="chatcmpl-mock-1",
        model=model,
        choices=[choice],
        usage=usage,
    )


def _run_offline(adapter: LiteLLMAdapter) -> None:
    """Drive the adapter without touching the network.

    Exercises the same callback path LiteLLM would invoke after a real
    completion: build kwargs, build a response, call
    :meth:`log_success_event` with deterministic timestamps.
    """
    print("Running offline (no provider API key required) ...")
    print("To run a live call: set LAYERLENS_LITELLM_LIVE=1 and OPENAI_API_KEY.")
    print()

    cases = [
        ("openai/gpt-4o-mini", "OpenAI via LiteLLM prefix routing"),
        ("anthropic/claude-3-5-sonnet", "Anthropic via LiteLLM prefix routing"),
        ("bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0", "Bedrock-routed Anthropic"),
        ("vertex_ai/gemini-1.5-pro", "Vertex AI Gemini"),
        ("gpt-4", "Bare model name routes to OpenAI heuristic"),
        ("claude-3-5-sonnet", "Bare model name routes to Anthropic heuristic"),
    ]

    assert adapter._callback is not None
    for model, description in cases:
        print(f"-- {description} (model={model!r})")
        adapter._callback.log_success_event(
            kwargs={
                "model": model,
                "messages": [{"role": "user", "content": "ping"}],
                "temperature": 0.7,
            },
            response_obj=_mocked_response(model),
            start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
        )


def _run_live(adapter: LiteLLMAdapter) -> int:
    """Run a real ``litellm.completion`` call. Requires a vendor API key."""
    try:
        import litellm  # type: ignore[import-not-found,unused-ignore]
    except ImportError:
        print(
            "litellm package not installed. Install with:\n"
            "    pip install 'layerlens[providers-litellm]'",
            file=sys.stderr,
        )
        return 2

    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is not set; live mode requires a vendor key.\n"
            "Run without LAYERLENS_LITELLM_LIVE for the offline sample.",
            file=sys.stderr,
        )
        return 2

    print("Running live LiteLLM completion (openai/gpt-4o-mini) ...")
    response = litellm.completion(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "What is 2 + 2?"},
        ],
        max_tokens=20,
    )
    choice = response.choices[0].message.content if response.choices else "(empty)"
    print(f"Response: {choice}")
    return 0


def main() -> int:
    stratix = _PrintingStratix()
    adapter = LiteLLMAdapter(stratix=stratix, capture_config=CaptureConfig.full())

    if os.environ.get("LAYERLENS_LITELLM_LIVE") == "1":
        adapter.connect()
        try:
            return _run_live(adapter)
        finally:
            adapter.disconnect()

    # Offline path: install a stub ``litellm`` module so ``connect`` does
    # not require the upstream package, then drive the callback directly.
    import sys as _sys

    if "litellm" not in _sys.modules:
        stub = MagicMock()
        stub.callbacks = []
        stub.success_callback = []
        stub.failure_callback = []
        stub.__version__ = "1.40.0"
        _sys.modules["litellm"] = stub

    adapter.connect()
    try:
        _run_offline(adapter)
    finally:
        adapter.disconnect()

    print()
    print(f"Captured {len(stratix.events)} events across the sample run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
