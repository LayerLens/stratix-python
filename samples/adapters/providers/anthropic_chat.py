"""Sample: instrument the Anthropic SDK with thinking-token + cache-read capture."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from adapters._shared import capture_events  # type: ignore[import-not-found]

from layerlens.instrument.adapters.providers.anthropic import (
    instrument_anthropic,
    uninstrument_anthropic,
)


def main() -> None:
    try:
        from anthropic import Anthropic  # type: ignore[import-not-found]
    except ImportError:
        print("Install the Anthropic extra: pip install 'layerlens[anthropic]'")
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY to run this sample against the live API.")
        return

    client = Anthropic()
    instrument_anthropic(client)
    try:
        with capture_events("anthropic_chat"):
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=80,
                messages=[{"role": "user", "content": "Name two oceans."}],
            )
            for block in resp.content:
                if getattr(block, "type", None) == "text":
                    print("reply:", block.text)
    finally:
        uninstrument_anthropic()


if __name__ == "__main__":
    main()
