"""Sample: instrument an OpenAI client and capture ``model.invoke`` + ``cost.record``.

Run with a real OpenAI key:
    OPENAI_API_KEY=sk-... python samples/adapters/providers/openai_chat.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from adapters._shared import capture_events  # type: ignore[import-not-found]

from layerlens.instrument.adapters.providers.openai import instrument_openai, uninstrument_openai


def main() -> None:
    try:
        from openai import OpenAI  # type: ignore[import-not-found]
    except ImportError:
        print("Install the OpenAI extra: pip install 'layerlens[openai]'")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run this sample against the live API.")
        return

    client = OpenAI()
    instrument_openai(client)
    try:
        with capture_events("openai_chat"):
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Say hi in five words."}],
                max_tokens=20,
            )
            print("reply:", resp.choices[0].message.content)
    finally:
        uninstrument_openai()


if __name__ == "__main__":
    main()
