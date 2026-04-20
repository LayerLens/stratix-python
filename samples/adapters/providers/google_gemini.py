"""Sample: Google Vertex AI (Gemini) adapter."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from adapters._shared import capture_events  # type: ignore[import-not-found]

from layerlens.instrument.adapters.providers.google_vertex import (
    instrument_google_vertex,
    uninstrument_google_vertex,
)


def main() -> None:
    try:
        from vertexai.generative_models import GenerativeModel  # type: ignore[import-not-found]
    except ImportError:
        print("Install the Vertex extra: pip install 'layerlens[google-vertex]'")
        return

    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") and not os.environ.get("GOOGLE_CLOUD_PROJECT"):
        print("Set GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_CLOUD_PROJECT to run against Vertex AI.")
        return

    model = GenerativeModel("gemini-1.5-flash")
    instrument_google_vertex(model)
    try:
        with capture_events("vertex"):
            resp = model.generate_content("Name a prime number.")
            print("reply:", resp.text)
    finally:
        uninstrument_google_vertex()


if __name__ == "__main__":
    main()
