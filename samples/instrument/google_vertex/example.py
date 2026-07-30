"""Runnable sample: Google Vertex AI + LayerLens instrumentation (LAY-3453).

Run with::

    pip install layerlens[providers-vertex]
    python samples/instrument/google_vertex/example.py

See ``docs/adapters/providers/google_vertex.md`` for the Service Account JSON
and Application Default Credentials (ADC) setup that authenticates the SDK.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import Mock


def main() -> int:
    layerlens_client = Mock(name="LayerLensClient")
    try:
        from layerlens.instrument.adapters.providers import GoogleVertexProvider
    except ImportError as exc:
        print(f"[skipped] {exc}")
        print("Install Vertex deps with: pip install layerlens[providers-vertex]")
        return 0

    print("GoogleVertexProvider available.")
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        print("[skipped] GOOGLE_CLOUD_PROJECT not set; printing wiring sketch only.")
        print()
        print("    import vertexai")
        print("    from vertexai.generative_models import GenerativeModel")
        print("    vertexai.init(project=os.environ['GOOGLE_CLOUD_PROJECT'], location='us-central1')")
        print("    model = GenerativeModel('gemini-2.5-pro')")
        print("    provider = GoogleVertexProvider()")
        print("    provider.connect(model)")
        print("    print(model.generate_content('Hello!').text)")
        return 0

    try:
        import vertexai  # pyright: ignore[reportMissingImports]
        from vertexai.generative_models import GenerativeModel  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        print(f"[skipped] {exc}")
        print("Install Vertex deps with: pip install layerlens[providers-vertex]")
        return 0

    vertexai.init(project=project, location=os.environ.get("VERTEX_LOCATION", "us-central1"))
    model = GenerativeModel(os.environ.get("VERTEX_MODEL", "gemini-2.5-flash"))
    provider = GoogleVertexProvider()
    provider.connect(model)

    response = model.generate_content("Say hello in one word.")
    print(f"Vertex says: {response.text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
