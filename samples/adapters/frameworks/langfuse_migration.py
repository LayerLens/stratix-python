"""Sample: Langfuse -> LayerLens trace migration adapter.

Real usage requires a Langfuse deployment:

    LANGFUSE_PUBLIC_KEY=... LANGFUSE_SECRET_KEY=... LANGFUSE_HOST=... \\
        python samples/adapters/frameworks/langfuse_migration.py

Without those the sample just confirms the adapter loads.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import Mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from adapters._shared import capture_events  # type: ignore[import-not-found]


def main() -> None:
    try:
        from layerlens.instrument.adapters.frameworks.langfuse import LangfuseAdapter
    except ImportError:
        print("Install: pip install 'layerlens[langfuse]' httpx")
        return

    required = {"LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"}
    if not required.issubset(os.environ):
        adapter = LangfuseAdapter(client=Mock())
        with capture_events("langfuse_migration"):
            info = adapter.adapter_info()
            print(f"adapter loaded: {info.name} (connected={info.connected})")
        print("Set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST to migrate real traces.")
        return

    adapter = LangfuseAdapter(client=Mock())
    adapter.connect(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        host=os.environ["LANGFUSE_HOST"],
    )
    try:
        with capture_events("langfuse_migration"):
            summary = adapter.import_traces(limit=5)
            print("summary:", summary)
    finally:
        adapter.disconnect()


if __name__ == "__main__":
    main()
