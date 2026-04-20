"""Sample: instantiate the Salesforce Agentforce adapter.

The real connect/import flow requires live Salesforce OAuth credentials:

    SF_CLIENT_ID=... SF_CLIENT_SECRET=... SF_INSTANCE_URL=... \\
        python samples/adapters/frameworks/agentforce_import.py

Without those the sample just confirms the adapter loads and exposes
its expected surface so the import path is regression-safe.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import Mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from adapters._shared import capture_events  # type: ignore[import-not-found]


def main() -> None:
    try:
        from layerlens.instrument.adapters.frameworks.agentforce import AgentforceAdapter
    except ImportError:
        print("Install: pip install 'layerlens[agentforce]' httpx")
        return

    required = {"SF_CLIENT_ID", "SF_CLIENT_SECRET", "SF_INSTANCE_URL"}
    if not required.issubset(os.environ):
        adapter = AgentforceAdapter(client=Mock())
        with capture_events("agentforce_import"):
            info = adapter.adapter_info()
            print(f"adapter loaded: {info.name} (connected={info.connected})")
        print("Set SF_CLIENT_ID / SF_CLIENT_SECRET / SF_INSTANCE_URL to run a real import.")
        return

    adapter = AgentforceAdapter(client=Mock())
    adapter.connect(
        credentials={
            "client_id": os.environ["SF_CLIENT_ID"],
            "client_secret": os.environ["SF_CLIENT_SECRET"],
            "instance_url": os.environ["SF_INSTANCE_URL"],
        }
    )
    try:
        with capture_events("agentforce_import"):
            summary = adapter.import_sessions(limit=5)
            print("summary:", summary)
    finally:
        adapter.disconnect()


if __name__ == "__main__":
    main()
