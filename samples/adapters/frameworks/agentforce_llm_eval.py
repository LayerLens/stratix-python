"""Sample: Agentforce LLM-evaluation run — imports an eval trace set."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from adapters._shared import capture_events  # type: ignore[import-not-found]

from layerlens.instrument.adapters.frameworks.agentforce import AgentforceAdapter


def main() -> None:
    creds = {
        "client_id": os.environ.get("SF_CLIENT_ID", ""),
        "client_secret": os.environ.get("SF_CLIENT_SECRET", ""),
        "instance_url": os.environ.get("SF_INSTANCE_URL", ""),
    }
    if not creds["client_id"]:
        print("Set SF_CLIENT_ID / SF_CLIENT_SECRET / SF_INSTANCE_URL to run against a live org.")
        return

    adapter = AgentforceAdapter(None)
    adapter.connect(creds)
    with capture_events("agentforce_llm_eval"):
        # Illustrative: pull recent LLM-evaluation traces and replay them through the adapter.
        traces = adapter.fetch_llm_eval_runs(limit=3)  # type: ignore[attr-defined]
        for t in traces:
            print("eval:", t.get("id"), t.get("score"))


if __name__ == "__main__":
    main()
