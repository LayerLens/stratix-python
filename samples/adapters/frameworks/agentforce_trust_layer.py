"""Sample: Agentforce Trust Layer — capture masking/grounding events."""

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
    with capture_events("agentforce_trust_layer"):
        # Illustrative: submit a prompt that exercises masking + grounding.
        out = adapter.invoke_with_trust_layer(  # type: ignore[attr-defined]
            agent_id="0XxAg00000Example",
            message="Summarise the account record for ACME Corp (contact: alice@example.com).",
        )
        print("masked_input:", out.get("masked_input"))
        print("grounded_output:", out.get("grounded_output"))


if __name__ == "__main__":
    main()
