"""Runnable sample: AutoGen + LayerLens instrumentation (LAY-3448).

Run with::

    pip install layerlens[autogen]
    python samples/instrument/autogen/example.py
"""

from __future__ import annotations

import sys
from unittest.mock import Mock


def main() -> int:
    layerlens_client = Mock(name="LayerLensClient")
    try:
        from layerlens.instrument.adapters.frameworks import AutoGenAdapter

        adapter = AutoGenAdapter(client=layerlens_client)
        adapter.connect()
    except ImportError as exc:
        print(f"[skipped] {exc}")
        print("Install AutoGen with: pip install layerlens[autogen]")
        return 0

    print("AutoGenAdapter connected.")
    print("Build your AutoGen agents and run them as usual:")
    print()
    print("    from autogen_agentchat.agents import AssistantAgent")
    print("    from autogen_agentchat.teams import RoundRobinGroupChat")
    print("    team = RoundRobinGroupChat([agent_a, agent_b])")
    print("    await team.run(task='...')")
    print()
    print("Then call ``adapter.disconnect()`` when done.")

    adapter.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
