"""Runnable sample: CrewAI + LayerLens instrumentation (LAY-3447).

Run with::

    pip install layerlens[crewai]
    python samples/instrument/crewai/example.py
"""

from __future__ import annotations

import sys
from unittest.mock import Mock


def main() -> int:
    layerlens_client = Mock(name="LayerLensClient")
    try:
        from layerlens.instrument.adapters.frameworks import CrewAIAdapter

        adapter = CrewAIAdapter(client=layerlens_client)
        adapter.connect()
    except ImportError as exc:
        print(f"[skipped] {exc}")
        print("Install CrewAI with: pip install layerlens[crewai]")
        return 0

    print(f"CrewAIAdapter connected: requires_pydantic={adapter.requires_pydantic}")
    print("The adapter is now registered on CrewAI's event bus.")
    print("Run your crew normally:")
    print()
    print("    from crewai import Agent, Crew, Task")
    print("    crew = Crew(agents=[...], tasks=[...])")
    print("    crew.kickoff()")
    print()
    print("Then call ``adapter.disconnect()`` when done.")

    adapter.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
