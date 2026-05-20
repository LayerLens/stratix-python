"""Runnable sample: Agentforce (Salesforce) + LayerLens instrumentation (LAY-3449).

Run with::

    pip install layerlens[agentforce]
    python samples/instrument/agentforce/example.py

See ``docs/adapters/frameworks/agentforce.md`` for the Connected App / OAuth
setup that produces ``SF_CLIENT_ID``, ``SF_CLIENT_SECRET``, and
``SF_INSTANCE_URL``.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import Mock


def main() -> int:
    layerlens_client = Mock(name="LayerLensClient")
    try:
        from layerlens.instrument.adapters.frameworks import AgentforceAdapter

        adapter = AgentforceAdapter(client=layerlens_client)
    except ImportError as exc:
        print(f"[skipped] {exc}")
        print("Install Agentforce deps with: pip install layerlens[agentforce]")
        return 0

    print("AgentforceAdapter constructed.")
    required = ("SF_CLIENT_ID", "SF_CLIENT_SECRET", "SF_INSTANCE_URL")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        print(f"[skipped] missing env vars: {', '.join(missing)}")
        print("Set these from your Salesforce Connected App and re-run to import live sessions.")
        print()
        print("    adapter.connect(credentials={")
        print("        'client_id':     os.environ['SF_CLIENT_ID'],")
        print("        'client_secret': os.environ['SF_CLIENT_SECRET'],")
        print("        'instance_url':  os.environ['SF_INSTANCE_URL'],")
        print("    })")
        print("    summary = adapter.import_sessions(limit=10)")
        print("    adapter.disconnect()")
        return 0

    adapter.connect(
        credentials={
            "client_id": os.environ["SF_CLIENT_ID"],
            "client_secret": os.environ["SF_CLIENT_SECRET"],
            "instance_url": os.environ["SF_INSTANCE_URL"],
        },
    )
    try:
        summary = adapter.import_sessions(limit=10)
        print(
            f"Imported {summary['sessions_imported']} sessions "
            f"({summary['events_emitted']} events, {summary['errors']} errors). "
            f"next_cursor={summary['next_cursor']}"
        )
    finally:
        adapter.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
