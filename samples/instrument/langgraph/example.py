"""Runnable sample: LangGraph + LayerLens instrumentation (LAY-3446).

Run with::

    pip install layerlens[langgraph]
    python samples/instrument/langgraph/example.py
"""

from __future__ import annotations

import sys
from unittest.mock import Mock


def main() -> int:
    layerlens_client = Mock(name="LayerLensClient")
    try:
        from layerlens.instrument.adapters.frameworks import LangGraphCallbackHandler

        handler = LangGraphCallbackHandler(client=layerlens_client)
    except ImportError as exc:
        print(f"[skipped] {exc}")
        print("Install LangGraph with: pip install layerlens[langgraph]")
        return 0

    print(f"LangGraphCallbackHandler ready: requires_pydantic={handler.requires_pydantic}")
    print("Pass `handler` into your LangGraph compiled graph via config={'callbacks': [handler]}:")
    print()
    print("    from langgraph.graph import StateGraph")
    print("    graph = StateGraph(state_schema=MyState)")
    print("    app = graph.compile()")
    print("    app.invoke(initial_state, config={'callbacks': [handler]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
