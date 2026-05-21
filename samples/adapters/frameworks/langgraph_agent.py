"""Sample: LangGraph stateful agent — two-node graph."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from adapters._shared import capture_events  # type: ignore[import-not-found]


def main() -> None:
    try:
        from langgraph.graph import END, StateGraph  # type: ignore[import-not-found]

        from layerlens.instrument.adapters.frameworks.langgraph import (
            LangGraphCallbackHandler,
        )
    except ImportError:
        print("Install: pip install 'layerlens[langchain]' langgraph")
        return

    def classify(state: dict) -> dict:
        state["kind"] = "question" if "?" in state["text"] else "statement"
        return state

    def respond(state: dict) -> dict:
        state["reply"] = f"{state['kind']}: {state['text']}"
        return state

    builder = StateGraph(dict)
    builder.add_node("classify", classify)
    builder.add_node("respond", respond)
    builder.set_entry_point("classify")
    builder.add_edge("classify", "respond")
    builder.add_edge("respond", END)
    graph = builder.compile()

    handler = LangGraphCallbackHandler(None)
    with capture_events("langgraph_agent"):
        out = graph.invoke({"text": "Is grass green?"}, config={"callbacks": [handler]})
        print("reply:", out.get("reply"))


if __name__ == "__main__":
    main()
