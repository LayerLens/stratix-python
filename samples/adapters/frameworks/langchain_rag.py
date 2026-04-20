"""Sample: LangChain callback handler — a tiny RAG-style chain."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from adapters._shared import capture_events  # type: ignore[import-not-found]


def main() -> None:
    try:
        from langchain_openai import ChatOpenAI  # type: ignore[import-not-found]
        from langchain_core.messages import HumanMessage  # type: ignore[import-not-found]

        from layerlens.instrument.adapters.frameworks.langchain import (
            LangChainCallbackHandler,
        )
    except ImportError:
        print("Install: pip install 'layerlens[langchain]' langchain-openai")
        return

    handler = LangChainCallbackHandler()
    llm = ChatOpenAI(model="gpt-4o-mini", callbacks=[handler])
    with capture_events("langchain_rag"):
        resp = llm.invoke([HumanMessage(content="Summarize: grass is green.")])
        print("reply:", resp.content)


if __name__ == "__main__":
    main()
