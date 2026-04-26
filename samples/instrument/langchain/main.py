"""Sample: instrument LangChain with the LayerLens callback handler.

Runs a single LCEL chain (prompt | llm) with ``LayerLensCallbackHandler``
installed on the chain. Every LLM/tool/chain callback fires a LayerLens
event that ships to atlas-app via ``HttpEventSink``.

Required environment:

* ``OPENAI_API_KEY`` — used by the underlying ``ChatOpenAI`` model.
* ``LAYERLENS_STRATIX_API_KEY`` — your LayerLens API key (optional).
* ``LAYERLENS_STRATIX_BASE_URL`` — atlas-app base URL (optional).

Run::

    pip install 'layerlens[langchain,providers-openai]'
    python -m samples.instrument.langchain.main
"""

from __future__ import annotations

import os
import sys

from layerlens.instrument.adapters._base import CaptureConfig
from layerlens.instrument.transport.sink_http import HttpEventSink
from layerlens.instrument.adapters.frameworks.langchain import LayerLensCallbackHandler


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set; cannot run sample.", file=sys.stderr)
        return 2

    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
    except ImportError:
        print(
            "langchain / langchain-openai not installed. Install with:\n"
            "    pip install 'layerlens[langchain,providers-openai]' langchain-openai",
            file=sys.stderr,
        )
        return 2

    sink = HttpEventSink(
        adapter_name="langchain",
        path="/telemetry/spans",
        max_batch=10,
        flush_interval_s=1.0,
    )

    handler = LayerLensCallbackHandler(capture_config=CaptureConfig.standard())
    handler.add_sink(sink)
    handler.connect()

    try:
        llm = ChatOpenAI(model="gpt-4o-mini", max_tokens=20, callbacks=[handler])
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are concise."),
                ("user", "{question}"),
            ],
        )
        chain = prompt | llm

        result = chain.invoke(
            {"question": "What is 2 + 2?"},
            config={"callbacks": [handler]},
        )

        text = getattr(result, "content", str(result))
        print(f"Response: {text}")
        print(f"Events captured: {len(handler.get_events())}")
    finally:
        sink.close()
        handler.disconnect()

    print("Telemetry shipped. Check the LayerLens dashboard adapter health page.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
