"""Sample: instrument a LlamaIndex chat call with the LayerLens adapter.

Registers the LayerLens event handler with the global LlamaIndex
``Dispatcher`` via ``LlamaIndexAdapter.instrument_workflow``, then runs a
single LLM ``chat`` call. The handler emits ``model.invoke`` (and any
``tool.call`` / ``agent.*`` events) which ship to atlas-app via
``HttpEventSink``.

Required environment:

* ``OPENAI_API_KEY`` — used by ``llama_index.llms.openai.OpenAI``.
* ``LAYERLENS_STRATIX_API_KEY`` — your LayerLens API key (optional).
* ``LAYERLENS_STRATIX_BASE_URL`` — atlas-app base URL (optional).

Run::

    pip install 'layerlens[llama-index,providers-openai]' llama-index-llms-openai
    python -m samples.instrument.llama_index.main
"""

from __future__ import annotations

import os
import sys

from layerlens.instrument.adapters._base import CaptureConfig
from layerlens.instrument.transport.sink_http import HttpEventSink
from layerlens.instrument.adapters.frameworks.llama_index import LlamaIndexAdapter


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set; cannot run sample.", file=sys.stderr)
        return 2

    try:
        from llama_index.core.llms import ChatMessage, MessageRole
        from llama_index.llms.openai import OpenAI as LlamaOpenAI
    except ImportError:
        print(
            "llama-index not installed. Install with:\n"
            "    pip install 'layerlens[llama-index,providers-openai]'"
            " llama-index-llms-openai",
            file=sys.stderr,
        )
        return 2

    sink = HttpEventSink(
        adapter_name="llama_index",
        path="/telemetry/spans",
        max_batch=10,
        flush_interval_s=1.0,
    )

    adapter = LlamaIndexAdapter(capture_config=CaptureConfig.standard())
    adapter.add_sink(sink)
    adapter.connect()
    adapter.instrument_workflow(None)  # global event handler registration

    llm = LlamaOpenAI(model="gpt-4o-mini", max_tokens=20)

    try:
        response = llm.chat(
            [
                ChatMessage(role=MessageRole.SYSTEM, content="Be concise."),
                ChatMessage(role=MessageRole.USER, content="What is 2 + 2?"),
            ]
        )
        text = getattr(response.message, "content", str(response))
        print(f"Response: {text}")
    finally:
        sink.close()
        adapter.disconnect()

    print("Telemetry shipped. Check the LayerLens dashboard adapter health page.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
