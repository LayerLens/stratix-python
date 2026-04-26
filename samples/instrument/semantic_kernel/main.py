"""Sample: instrument a Semantic Kernel prompt invocation with LayerLens.

Builds a ``Kernel`` with an OpenAI chat completion service, registers the
LayerLens filters via ``SemanticKernelAdapter.instrument_kernel``, and runs a
single ``invoke_prompt`` call. Filter callbacks emit ``agent.input`` /
``agent.output`` / ``model.invoke`` events that ship to atlas-app via
``HttpEventSink``.

Required environment:

* ``OPENAI_API_KEY`` — used by ``OpenAIChatCompletion``.
* ``LAYERLENS_STRATIX_API_KEY`` — your LayerLens API key (optional).
* ``LAYERLENS_STRATIX_BASE_URL`` — atlas-app base URL (optional).

Run::

    pip install 'layerlens[semantic-kernel,providers-openai]'
    python -m samples.instrument.semantic_kernel.main
"""

from __future__ import annotations

import os
import sys
import asyncio

from layerlens.instrument.adapters._base import CaptureConfig
from layerlens.instrument.transport.sink_http import HttpEventSink
from layerlens.instrument.adapters.frameworks.semantic_kernel import SemanticKernelAdapter


async def _run(kernel: object) -> str:
    # Imported here to keep the top-level module importable without semantic-kernel.
    from semantic_kernel.functions import KernelArguments  # type: ignore[import-not-found,unused-ignore]

    result = await kernel.invoke_prompt(  # type: ignore[attr-defined]
        prompt="Reply with just the digit. What is 2 + 2?",
        arguments=KernelArguments(),
    )
    return str(result)


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set; cannot run sample.", file=sys.stderr)
        return 2

    try:
        from semantic_kernel import Kernel
        from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
    except ImportError:
        print(
            "semantic-kernel not installed. Install with:\n"
            "    pip install 'layerlens[semantic-kernel,providers-openai]'",
            file=sys.stderr,
        )
        return 2

    sink = HttpEventSink(
        adapter_name="semantic_kernel",
        path="/telemetry/spans",
        max_batch=10,
        flush_interval_s=1.0,
    )

    adapter = SemanticKernelAdapter(capture_config=CaptureConfig.standard())
    adapter.add_sink(sink)
    adapter.connect()

    kernel = Kernel()
    kernel.add_service(OpenAIChatCompletion(ai_model_id="gpt-4o-mini"))
    adapter.instrument_kernel(kernel)

    try:
        response = asyncio.run(_run(kernel))
        print(f"Response: {response}")
    finally:
        sink.close()
        adapter.disconnect()

    print("Telemetry shipped. Check the LayerLens dashboard adapter health page.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
