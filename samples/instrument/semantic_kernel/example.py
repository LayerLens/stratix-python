"""Runnable sample: Semantic Kernel + LayerLens instrumentation (LAY-3450).

Run with::

    pip install layerlens[semantic-kernel]
    python samples/instrument/semantic_kernel/example.py

Note: Semantic Kernel requires Python 3.10+.
"""

from __future__ import annotations

import sys
from unittest.mock import Mock


def main() -> int:
    layerlens_client = Mock(name="LayerLensClient")
    try:
        from layerlens.instrument.adapters.frameworks import SemanticKernelAdapter

        adapter = SemanticKernelAdapter(client=layerlens_client)
    except ImportError as exc:
        print(f"[skipped] {exc}")
        print("Install Semantic Kernel with: pip install layerlens[semantic-kernel] (Python 3.10+)")
        return 0

    try:
        from semantic_kernel import Kernel  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        print(f"[skipped] {exc}")
        print("Install Semantic Kernel with: pip install layerlens[semantic-kernel] (Python 3.10+)")
        return 0

    kernel = Kernel()
    adapter.connect(target=kernel)
    print("SemanticKernelAdapter connected to a fresh Kernel.")
    print("Register your chat service and invoke as usual:")
    print()
    print("    from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion")
    print("    kernel.add_service(OpenAIChatCompletion(service_id='gpt4', ai_model_id='gpt-4o'))")
    print("    result = await kernel.invoke_prompt('Hello!')")
    print()
    print("Then call ``adapter.disconnect()`` when done.")

    adapter.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
