"""Runnable sample: LiteLLM + LayerLens instrumentation (LAY-3455).

Run with::

    pip install layerlens[litellm]
    python samples/instrument/litellm/example.py

See ``docs/adapters/providers/litellm.md`` for how routing works — LiteLLM
dispatches to the underlying provider by model string, and cost is resolved
from the bundled pricing manifest by the routed model.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    try:
        from layerlens.instrument.adapters.providers import LiteLLMProvider
    except ImportError as exc:
        print(f"[skipped] {exc}")
        print("Install LiteLLM deps with: pip install layerlens[litellm]")
        return 0

    print("LiteLLMProvider available.")
    try:
        import litellm  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        print(f"[skipped] {exc}")
        print("Install LiteLLM: pip install layerlens[litellm]")
        return 0

    print("Wiring against litellm.completion / litellm.acompletion")
    print()
    print("    provider = LiteLLMProvider()")
    print("    provider.connect()   # monkey-patches litellm.completion / acompletion")
    print("    resp = litellm.completion(")
    print("        model='gpt-4o-mini',   # or 'claude-3-5-sonnet', 'bedrock/anthropic.claude-3-...'")
    print("        messages=[{'role': 'user', 'content': 'Hi'}],")
    print("    )")

    # litellm.completion needs a provider key (e.g. OPENAI_API_KEY) to run for real.
    if not any(os.environ.get(k) for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY")):
        print("\n[live call skipped] set a provider key (e.g. OPENAI_API_KEY) to run for real.")
        return 0
    try:
        provider = LiteLLMProvider()
        provider.connect()
        resp = litellm.completion(
            model=os.environ.get("LITELLM_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": "Say hello in one word."}],
            max_tokens=16,
        )
        print(f"LiteLLM responded: {resp.choices[0].message.content}")
    except Exception as exc:  # noqa: BLE001 -- sample shouldn't hard-fail
        print(f"[litellm call skipped] {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
