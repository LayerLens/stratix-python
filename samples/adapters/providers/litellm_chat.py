"""Sample: LiteLLM multi-provider proxy adapter."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from adapters._shared import capture_events  # type: ignore[import-not-found]

from layerlens.instrument.adapters.providers.litellm import instrument_litellm, uninstrument_litellm


def main() -> None:
    try:
        import litellm  # type: ignore[import-not-found]
    except ImportError:
        print("Install the LiteLLM extra: pip install 'layerlens[litellm]'")
        return

    # LiteLLM proxies many providers — default route is OpenAI, so require that key.
    if not any(os.environ.get(k) for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LITELLM_API_KEY")):
        print("Set OPENAI_API_KEY (or another provider key) to run LiteLLM live.")
        return

    instrument_litellm()
    try:
        with capture_events("litellm"):
            resp = litellm.completion(
                model=os.environ.get("LITELLM_MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": "Name a star."}],
                max_tokens=20,
            )
            print("reply:", resp["choices"][0]["message"]["content"])
    finally:
        uninstrument_litellm()


if __name__ == "__main__":
    main()
