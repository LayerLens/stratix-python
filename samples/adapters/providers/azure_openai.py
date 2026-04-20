"""Sample: Azure OpenAI adapter — captures ``azure_deployment`` metadata and Azure pricing.

Env:
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from adapters._shared import capture_events  # type: ignore[import-not-found]

from layerlens.instrument.adapters.providers.azure_openai import (
    instrument_azure_openai,
    uninstrument_azure_openai,
)


def main() -> None:
    try:
        from openai import AzureOpenAI  # type: ignore[import-not-found]
    except ImportError:
        print("Install the Azure extra: pip install 'layerlens[azure]'")
        return

    required = {"AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"}
    if not required.issubset(os.environ):
        print("Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY to run against Azure.")
        return

    client = AzureOpenAI(
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    )
    instrument_azure_openai(client)
    try:
        with capture_events("azure_openai"):
            resp = client.chat.completions.create(
                model=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
                messages=[{"role": "user", "content": "Name a planet."}],
                max_tokens=10,
            )
            print("reply:", resp.choices[0].message.content)
    finally:
        uninstrument_azure_openai()


if __name__ == "__main__":
    main()
