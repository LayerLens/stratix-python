"""Runnable sample: Azure OpenAI + LayerLens instrumentation (LAY-3451).

Run with::

    pip install layerlens[azure]
    python samples/instrument/azure_openai/example.py

See ``docs/adapters/providers/azure_openai.md`` for deployment setup
(endpoint, ``OPENAI_API_VERSION``, deployment name).
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    try:
        from layerlens.instrument.adapters.providers import AzureOpenAIProvider
    except ImportError as exc:
        print(f"[skipped] {exc}")
        print("Install Azure deps with: pip install layerlens[azure]")
        return 0

    print("AzureOpenAIProvider available.")
    try:
        from openai import AzureOpenAI  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        print(f"[skipped] {exc}")
        print("Install the OpenAI SDK: pip install layerlens[azure]")
        return 0

    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "https://<resource>.openai.azure.com")
    print(f"Wiring against Azure OpenAI endpoint {endpoint}")
    print()
    print("    client = AzureOpenAI(")
    print("        azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],")
    print("        api_key=os.environ['AZURE_OPENAI_API_KEY'],")
    print("        api_version=os.environ.get('OPENAI_API_VERSION', '2024-06-01'),")
    print("    )")
    print("    provider = AzureOpenAIProvider()")
    print("    provider.connect(client)   # monkey-patches chat.completions/responses/embeddings")
    print("    resp = client.chat.completions.create(")
    print("        model='<your-deployment-name>',")
    print("        messages=[{'role': 'user', 'content': 'Hi'}],")
    print("    )")

    if not os.environ.get("AZURE_OPENAI_API_KEY"):
        print("\n[live call skipped] set AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY to run for real.")
        return 0
    try:
        client = AzureOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ.get("OPENAI_API_VERSION", "2024-06-01"),
        )
        provider = AzureOpenAIProvider()
        provider.connect(client)
        resp = client.chat.completions.create(
            model=os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
            messages=[{"role": "user", "content": "Say hello in one word."}],
        )
        print(f"Azure OpenAI says: {resp.choices[0].message.content}")
    except Exception as exc:  # noqa: BLE001 -- sample shouldn't hard-fail
        print(f"[azure call skipped] {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
