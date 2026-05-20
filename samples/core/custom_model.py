"""
Custom Model Registration -- LayerLens Python SDK Sample
========================================================

Demonstrates registering a custom model backed by an
OpenAI-compatible chat-completions endpoint.

Custom models let you evaluate any model accessible via a
``/v1/chat/completions``-style API -- self-hosted vLLM instances,
fine-tuned models behind a gateway, etc.

Prerequisites
-------------
* ``pip install layerlens --index-url https://sdk.layerlens.ai/package``
* Set ``LAYERLENS_STRATIX_API_KEY`` environment variable

Usage
-----
::

    export LAYERLENS_STRATIX_API_KEY=your-api-key
    python custom_model.py
"""

from __future__ import annotations

from layerlens import Stratix


def main() -> None:
    client = Stratix()

    # ── Create a custom model ─────────────────────────────────────────
    #
    # Key format: lowercase alphanumeric with dots, hyphens, slashes
    #   e.g. "my-org/custom-llama-3.1-70b"

    result = client.models.create_custom(
        name="My Custom Model",
        key="my-org/custom-model-v1",
        description="Custom fine-tuned model served via vLLM",
        api_url="https://my-model-endpoint.example.com/v1/chat/completions",
        api_key="my-provider-api-key",
        max_tokens=4096,
        # Optional -- merged into every outgoing request body. Customer
        # values win on conflict with our hardcoded defaults (we send
        # `temperature: 0` for reproducible evals; override here for
        # providers that reject it, e.g. {"temperature": 1}).
        extra_payload={"top_p": 0.9},
    )

    if result:
        print(f"Custom model created: {result.model_id}")
    else:
        print("Failed to create custom model")

    # ── Verify the model was added to the project ─────────────────────
    models = client.models.get(type="custom")
    if models:
        print(f"\nCustom models in project ({len(models)}):")
        for m in models:
            print(f"  - {m.name} (id={m.id}, key={m.key})")
    else:
        print("\nNo custom models found in project")

    if not result:
        return

    # ── Update mutable fields (e.g. repoint api_url) ──────────────────
    #
    # Use this when your endpoint URL changes -- common for vLLM
    # instances served behind cloudflared tunnels whose hostname
    # rotates between sessions.

    updated = client.models.update_custom(
        result.model_id,
        api_url="https://my-new-endpoint.example.com/v1/chat/completions",
    )
    if updated:
        print(f"\nCustom model {result.model_id} api_url updated")

    # ── Full teardown ─────────────────────────────────────────────────
    #
    # ``delete_custom`` disables the model, removes it from
    # ``Project.Models``, and releases its name for reuse. Evaluation
    # references to the disabled record are preserved.

    deleted = client.models.delete_custom(result.model_id)
    if deleted:
        print(f"Custom model {result.model_id} deleted")


if __name__ == "__main__":
    main()
