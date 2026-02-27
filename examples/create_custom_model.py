#!/usr/bin/env -S poetry run python

from layerlens import Stratix


def main():
    # Construct client (API key from env or inline)
    client = Stratix()

    # --- Create a custom model backed by an OpenAI-compatible API
    #
    # Custom models let you evaluate any model accessible via an
    # OpenAI-compatible chat completions endpoint.
    #
    # Key format: lowercase alphanumeric with dots, hyphens, slashes
    #   e.g. "my-org/custom-llama-3.1-70b"

    result = client.models.create_custom(
        name="My Custom Model",
        key="my-org/custom-model-v1",
        description="Custom fine-tuned model served via vLLM",
        api_url="https://my-model-endpoint.example.com/v1",
        api_key="my-provider-api-key",
        max_tokens=4096,
    )

    if result:
        print(f"Custom model created: {result.model_id}")
    else:
        print("Failed to create custom model")

    # --- Verify the model was added to the project
    models = client.models.get(type="custom")
    if models:
        print(f"\nCustom models in project ({len(models)}):")
        for m in models:
            print(f"  - {m.name} (id={m.id}, key={m.key})")


if __name__ == "__main__":
    main()
