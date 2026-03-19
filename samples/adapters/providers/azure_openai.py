"""
Azure OpenAI GPT Chat with STRATIX Instrumentation

Demonstrates:
- Azure OpenAI chat completion with Azure-specific configuration
- Streaming chat completion
- Content safety filter detection in responses
- STRATIX event emission for model invocations and cost tracking

Requirements:
    pip install openai layerlens

Usage:
    export AZURE_OPENAI_API_KEY=...
    export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
    export AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
    python azure_openai.py
    python azure_openai.py --stream --prompt "Tell me a story"
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from layerlens.instrument import STRATIX, emit_model_invoke, record_token_cost


def simple_chat(client, deployment: str, prompt: str) -> None:
    """Non-streaming Azure OpenAI chat with content safety awareness."""
    print(f"\n--- Simple Chat ({deployment}) ---")
    start = time.perf_counter()

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": "You are a helpful, safe assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=256,
    )

    latency_ms = (time.perf_counter() - start) * 1000
    choice = response.choices[0]
    usage = response.usage

    # Azure content safety: check finish_reason for content_filter
    if choice.finish_reason == "content_filter":
        print("Warning: Response was filtered by Azure content safety.")
    content = choice.message.content or "(filtered)"

    emit_model_invoke(
        provider="azure_openai",
        name=deployment,
        parameters={"temperature": 0.7, "max_tokens": 256},
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        latency_ms=latency_ms,
    )
    record_token_cost(
        provider="azure_openai",
        model=deployment,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
    )

    print(f"Response: {content}")
    print(f"Finish reason: {choice.finish_reason}")
    print(f"Tokens: {usage.prompt_tokens} in / {usage.completion_tokens} out | {latency_ms:.0f}ms")


def streaming_chat(client, deployment: str, prompt: str) -> None:
    """Streaming Azure OpenAI chat with instrumentation."""
    print(f"\n--- Streaming Chat ({deployment}) ---")
    start = time.perf_counter()

    stream = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": "You are a helpful, safe assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
        max_tokens=512,
        stream=True,
        stream_options={"include_usage": True},
    )

    collected: list[str] = []
    finish_reason = None
    final_usage = None

    print("Streaming: ", end="", flush=True)
    for chunk in stream:
        if chunk.usage:
            final_usage = chunk.usage
        for choice in chunk.choices:
            if choice.finish_reason:
                finish_reason = choice.finish_reason
            if choice.delta.content:
                collected.append(choice.delta.content)
                print(choice.delta.content, end="", flush=True)
    print()

    latency_ms = (time.perf_counter() - start) * 1000
    prompt_tokens = getattr(final_usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(final_usage, "completion_tokens", 0) or 0

    if finish_reason == "content_filter":
        print("Warning: Response was filtered by Azure content safety.")

    emit_model_invoke(
        provider="azure_openai",
        name=deployment,
        parameters={"temperature": 0.5, "max_tokens": 512, "stream": True},
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        latency_ms=latency_ms,
    )
    record_token_cost(
        provider="azure_openai",
        model=deployment,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )

    print(f"Finish reason: {finish_reason}")
    print(f"Tokens: {prompt_tokens} in / {completion_tokens} out | {latency_ms:.0f}ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="Azure OpenAI chat with STRATIX instrumentation")
    parser.add_argument("--deployment", default=None, help="Deployment name (or set AZURE_OPENAI_DEPLOYMENT)")
    parser.add_argument("--prompt", default="What are three benefits of cloud computing?", help="User prompt")
    parser.add_argument("--stream", action="store_true", help="Use streaming mode")
    args = parser.parse_args()

    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    deployment = args.deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")

    if not api_key or not endpoint:
        print(
            "Error: Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables.",
            file=sys.stderr,
        )
        sys.exit(1)

    from openai import AzureOpenAI

    client = AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version="2024-10-21",
    )

    stratix = STRATIX(
        policy_ref="stratix-policy-samples@1.0.0",
        agent_id="azure-openai-demo",
        framework="azure_openai",
    )
    ctx = stratix.start_trial()

    try:
        if args.stream:
            streaming_chat(client, deployment, args.prompt)
        else:
            simple_chat(client, deployment, args.prompt)
    finally:
        summary = stratix.end_trial()
        events = stratix.get_events()
        print(f"\n--- Trace Summary ---")
        print(f"Status: {summary.get('status')}  |  Events emitted: {len(events)}")


if __name__ == "__main__":
    main()
