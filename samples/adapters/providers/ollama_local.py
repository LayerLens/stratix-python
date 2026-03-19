"""
Ollama Local Model Chat and Embedding with STRATIX Instrumentation

Demonstrates:
- Local model chat completion via Ollama
- Local model embeddings
- Streaming chat
- STRATIX event emission with zero cost (local inference)

Requirements:
    pip install ollama layerlens
    # Ollama must be running locally: https://ollama.com
    # Pull a model first: ollama pull llama3.2

Usage:
    python ollama_local.py
    python ollama_local.py --model llama3.2 --stream
    python ollama_local.py --embedding --prompt "Hello world"
"""

from __future__ import annotations

import argparse
import sys
import time

from layerlens.instrument import STRATIX, emit_model_invoke, record_token_cost


def simple_chat(client, model: str, prompt: str) -> None:
    """Non-streaming local chat with instrumentation."""
    print(f"\n--- Ollama Chat ({model}) ---")
    start = time.perf_counter()

    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0.7, "num_predict": 256},
    )

    latency_ms = (time.perf_counter() - start) * 1000
    content = response.get("message", {}).get("content", "")
    prompt_tokens = response.get("prompt_eval_count", 0)
    completion_tokens = response.get("eval_count", 0)

    emit_model_invoke(
        provider="ollama",
        name=model,
        parameters={"temperature": 0.7, "num_predict": 256},
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        latency_ms=latency_ms,
    )
    # Local inference: zero cost
    record_token_cost(
        provider="ollama",
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_per_1k_prompt=0.0,
        cost_per_1k_completion=0.0,
    )

    print(f"Response: {content}")
    print(f"Tokens: {prompt_tokens} in / {completion_tokens} out | {latency_ms:.0f}ms | Cost: $0.00")


def streaming_chat(client, model: str, prompt: str) -> None:
    """Streaming local chat with instrumentation."""
    print(f"\n--- Streaming Ollama Chat ({model}) ---")
    start = time.perf_counter()

    stream = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0.5, "num_predict": 512},
        stream=True,
    )

    collected: list[str] = []
    prompt_tokens = 0
    completion_tokens = 0

    print("Streaming: ", end="", flush=True)
    for chunk in stream:
        text = chunk.get("message", {}).get("content", "")
        if text:
            collected.append(text)
            print(text, end="", flush=True)
        if chunk.get("done"):
            prompt_tokens = chunk.get("prompt_eval_count", 0)
            completion_tokens = chunk.get("eval_count", 0)
    print()

    latency_ms = (time.perf_counter() - start) * 1000

    emit_model_invoke(
        provider="ollama",
        name=model,
        parameters={"temperature": 0.5, "num_predict": 512, "stream": True},
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        latency_ms=latency_ms,
    )
    record_token_cost(
        provider="ollama",
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_per_1k_prompt=0.0,
        cost_per_1k_completion=0.0,
    )

    print(f"Tokens: {prompt_tokens} in / {completion_tokens} out | {latency_ms:.0f}ms | Cost: $0.00")


def embedding(client, model: str, prompt: str) -> None:
    """Generate embeddings with instrumentation."""
    print(f"\n--- Ollama Embedding ({model}) ---")
    start = time.perf_counter()

    response = client.embed(model=model, input=prompt)
    latency_ms = (time.perf_counter() - start) * 1000

    embeddings = response.get("embeddings", [[]])
    dim = len(embeddings[0]) if embeddings else 0
    prompt_tokens = response.get("prompt_eval_count", 0)

    emit_model_invoke(
        provider="ollama",
        name=model,
        parameters={"request_type": "embedding"},
        prompt_tokens=prompt_tokens,
        completion_tokens=0,
        total_tokens=prompt_tokens,
        latency_ms=latency_ms,
    )

    print(f"Embedding dimension: {dim}")
    print(f"First 5 values: {embeddings[0][:5]}")
    print(f"Tokens: {prompt_tokens} in | {latency_ms:.0f}ms | Cost: $0.00")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ollama local model with STRATIX instrumentation")
    parser.add_argument("--model", default="llama3.2", help="Ollama model name")
    parser.add_argument("--prompt", default="Explain the difference between TCP and UDP.", help="User prompt")
    parser.add_argument("--stream", action="store_true", help="Use streaming mode")
    parser.add_argument("--embedding", action="store_true", help="Generate embeddings instead of chat")
    args = parser.parse_args()

    try:
        import ollama
    except ImportError:
        print("Error: Install ollama: pip install ollama", file=sys.stderr)
        sys.exit(1)

    client = ollama.Client()

    stratix = STRATIX(
        policy_ref="stratix-policy-samples@1.0.0",
        agent_id="ollama-demo",
        framework="ollama",
    )
    ctx = stratix.start_trial()

    try:
        if args.embedding:
            embedding(client, args.model, args.prompt)
        elif args.stream:
            streaming_chat(client, args.model, args.prompt)
        else:
            simple_chat(client, args.model, args.prompt)
    finally:
        summary = stratix.end_trial()
        events = stratix.get_events()
        print(f"\n--- Trace Summary ---")
        print(f"Status: {summary.get('status')}  |  Events emitted: {len(events)}")


if __name__ == "__main__":
    main()
