"""
Google Gemini Text and Multimodal Generation with STRATIX Instrumentation

Demonstrates:
- Text generation with Gemini models
- Multimodal generation (text + image via URL)
- Streaming generation
- STRATIX event emission for model invocations and cost tracking

Requirements:
    pip install google-generativeai layerlens Pillow

Usage:
    export GOOGLE_API_KEY=AI...
    python google_gemini.py
    python google_gemini.py --model gemini-2.0-flash --stream
    python google_gemini.py --multimodal --image-url "https://example.com/photo.jpg"
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from layerlens.instrument import STRATIX, emit_model_invoke, record_token_cost


def text_generation(model_obj, model_name: str, prompt: str, stream: bool) -> None:
    """Text generation with optional streaming."""
    label = "Streaming" if stream else "Simple"
    print(f"\n--- {label} Text Generation ({model_name}) ---")
    start = time.perf_counter()

    if stream:
        response_stream = model_obj.generate_content(prompt, stream=True)
        collected: list[str] = []
        print("Streaming: ", end="", flush=True)
        for chunk in response_stream:
            if chunk.text:
                collected.append(chunk.text)
                print(chunk.text, end="", flush=True)
        print()
        # Resolve to get final metadata; usage_metadata is on the last chunk
        latency_ms = (time.perf_counter() - start) * 1000
        usage = getattr(chunk, "usage_metadata", None)
    else:
        response = model_obj.generate_content(prompt)
        latency_ms = (time.perf_counter() - start) * 1000
        print(f"Response: {response.text}")
        usage = getattr(response, "usage_metadata", None)

    prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
    completion_tokens = getattr(usage, "candidates_token_count", 0) or 0
    total_tokens = getattr(usage, "total_token_count", 0) or (prompt_tokens + completion_tokens)

    emit_model_invoke(
        provider="google",
        name=model_name,
        parameters={"stream": stream},
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        latency_ms=latency_ms,
    )
    record_token_cost(
        provider="google",
        model=model_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )

    print(f"Tokens: {prompt_tokens} in / {completion_tokens} out | {latency_ms:.0f}ms")


def multimodal_generation(model_obj, model_name: str, prompt: str, image_url: str) -> None:
    """Multimodal generation with text + image."""
    print(f"\n--- Multimodal Generation ({model_name}) ---")
    import urllib.request
    from io import BytesIO

    try:
        from PIL import Image
    except ImportError:
        print("Error: Install Pillow for multimodal demo: pip install Pillow", file=sys.stderr)
        sys.exit(1)

    print(f"Fetching image: {image_url}")
    with urllib.request.urlopen(image_url, timeout=15) as resp:
        img = Image.open(BytesIO(resp.read()))

    start = time.perf_counter()
    response = model_obj.generate_content([prompt, img])
    latency_ms = (time.perf_counter() - start) * 1000

    print(f"Response: {response.text}")

    usage = getattr(response, "usage_metadata", None)
    prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
    completion_tokens = getattr(usage, "candidates_token_count", 0) or 0

    emit_model_invoke(
        provider="google",
        name=model_name,
        parameters={"multimodal": True},
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        latency_ms=latency_ms,
    )
    record_token_cost(
        provider="google",
        model=model_name,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )

    print(f"Tokens: {prompt_tokens} in / {completion_tokens} out | {latency_ms:.0f}ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="Google Gemini with STRATIX instrumentation")
    parser.add_argument("--model", default="gemini-2.0-flash", help="Model name")
    parser.add_argument("--prompt", default="Summarize the key ideas behind general relativity.", help="User prompt")
    parser.add_argument("--stream", action="store_true", help="Use streaming mode")
    parser.add_argument("--multimodal", action="store_true", help="Run multimodal demo")
    parser.add_argument("--image-url", default="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg",
                        help="Image URL for multimodal demo")
    args = parser.parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Error: Set GOOGLE_API_KEY environment variable.", file=sys.stderr)
        sys.exit(1)

    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model_obj = genai.GenerativeModel(args.model)

    stratix = STRATIX(
        policy_ref="stratix-policy-samples@1.0.0",
        agent_id="gemini-demo",
        framework="google-genai",
    )
    ctx = stratix.start_trial()

    try:
        if args.multimodal:
            multimodal_generation(model_obj, args.model, args.prompt, args.image_url)
        else:
            text_generation(model_obj, args.model, args.prompt, args.stream)
    finally:
        summary = stratix.end_trial()
        events = stratix.get_events()
        print(f"\n--- Trace Summary ---")
        print(f"Status: {summary.get('status')}  |  Events emitted: {len(events)}")


if __name__ == "__main__":
    main()
