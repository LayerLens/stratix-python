"""Canonical agentic scenarios per provider.

Each ``run_*`` function instruments the real provider SDK, drives one of three
*flows*, and uninstruments. It must be called inside the harness's active
``TraceCollector`` context so the adapter's emitted events are captured.

Flows:
- ``"default"``  — the canonical workflow: a tool-use loop (tool-capable
  providers) or a multi-turn chat (chat providers). 5-15 events.
- ``"streaming"`` — a single streamed call (tool-capable providers only) so the
  ``_streaming`` path runs and ``ttft_ms`` is emitted.
- ``"error"``    — a call against an invalid model id so the adapter emits
  ``agent.error``. The provider exception is swallowed (it is expected).
- ``"async"`` / ``"async-streaming"`` — openai + anthropic only: the canonical
  tool loop / streamed call driven through the **async** client
  (``AsyncOpenAI`` / ``AsyncAnthropic``) via ``asyncio.run``. The harness sets
  the collector ContextVar before calling the runner and ``asyncio.run``
  copies the current context, so events emitted inside the coroutine land in
  the same collector. These flows verify the adapters' async-client routing
  (T1 / finding N5: ``create`` is a coroutine method on async clients — the
  sync wrapper must not swallow it).

Every prompt embeds ``SENTINEL`` so the redaction check (run under
``capture_content=False``) can assert it never reaches the stored trace.

SDK imports are lazy (inside each function) so collection never requires a
provider package to be installed; the test skips via ``importorskip`` instead.
"""

from __future__ import annotations

import os
import json
import asyncio
from typing import Any

from layerlens.instrument.adapters.providers.ollama import instrument_ollama, uninstrument_ollama
from layerlens.instrument.adapters.providers.openai import instrument_openai, uninstrument_openai
from layerlens.instrument.adapters.providers.bedrock import instrument_bedrock, uninstrument_bedrock
from layerlens.instrument.adapters.providers.litellm import instrument_litellm, uninstrument_litellm
from layerlens.instrument.adapters.providers.anthropic import instrument_anthropic, uninstrument_anthropic
from layerlens.instrument.adapters.providers.azure_openai import (
    instrument_azure_openai,
    uninstrument_azure_openai,
)
from layerlens.instrument.adapters.providers.google_vertex import (
    instrument_google_vertex,
    uninstrument_google_vertex,
)

#: Embedded in every prompt; the redaction flow asserts it never reaches the
#: stored trace once content capture is disabled.
SENTINEL = "LL-SENTINEL-7f3a9c2e"

_BAD_MODEL = "layerlens-live-nonexistent-model-xyz"

_WEATHER_PROMPT = f"What is the weather in Paris? Use the get_weather tool. {SENTINEL}"

#: Flows driven through the async client (AsyncOpenAI / AsyncAnthropic).
_ASYNC_FLOWS = ("async", "async-streaming")

# Shared get_weather tool schemas (sync + async flows drive the same loop).
_WEATHER_TOOL_ANTHROPIC = {
    "name": "get_weather",
    "description": "Get the current weather for a city.",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}
_WEATHER_TOOL_OPENAI = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


# --------------------------------------------------------------------------- #
# Anthropic
# --------------------------------------------------------------------------- #
def run_anthropic(flow: str) -> None:
    import anthropic

    model = os.environ.get("LL_ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

    if flow in _ASYNC_FLOWS:
        client = anthropic.AsyncAnthropic()
        instrument_anthropic(client)
        try:
            asyncio.run(_run_anthropic_async(flow, client, model))
        finally:
            uninstrument_anthropic()
        return

    client = anthropic.Anthropic()
    instrument_anthropic(client)
    try:
        if flow == "error":
            try:
                client.messages.create(
                    model=_BAD_MODEL,
                    max_tokens=16,
                    messages=[{"role": "user", "content": f"hi {SENTINEL}"}],
                )
            except Exception:
                pass
            return

        if flow == "streaming":
            # Iterate the wrapped stream so the adapter's proxy taps each event.
            # Convenience consumers (until_done / get_final_message / text_stream)
            # delegate to the inner stream via __getattr__ and are NOT captured.
            with client.messages.stream(
                model=model,
                max_tokens=64,
                messages=[{"role": "user", "content": f"Name two oceans. {SENTINEL}"}],
            ) as stream:
                for _ in stream:
                    pass
            return

        messages: list[dict[str, Any]] = [{"role": "user", "content": _WEATHER_PROMPT}]
        first = client.messages.create(model=model, max_tokens=256, messages=messages, tools=[_WEATHER_TOOL_ANTHROPIC])
        messages.append({"role": "assistant", "content": first.content})
        messages.append(_anthropic_tool_followup(first))
        client.messages.create(model=model, max_tokens=128, messages=messages)
    finally:
        uninstrument_anthropic()


async def _run_anthropic_async(flow: str, client: Any, model: str) -> None:
    """Async twin of the anthropic flows, against an instrumented ``AsyncAnthropic``."""
    try:
        if flow == "async-streaming":
            # AsyncAnthropic's messages.stream is an async context manager; the
            # adapter's _TracedMessageStream proxy supports __aenter__/__aiter__,
            # and iterating it taps each event (ttft_ms lands on model.invoke).
            async with client.messages.stream(
                model=model,
                max_tokens=64,
                messages=[{"role": "user", "content": f"Name two oceans. {SENTINEL}"}],
            ) as stream:
                async for _ in stream:
                    pass
            return

        # flow == "async": the canonical tool-use loop, awaited end to end.
        messages: list[dict[str, Any]] = [{"role": "user", "content": _WEATHER_PROMPT}]
        first = await client.messages.create(
            model=model, max_tokens=256, messages=messages, tools=[_WEATHER_TOOL_ANTHROPIC]
        )
        messages.append({"role": "assistant", "content": first.content})
        messages.append(_anthropic_tool_followup(first))
        await client.messages.create(model=model, max_tokens=128, messages=messages)
    finally:
        await client.close()


def _anthropic_tool_followup(first: Any) -> dict[str, Any]:
    """The user turn answering ``first``'s tool_use (or a plain follow-up)."""
    tool_use = next((b for b in first.content if getattr(b, "type", None) == "tool_use"), None)
    if tool_use is not None:
        return {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": tool_use.id, "content": "Sunny, 21C."}],
        }
    return {"role": "user", "content": "Thanks — summarize in one sentence."}


# --------------------------------------------------------------------------- #
# OpenAI / Azure OpenAI (shared chat-completions tool loop)
# --------------------------------------------------------------------------- #
def _run_openai_like(flow: str, client: Any, model: str) -> None:
    if flow == "error":
        try:
            client.chat.completions.create(
                model=_BAD_MODEL,
                messages=[{"role": "user", "content": f"hi {SENTINEL}"}],
                max_tokens=16,
            )
        except Exception:
            pass
        return

    if flow == "streaming":
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": f"Name two oceans. {SENTINEL}"}],
            max_tokens=32,
            stream=True,
        )
        for _ in stream:
            pass
        return

    messages: list[dict[str, Any]] = [{"role": "user", "content": _WEATHER_PROMPT}]
    first = client.chat.completions.create(model=model, messages=messages, max_tokens=256, tools=[_WEATHER_TOOL_OPENAI])
    messages.extend(_openai_tool_followup(first))
    client.chat.completions.create(model=model, messages=messages, max_tokens=128)


async def _run_openai_like_async(flow: str, client: Any, model: str) -> None:
    """Async twin of ``_run_openai_like``, against an instrumented ``AsyncOpenAI``."""
    try:
        if flow == "async-streaming":
            # The wrapped async create returns the adapter's async stream
            # iterator; consuming it emits model.invoke with ttft_ms.
            stream = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": f"Name two oceans. {SENTINEL}"}],
                max_tokens=32,
                stream=True,
            )
            async for _ in stream:
                pass
            return

        # flow == "async": the canonical tool-use loop, awaited end to end.
        messages: list[dict[str, Any]] = [{"role": "user", "content": _WEATHER_PROMPT}]
        first = await client.chat.completions.create(
            model=model, messages=messages, max_tokens=256, tools=[_WEATHER_TOOL_OPENAI]
        )
        messages.extend(_openai_tool_followup(first))
        await client.chat.completions.create(model=model, messages=messages, max_tokens=128)
    finally:
        await client.close()


def _openai_tool_followup(first: Any) -> list[dict[str, Any]]:
    """The assistant turn from ``first`` plus the tool results (or a plain follow-up)."""
    choice = first.choices[0].message
    out: list[dict[str, Any]] = [choice.model_dump(exclude_none=True)]
    if choice.tool_calls:
        for tc in choice.tool_calls:
            out.append({"role": "tool", "tool_call_id": tc.id, "content": "Sunny, 21C."})
    else:
        out.append({"role": "user", "content": "Thanks — summarize in one sentence."})
    return out


def run_openai(flow: str) -> None:
    import openai

    model = os.environ.get("LL_OPENAI_MODEL", "gpt-4o-mini")

    if flow in _ASYNC_FLOWS:
        client = openai.AsyncOpenAI()
        instrument_openai(client)
        try:
            asyncio.run(_run_openai_like_async(flow, client, model))
        finally:
            uninstrument_openai()
        return

    client = openai.OpenAI()
    instrument_openai(client)
    try:
        _run_openai_like(flow, client, model)
    finally:
        uninstrument_openai()


def run_azure_openai(flow: str) -> None:
    import openai

    client = openai.AzureOpenAI(
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    )
    instrument_azure_openai(client)
    try:
        _run_openai_like(flow, client, os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"))
    finally:
        uninstrument_azure_openai()


# --------------------------------------------------------------------------- #
# Google Vertex (Gemini) — single-turn function call + a plain follow-up
# --------------------------------------------------------------------------- #
def run_google_vertex(flow: str) -> None:
    from vertexai.generative_models import Tool, GenerativeModel, FunctionDeclaration

    model_name = os.environ.get("LL_VERTEX_MODEL", "gemini-1.5-flash")

    if flow == "error":
        bad = GenerativeModel(_BAD_MODEL)
        instrument_google_vertex(bad)
        try:
            try:
                bad.generate_content(f"hi {SENTINEL}")
            except Exception:
                pass
        finally:
            uninstrument_google_vertex()
        return

    weather = FunctionDeclaration(
        name="get_weather",
        description="Get the current weather for a city.",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )
    model = GenerativeModel(model_name, tools=[Tool(function_declarations=[weather])])
    instrument_google_vertex(model)
    try:
        if flow == "streaming":
            for _ in model.generate_content(f"Name two oceans. {SENTINEL}", stream=True):
                pass
            return
        # Vertex returns the function call in a single turn -> emits tool.call.
        model.generate_content(_WEATHER_PROMPT)
        # A plain follow-up gives a second model.invoke so we clear the event floor.
        model.generate_content(f"Name a prime number. {SENTINEL}")
    finally:
        uninstrument_google_vertex()


# --------------------------------------------------------------------------- #
# Chat-only providers: ollama, bedrock, litellm (3-turn chats; no tool.call)
# --------------------------------------------------------------------------- #
_CHAT_TURNS = (
    f"Name a mountain. {SENTINEL}",
    f"Name a river. {SENTINEL}",
    f"Name a lake. {SENTINEL}",
)


def run_ollama(flow: str) -> None:
    import ollama

    client = ollama.Client()
    instrument_ollama(client, cost_per_second=0.0001)
    model = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
    try:
        if flow == "error":
            try:
                client.chat(model=_BAD_MODEL, messages=[{"role": "user", "content": f"hi {SENTINEL}"}])
            except Exception:
                pass
            return
        for turn in _CHAT_TURNS:
            client.chat(model=model, messages=[{"role": "user", "content": turn}])
    finally:
        uninstrument_ollama()


def run_bedrock(flow: str) -> None:
    import boto3

    client = boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    instrument_bedrock(client)
    # Default to an ACTIVE model that also exercises the Amazon Nova invoke_model
    # parser (LAY-3605). The old default (anthropic.claude-3-haiku) is now a LEGACY
    # model — on-demand DENIED — so it failed the provider live case by default.
    model_id = os.environ.get("LL_BEDROCK_MODEL", "amazon.nova-micro-v1:0")

    def _body(model: str, text: str) -> str:
        # Strip an inference-profile region prefix (us./eu./apac./us-gov.) to get the family.
        parts = model.split(".")
        family = parts[1] if parts[0] in ("us", "eu", "apac", "us-gov") and len(parts) > 1 else parts[0]
        if family == "amazon":  # Amazon Nova: Converse-shaped invoke_model body
            return json.dumps(
                {"messages": [{"role": "user", "content": [{"text": text}]}], "inferenceConfig": {"maxTokens": 32}}
            )
        # Anthropic Claude messages body
        return json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 32,
                "messages": [{"role": "user", "content": text}],
            }
        )

    def _invoke(text: str, model: str) -> None:
        client.invoke_model(modelId=model, body=_body(model, text))

    try:
        if flow == "error":
            try:
                _invoke(f"hi {SENTINEL}", _BAD_MODEL)
            except Exception:
                pass
            return
        for turn in _CHAT_TURNS:
            _invoke(turn, model_id)
    finally:
        uninstrument_bedrock()


def run_litellm(flow: str) -> None:
    import litellm

    instrument_litellm()
    model = os.environ.get("LITELLM_MODEL", "gpt-4o-mini")
    try:
        if flow == "error":
            try:
                litellm.completion(
                    model=_BAD_MODEL,
                    messages=[{"role": "user", "content": f"hi {SENTINEL}"}],
                    max_tokens=16,
                )
            except Exception:
                pass
            return
        for turn in _CHAT_TURNS:
            litellm.completion(model=model, messages=[{"role": "user", "content": turn}], max_tokens=20)
    finally:
        uninstrument_litellm()
