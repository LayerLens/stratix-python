"""Real-Mirascope test rig — the network is the only thing faked.

Mirascope v2 resolves a provider out of its own registry and builds a real
``openai.OpenAI`` client inside it, so the honest seam for an offline test is
that client's HTTP transport: :func:`register_openai` swaps in an
``httpx.MockTransport`` and leaves the real ``@llm.call`` decorator, the real
``Prompt``, the real provider, the real OpenAI SDK deserialisation and the real
``llm.Response`` untouched. Every ``model_id`` / ``provider_id`` / ``usage`` an
assertion reads is therefore produced by Mirascope, not by a stub.

``register_provider`` mutates a process-global registry, so
:func:`mirascope_openai` restores it (and the ``provider_singleton`` cache)
afterwards — otherwise a leaked provider would silently serve later tests.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Callable, Iterator, Optional
from contextlib import contextmanager

import httpx
import mirascope.llm as llm  # pyright: ignore[reportMissingImports]
from mirascope.llm.providers.openai import OpenAIProvider  # pyright: ignore[reportMissingImports]
from mirascope.llm.providers.provider_registry import (  # pyright: ignore[reportMissingImports]
    PROVIDER_REGISTRY,
    provider_singleton,
    reset_provider_registry,
)

from openai import OpenAI, AsyncOpenAI

#: ``:completions`` pins the chat-completions transport so one recorded body
#: shape serves every lane; without it Mirascope picks the responses API for
#: known OpenAI models and the fixture would have to model both wire formats.
MODEL_ID = "openai/gpt-4o-mini:completions"

#: What ``model_id`` above must normalise to before LayerLens can price it.
BARE_MODEL = "gpt-4o-mini"


def completion_body(
    content: str = "Dune by Frank Herbert",
    *,
    prompt_tokens: int = 17,
    completion_tokens: int = 6,
) -> Dict[str, Any]:
    """A real OpenAI chat-completion wire body (the shape the SDK deserialises)."""
    return {
        "id": "chatcmpl-layerlens-test",
        "object": "chat.completion",
        "created": 1730000000,
        "model": "gpt-4o-mini-2024-07-18",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def ok_handler(body: Optional[Dict[str, Any]] = None) -> Callable[[httpx.Request], httpx.Response]:
    """A transport handler returning *body* (default: a plain completion)."""
    payload = completion_body() if body is None else body

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    return handler


def not_found_handler() -> Callable[[httpx.Request], httpx.Response]:
    """A transport handler returning OpenAI's real 404 model_not_found body.

    The real ``OpenAIProvider.error_map`` turns this into a genuine
    ``mirascope.llm.exceptions.NotFoundError`` — the SDK exception shape a
    failing Mirascope call actually raises.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={
                "error": {
                    "message": "The model `gpt-4o-mini-ghost` does not exist",
                    "type": "invalid_request_error",
                    "code": "model_not_found",
                }
            },
        )

    return handler


def recording_handler(
    seen: List[httpx.Request],
    body: Optional[Dict[str, Any]] = None,
) -> Callable[[httpx.Request], httpx.Response]:
    """Like :func:`ok_handler` but appends every request to *seen*."""
    payload = completion_body() if body is None else body

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=payload)

    return handler


@contextmanager
def mirascope_openai(handler: Callable[[httpx.Request], httpx.Response]) -> Iterator[OpenAIProvider]:
    """Register a REAL ``OpenAIProvider`` whose only fake is its HTTP transport."""
    saved = dict(PROVIDER_REGISTRY)
    provider = OpenAIProvider(api_key="test-key")
    sync_client = OpenAI(api_key="test-key", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    async_client = AsyncOpenAI(
        api_key="test-key", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    # OpenAIProvider fans out to a completions and a responses sub-provider; both
    # hold their own clients, so both transports must be swapped.
    for sub in (provider._completions_provider, provider._responses_provider):
        sub.client = sync_client
        sub.async_client = async_client
    provider.client = sync_client
    llm.register_provider(provider, scope="openai/")
    try:
        yield provider
    finally:
        reset_provider_registry()
        PROVIDER_REGISTRY.update(saved)
        provider_singleton.cache_clear()


def json_body(value: Any, **kwargs: Any) -> Dict[str, Any]:
    """A completion body whose assistant message is ``json.dumps(value)``."""
    return completion_body(json.dumps(value), **kwargs)


@contextmanager
def capture_raw_emissions() -> Iterator[List[tuple]]:
    """Record ``(event_type, payload)`` exactly as the ADAPTER hands them over.

    ``TraceCollector.emit`` runs ``redact_payload`` — the collector-tier
    ``_CONTENT_KEYS`` backstop — before anything is stored, so an assertion made
    against a stored trace passes whether or not the adapter gated at emit time:
    it proves the backstop, not the adapter. Intercepting at the collector's
    front door is the only way to hold the adapter's OWN ``_set_if_capturing``
    discipline to account (defense in depth means both layers must be tested,
    and only one of them is testable downstream).
    """
    from copy import deepcopy

    from layerlens.instrument._collector import TraceCollector

    seen: List[tuple] = []
    original = TraceCollector.emit

    def spy(self, event_type, payload, *args, **kwargs):
        seen.append((event_type, deepcopy(payload)))
        return original(self, event_type, payload, *args, **kwargs)

    TraceCollector.emit = spy
    try:
        yield seen
    finally:
        TraceCollector.emit = original


def call_classes() -> tuple:
    """The four real v2 ``Call`` classes the adapter patches."""
    return (llm.Call, llm.AsyncCall, llm.ContextCall, llm.AsyncContextCall)


def restore_call_classes() -> None:
    """Force-unpatch every ``Call`` class, whatever left the patch behind.

    A test whose traced call raises before ``disconnect()`` would otherwise leak
    a class-level patch bound to a dead mock client, and every later lane would
    silently emit into that adapter's collector instead of its own — green tests
    asserting nothing. ``functools.wraps`` leaves ``__wrapped__`` pointing at the
    pristine implementation, which is what we restore.
    """
    for cls in call_classes():
        current = cls.__dict__.get("call")
        if getattr(current, "_layerlens_traced", False):
            cls.call = current.__wrapped__
