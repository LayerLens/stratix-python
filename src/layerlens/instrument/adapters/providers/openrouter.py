from __future__ import annotations

import math
import logging
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from .openai import _CAPTURE_PARAMS, OpenAIProvider  # type: ignore[attr-defined]

log: logging.Logger = logging.getLogger(__name__)

#: OpenRouter's OpenAI-compatible API root.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterProvider(OpenAIProvider):
    """OpenRouter adapter (OpenAI-compatible gateway).

    OpenRouter exposes an OpenAI-compatible Chat Completions API, so it is driven
    through the same ``openai`` SDK with a custom ``base_url``. Every
    request/response/stream extraction is inherited from :class:`OpenAIProvider`;
    this adapter re-tags the call as OpenRouter and records the gateway's own
    billed charge.

    The model captured on each event is the ROUTED slug the gateway reports (e.g.
    ``anthropic/claude-opus-4.8``), which records the model that actually served
    the request — strictly more honest than the requested slug, which for
    ``openrouter/auto`` names no model at all.

    Cost: OpenRouter bills at its own rates, which no table we ship holds, so the
    gateway is the sole authority (``provider_cost_only``). Enable usage
    accounting per request with ``extra_body={"usage": {"include": True}}`` and
    the reported ``usage.cost`` becomes ``cost.record.cost_usd`` with
    ``cost_source="provider"``. Without it, no ``cost.record`` is emitted —
    tokens still ride ``model.invoke``, but a price would be invented.
    """

    name = "openrouter"
    capture_params = _CAPTURE_PARAMS
    event_prefix = "openrouter"
    provider_cost_only = True

    @staticmethod
    def classify_provider(event_name: str, kwargs: Dict[str, Any]) -> Optional[str]:  # noqa: ARG004
        # Mandatory, not cosmetic: the shared emitter otherwise derives the
        # provider from the event name, so an Anthropic-served call routed through
        # OpenRouter would claim provider="openai" on every cost.record/tool.call.
        return "openrouter"

    @staticmethod
    def extract_provider_cost(response: Any) -> Optional[float]:
        """The charge OpenRouter reported for this call, or None.

        Present only when the caller enabled usage accounting. ``bool`` is
        rejected explicitly because ``float(True)`` is ``1.0`` — a $1 charge
        conjured out of a flag. Any value that will not coerce is left as None
        and no cost is recorded; a malformed gateway field must never become a
        price.

        Coercing is necessary but NOT sufficient. ``float("nan")``/``float("inf")``
        succeed, so the type checks above never see them: without the finiteness
        guard an undefined or infinite charge would be emitted stamped
        ``cost_source="provider"`` — asserted as a BILLED FACT — and would serialize
        to bare ``NaN``/``Infinity``, invalid JSON per RFC 8259, poisoning every
        downstream SUM it reaches. A negative value is rejected for the same reason:
        a gateway does not bill a negative charge for a completion, so it is a
        malformed field and not a credit we could honestly model. A reported ZERO is
        kept — ``:free`` slugs genuinely bill $0, and that zero is a fact.
        """
        usage = getattr(response, "usage", None)
        cost = getattr(usage, "cost", None) if usage is not None else None
        if cost is None or isinstance(cost, bool):
            return None
        try:
            value = float(cost)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value) or value < 0:
            log.warning(
                "openrouter reported a malformed usage.cost (%r); no cost.record emitted for this call",
                cost,
            )
            return None
        return value

    def connect(self, target: Any = None, **kwargs: Any) -> Any:
        # Nothing about an openai client proves it points at OpenRouter, and this
        # adapter re-tags unconditionally — so instrumenting a plain OpenAI client
        # would silently relabel real OpenAI calls. Warn rather than raise: a
        # caller may legitimately front OpenRouter with a proxy or a gateway.
        base_url = _client_base_url(target)
        if base_url is not None and urlparse(base_url).netloc != urlparse(OPENROUTER_BASE_URL).netloc:
            log.warning(
                "instrumenting a client whose base_url is %s as openrouter: every call will be "
                "tagged provider='openrouter'. Use instrument_openai() for a plain OpenAI client.",
                base_url,
            )
        return super().connect(target, **kwargs)


def _client_base_url(client: Any) -> Optional[str]:
    """The client's base URL without query string — never log the api-key."""
    url = getattr(client, "base_url", None)
    if url is None:
        return None
    try:
        parsed = urlparse(str(url))
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")
    except ValueError:
        return None


def build_client(
    api_key: str,
    *,
    http_referer: Optional[str] = None,
    x_title: Optional[str] = None,
    base_url: str = OPENROUTER_BASE_URL,
    **client_kwargs: Any,
) -> Any:
    """Construct an ``openai`` SDK client pointed at OpenRouter.

    Sets OpenRouter's optional attribution headers — ``HTTP-Referer`` and
    ``X-Title`` — which surface the calling app on the OpenRouter dashboard and
    model rankings. Pass the returned client to :func:`instrument_openrouter`.

    Args:
        api_key: OpenRouter API key (``sk-or-...``).
        http_referer: App URL for OpenRouter attribution (``HTTP-Referer``).
        x_title: Human-readable app name (``X-Title``).
        base_url: Override the OpenRouter API root.
        client_kwargs: Forwarded to ``openai.OpenAI`` (e.g. ``timeout``,
            ``max_retries``, ``default_headers``).
    """
    try:
        from openai import OpenAI
    except ImportError as err:
        raise ImportError(
            "The 'openai' package is required for OpenRouter instrumentation "
            "(OpenRouter is an OpenAI-compatible API). Install it with: pip install openai"
        ) from err

    # Merge rather than pass a second `default_headers=`: a caller supplying their
    # own headers would otherwise hit "got multiple values for keyword argument".
    headers: Dict[str, str] = dict(client_kwargs.pop("default_headers", None) or {})
    if http_referer:
        headers["HTTP-Referer"] = http_referer
    if x_title:
        headers["X-Title"] = x_title
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
        default_headers=headers or None,
        **client_kwargs,
    )


# --- Convenience API ---


def instrument_openrouter(client: Any) -> OpenRouterProvider:
    from .._registry import get, register

    existing = get("openrouter")
    if existing is not None:
        existing.disconnect()
    provider = OpenRouterProvider()
    provider.connect(client)
    register("openrouter", provider)
    return provider


def uninstrument_openrouter() -> None:
    from .._registry import unregister

    unregister("openrouter")
