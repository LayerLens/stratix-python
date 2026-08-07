"""LLM model pricing tables and cost calculation.

Per-1K-token rates (USD). Providers that ship their own pricing table (Azure,
Bedrock) pass their override table into :func:`calculate_cost`.

Pricing is updateable without code changes (LAY-3327 / LAY-3330 ACs):
set the ``LAYERLENS_PRICING_TABLE`` env var to the path of a JSON file
shaped ``{"model-name": {"input": N, "output": N}, ...}`` to override or
extend the bundled table. Env-level overrides take precedence over any
caller-supplied ``pricing_table`` and over the bundled ``PRICING``.
"""

from __future__ import annotations

import os
import re
import json
import logging
from typing import Optional
from dataclasses import dataclass

from .token_usage import NormalizedTokenUsage

log: logging.Logger = logging.getLogger(__name__)

PRICING_OVERRIDE_ENV = "LAYERLENS_PRICING_TABLE"

# Matches an OpenAI-style dated suffix ``-YYYY-MM-DD`` or an Anthropic-style
# ``-YYYYMMDD``. Used to fall back to the base model's pricing when the
# specific dated variant isn't in the table (LAY-3330 fuzzy matching AC).
_DATE_SUFFIX_RE = re.compile(r"-(?:\d{4}-\d{2}-\d{2}|\d{8})$")

PRICING: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o": {"input": 0.0025, "output": 0.0100},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4o-2024-11-20": {"input": 0.0025, "output": 0.0100},
    "gpt-4.1": {"input": 0.002, "output": 0.008},
    "gpt-4.1-mini": {"input": 0.0004, "output": 0.0016},
    "gpt-4.1-nano": {"input": 0.0001, "output": 0.0004},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "o1": {"input": 0.015, "output": 0.060},
    "o1-mini": {"input": 0.003, "output": 0.012},
    "o3": {"input": 0.010, "output": 0.040},
    "o3-mini": {"input": 0.0011, "output": 0.0044},
    "o4-mini": {"input": 0.0011, "output": 0.0044},
    # Anthropic — both dated variants and base names; fuzzy matching below
    # also falls back from ``claude-foo-YYYYMMDD`` to ``claude-foo``.
    "claude-sonnet-4-5-20250929": {"input": 0.003, "output": 0.015},
    "claude-sonnet-4-5": {"input": 0.003, "output": 0.015},
    "claude-opus-4-20250115": {"input": 0.015, "output": 0.075},
    "claude-opus-4-6": {"input": 0.015, "output": 0.075},
    "claude-opus-4-7": {"input": 0.015, "output": 0.075},
    "claude-haiku-4-5-20251001": {"input": 0.0008, "output": 0.004},
    "claude-haiku-4-5": {"input": 0.0008, "output": 0.004},
    "claude-haiku-3-5-20241022": {"input": 0.0008, "output": 0.004},
    "claude-haiku-3-5": {"input": 0.0008, "output": 0.004},
    "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
    "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
    "claude-3-opus": {"input": 0.015, "output": 0.075},
    "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    # Google
    "gemini-2.5-pro": {"input": 0.00125, "output": 0.01},
    "gemini-2.5-flash": {"input": 0.000075, "output": 0.0003},
    "gemini-2.0-flash": {"input": 0.0001, "output": 0.0004},
    "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
    "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
    # Meta
    "llama-3.3-70b": {"input": 0.00099, "output": 0.00099},
    "llama-3.1-70b": {"input": 0.00099, "output": 0.00099},
    "llama-3.1-8b": {"input": 0.00022, "output": 0.00022},
    # Mistral
    "mistral-large": {"input": 0.002, "output": 0.006},
    "mistral-small": {"input": 0.0002, "output": 0.0006},
    # OpenAI embeddings — input-only (no completion tokens); per-1K USD. The
    # embedding adapter emits a priced cost.record for these; without a rate here
    # a real embedding call shipped tokens-only (no cost_usd / no rollup).
    "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
    "text-embedding-3-large": {"input": 0.00013, "output": 0.0},
    "text-embedding-ada-002": {"input": 0.0001, "output": 0.0},
}

AZURE_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {"input": 0.00275, "output": 0.011},
    "gpt-4o-mini": {"input": 0.000165, "output": 0.00066},
    "gpt-4-turbo": {"input": 0.011, "output": 0.033},
    "gpt-4": {"input": 0.033, "output": 0.066},
    "gpt-35-turbo": {"input": 0.00055, "output": 0.00165},
}

BEDROCK_PRICING: dict[str, dict[str, float]] = {
    "anthropic.claude-3-5-sonnet-20241022-v2:0": {"input": 0.003, "output": 0.015},
    "anthropic.claude-3-opus-20240229-v1:0": {"input": 0.015, "output": 0.075},
    "anthropic.claude-3-haiku-20240307-v1:0": {"input": 0.00025, "output": 0.00125},
    "meta.llama3-1-70b-instruct-v1:0": {"input": 0.00099, "output": 0.00099},
    "meta.llama3-1-8b-instruct-v1:0": {"input": 0.00022, "output": 0.00022},
    "cohere.command-r-plus-v1:0": {"input": 0.003, "output": 0.015},
    "cohere.command-r-v1:0": {"input": 0.0005, "output": 0.0015},
    # Amazon Nova (the cheap default many Bedrock Agents run on). Region-prefixed
    # inference-profile ids (us./eu./apac.) resolve to these via _resolve_rates.
    "amazon.nova-micro-v1:0": {"input": 0.000035, "output": 0.00014},
    "amazon.nova-lite-v1:0": {"input": 0.00006, "output": 0.00024},
    "amazon.nova-pro-v1:0": {"input": 0.0008, "output": 0.0032},
    "amazon.nova-premier-v1:0": {"input": 0.0025, "output": 0.0125},
    # Amazon Titan Text on Bedrock. The family is parsed by the provider adapter
    # (results[0].tokenCount for completion) but had no rates, so cost resolved
    # to None (ADP-W1 BUG-2). us-east-1 on-demand per-1K USD.
    "amazon.titan-text-lite-v1": {"input": 0.00015, "output": 0.0002},
    "amazon.titan-text-express-v1": {"input": 0.0002, "output": 0.0006},
    "amazon.titan-text-premier-v1:0": {"input": 0.0005, "output": 0.0015},
    # Mistral on Bedrock (LAY-3452). The family is parsed by the provider
    # adapter but had no rates, so cost resolved to None.
    "mistral.mistral-7b-instruct-v0:2": {"input": 0.00015, "output": 0.0002},
    "mistral.mixtral-8x7b-instruct-v0:1": {"input": 0.00045, "output": 0.0007},
    "mistral.mistral-large-2402-v1:0": {"input": 0.004, "output": 0.012},
    "mistral.mistral-large-2407-v1:0": {"input": 0.002, "output": 0.006},
    "mistral.mistral-small-2402-v1:0": {"input": 0.001, "output": 0.003},
    # AI21 Jamba on Bedrock (LAY-3452).
    "ai21.jamba-1-5-large-v1:0": {"input": 0.002, "output": 0.008},
    "ai21.jamba-1-5-mini-v1:0": {"input": 0.0002, "output": 0.0004},
}

# Cross-region inference-profile prefixes Bedrock prepends to a model id
# (e.g. ``us.amazon.nova-lite-v1:0``). Nova on Agents requires an inference
# profile, so the live ``foundationModel`` is always prefixed.
_INFERENCE_PROFILE_PREFIX_RE = re.compile(r"^(?:us-gov|us|eu|apac)\.(.+)$")


def _normalize_model_id(model: str) -> str:
    """Strip a Bedrock inference-profile wrapper down to the bare model id.

    Handles a full inference-profile/foundation-model ARN (take the trailing
    segment) and a cross-region prefix (``us.``/``eu.``/``apac.``/``us-gov.``).
    Returns ``model`` unchanged when neither applies.
    """
    if model.startswith("arn:"):
        model = model.rsplit("/", 1)[-1]
    match = _INFERENCE_PROFILE_PREFIX_RE.match(model)
    if match:
        model = match.group(1)
    return model


def _cached_token_discount(model: str) -> float:
    """Cached-token (cache-READ) rate as a fraction of the input price.

    - Anthropic: 90% off (10% of input)
    - Google: 75% off (25% of input)
    - Others (OpenAI et al.): 50% off
    """
    lower = model.lower()
    if lower.startswith("claude") or "anthropic." in lower:
        return 0.1
    if lower.startswith("gemini"):
        return 0.25
    return 0.5


# Per-request service-tier multipliers (A2 / LAY-3626). OpenAI flex/batch are
# priced at Batch rates (~50% off); priority buys ~2-2.5x for SLA latency. The
# response echoes ``service_tier``; pricing was model-keyed only, so flex was
# over-billed ~2x and priority UNDER-billed (the dangerous direction — budgets
# and the per-run dollar ceiling read cost_usd). Unknown tiers => 1.0 (standard).
# Web-grounded 2026-06-25: developers.openai.com/api/docs/pricing (priority
# ~2-2.5x) + flex-processing guide (flex == Batch rates ~0.5x).
TIER_MULTIPLIERS: dict[str, float] = {
    "flex": 0.5,
    "batch": 0.5,
    "scale": 0.5,
    "priority": 2.0,
    "standard": 1.0,
    "default": 1.0,
    "auto": 1.0,
    "on_demand": 1.0,
}

# Anthropic prompt-cache WRITE multiplier on base input (A3 / LAY-3626):
# 5-minute TTL writes cost 1.25x base input (1-hour = 2x). The usage wire does
# not carry the TTL, so we price the common 5-minute write. Cache READ uses
# _cached_token_discount (0.1x). cache_creation tokens were priced at $0 before
# this — a systematic under-bill on the canonical long-running-agent pattern.
CACHE_WRITE_MULTIPLIER = 1.25


def _tier_multiplier(service_tier: Optional[str]) -> float:
    if not service_tier:
        return 1.0
    return TIER_MULTIPLIERS.get(str(service_tier).lower(), 1.0)


def _cost_from_rates(
    rates: dict[str, float],
    model: str,
    usage: NormalizedTokenUsage,
    service_tier: Optional[str] = None,
) -> Optional[float]:
    """The single cost formula: tier-scaled input/output + cache READ (discount)
    + cache WRITE (1.25x) + completion. Shared by :func:`calculate_cost` and
    :class:`PricingTable` so the math can never drift between the two paths.

    Returns ``None`` for a token shape this formula CANNOT price (LAY-3622 / A4b).
    The formula reads prompt / cached / cache_creation / completion and never
    ``total_tokens``, so a usage carrying only a total has no priceable dimension
    at all — and summing four zeroes yielded ``0.0``, a *computed-looking* zero for
    a call the provider really billed. ``0.0 is not None``, so every downstream
    "did we get a price?" guard passed and the zero shipped as a derived cost
    (.claude/CLAUDE.md rule 3). ``None`` is the honest answer: we know the tokens,
    we cannot price them; the caller then omits the cost instead of inventing one.

    A usage whose dimensions are all genuinely zero is NOT this case — it billed
    nothing and 0.0 is arithmetic. The discriminator is a positive total (or any
    positive count) with nothing priceable behind it.

    ``reasoning_tokens`` / ``thinking_tokens`` are DELIBERATELY not summed here
    (LAY-3622 F4). They are a BREAKDOWN, not an extra dimension, for the two
    providers that populate them:

    * OpenAI reads ``reasoning_tokens`` out of ``completion_tokens_details``
      (``openai.py``) — a decomposition of ``completion_tokens``, which the output
      leg above already prices. Adding it would DOUBLE-bill.
    * Anthropic's ``thinking_tokens`` is a ``len(text) // 4`` ESTIMATE
      (``_count_thinking_tokens``: "Anthropic does not surface a dedicated
      thinking_tokens field today"), and ``completion_tokens`` is its
      ``output_tokens``, which already carries the thinking. Charging money against
      a character-count estimate is .claude/CLAUDE.md rule 3.

    The genuine gap — Gemini reports ``thoughtsTokenCount`` OUTSIDE
    ``candidatesTokenCount`` while ``total_token_count`` includes it — is therefore
    not fixed by reading these fields. It is DETECTED instead, by comparing
    :func:`priced_token_count` against the reported total at the collector
    chokepoint, which marks the record rather than inventing a rate for it. That
    detection is correct under BOTH readings of a provider's semantics: if reasoning
    really is inside ``completion_tokens`` the total matches and nothing is marked;
    if it is outside, the total exceeds what we priced and the gap is recorded.
    """
    tier = _tier_multiplier(service_tier)
    input_rate = rates.get("input", 0.0) * tier
    output_rate = rates.get("output", 0.0) * tier

    cached = usage.cached_tokens or 0
    cache_creation = usage.cache_creation_tokens or 0
    non_cached = max(usage.prompt_tokens - cached, 0)
    cached_rate = input_rate * _cached_token_discount(model)

    # UNPRICEABLE SHAPE GATE: no dimension this formula can read carries a count.
    # ``prompt_tokens`` is checked (not ``non_cached``) so a fully-cached turn —
    # prompt == cached, non_cached == 0 — still prices via the cached leg.
    if not (usage.prompt_tokens or usage.completion_tokens or cached or cache_creation):
        if usage.total_tokens:
            # Tokens were billed; we just cannot attribute them to a rate.
            return None
        # Nothing was billed at all: an honest zero.
        return 0.0

    cost = (
        (non_cached * input_rate / 1000)
        + (cached * cached_rate / 1000)
        + (cache_creation * input_rate * CACHE_WRITE_MULTIPLIER / 1000)
        + (usage.completion_tokens * output_rate / 1000)
    )
    return round(cost, 8)


_env_overrides_cache: Optional[dict[str, dict[str, float]]] = None


def _load_env_overrides() -> dict[str, dict[str, float]]:
    """Load (and memoise) env-var-driven pricing overrides.

    Reads ``LAYERLENS_PRICING_TABLE``. Bad JSON or unreadable files log a
    warning and resolve to an empty override map (don't crash the request
    path over an ops-config error). Tests call :func:`reset_pricing_cache`
    after mutating the env var.

    The cache is invalidated by :func:`reset_pricing_cache` (typically only
    needed in tests; production reads the env once per process).
    """
    global _env_overrides_cache
    if _env_overrides_cache is not None:
        return _env_overrides_cache
    path = os.environ.get(PRICING_OVERRIDE_ENV)
    if not path:
        _env_overrides_cache = {}
        return _env_overrides_cache
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("pricing override %s unreadable: %s", path, exc)
        _env_overrides_cache = {}
        return _env_overrides_cache
    if not isinstance(data, dict):
        log.warning("pricing override %s is not a JSON object", path)
        _env_overrides_cache = {}
        return _env_overrides_cache
    _env_overrides_cache = {k: v for k, v in data.items() if isinstance(v, dict)}
    return _env_overrides_cache


def reset_pricing_cache() -> None:
    """Clear cached env overrides. Call after mutating ``LAYERLENS_PRICING_TABLE``."""
    global _env_overrides_cache
    _env_overrides_cache = None


def _lookup_rates(model: str, table: dict[str, dict[str, float]]) -> dict[str, float] | None:
    """Exact / dated-suffix / longest-prefix lookup (LAY-3330 AC).

    1. Exact match on ``model``.
    2. Strip a trailing dated suffix (``-YYYY-MM-DD`` or ``-YYYYMMDD``) and
       look up the base model name.
    3. Longest-prefix match: pick the longest table key ``K`` such that the
       requested model starts with ``K + "-"`` (disambiguates ``gpt-4o`` from
       ``gpt-4`` when both are in the table).
    """
    rates = table.get(model)
    if rates is not None:
        return rates
    stripped = _DATE_SUFFIX_RE.sub("", model)
    if stripped != model:
        rates = table.get(stripped)
        if rates is not None:
            return rates
    prefix_matches = [k for k in table if model.startswith(k + "-")]
    if prefix_matches:
        best = max(prefix_matches, key=len)
        return table[best]
    return None


def _resolve_rates(model: str, table: dict[str, dict[str, float]]) -> dict[str, float] | None:
    """Resolve rates, retrying against the bare model id when ``model`` is a
    Bedrock inference-profile ARN / region-prefixed id (LAY-3605)."""
    rates = _lookup_rates(model, table)
    if rates is not None:
        return rates
    normalized = _normalize_model_id(model)
    if normalized != model:
        return _lookup_rates(normalized, table)
    return None


def calculate_cost(
    model: str,
    usage: NormalizedTokenUsage,
    pricing_table: dict[str, dict[str, float]] | None = None,
    *,
    service_tier: Optional[str] = None,
) -> float | None:
    """Return USD cost for a model invocation, or ``None`` if model is unpriced.

    Resolution precedence: env-loaded overrides > caller-supplied
    ``pricing_table`` > bundled ``PRICING``. Each layer supports the same
    fuzzy date-suffix and longest-prefix fallback (LAY-3330).

    ``service_tier`` (flex/batch/priority/standard) scales the per-token rate
    (A2 / LAY-3626); cache-write tokens (``usage.cache_creation_tokens``) are
    priced at 1.25x input (A3). Both default to no-op so existing callers are
    unaffected.
    """
    rates: dict[str, float] | None = None
    env_overrides = _load_env_overrides()
    if env_overrides:
        rates = _resolve_rates(model, env_overrides)
    if rates is None:
        table = pricing_table if pricing_table is not None else PRICING
        rates = _resolve_rates(model, table)
    if rates is None:
        return None
    return _cost_from_rates(rates, model, usage, service_tier)


# Provider field -> override pricing table for the centralized chokepoint.
# Azure and Bedrock have their own rates (azure gpt-4o != openai gpt-4o); other
# providers (openai/anthropic/google/litellm/ollama/strands-on-non-bedrock) use
# the bundled PRICING with fuzzy matching. strands runs Bedrock models.
_PROVIDER_TABLES: dict[str, dict[str, dict[str, float]]] = {
    "azure": AZURE_PRICING,
    "azure_openai": AZURE_PRICING,
    "bedrock": BEDROCK_PRICING,
    "bedrock_agents": BEDROCK_PRICING,
    "strands": BEDROCK_PRICING,
}


def _payload_int(payload: dict, *keys: str) -> int:
    for k in keys:
        v = payload.get(k)
        if v is not None:
            try:
                return int(v)
            except (TypeError, ValueError):
                return 0
    return 0


def _payload_opt_int(payload: dict, *keys: str) -> Optional[int]:
    for k in keys:
        v = payload.get(k)
        if v is not None:
            try:
                return int(v)
            except (TypeError, ValueError):
                return None
    return None


def is_priced(model: Optional[str], provider: Optional[str] = None) -> bool:
    """True iff *model* resolves to a rate (env override / provider table /
    bundled PRICING, with fuzzy matching). Used by the schema lock (A11) to
    require a cost_usd on a priced cost.record — a priced model with no cost is
    a dropped price (fail closed), not a genuinely-unpriced local/custom model."""
    if not model or not isinstance(model, str):
        return False
    env = _load_env_overrides()
    if env and _resolve_rates(model, env) is not None:
        return True
    table = _PROVIDER_TABLES.get(str(provider or "").lower(), PRICING)
    return _resolve_rates(model, table) is not None


def usage_from_payload(payload: dict) -> NormalizedTokenUsage:
    """Map a ``cost.record`` payload's token keys onto :class:`NormalizedTokenUsage`.

    ONE mapping, shared by :func:`price_cost_record` and
    :func:`unpriced_token_count`, so what we price and what we call unpriced can
    never disagree about which key meant what — the same reason the cost formula
    itself lives in exactly one place.

    Accepts the provider vocabulary (``prompt_tokens``/``input_tokens``), the raw
    provider-wire names (``cache_creation_input_tokens``) and the framework flat
    vocabulary (``tokens_prompt``/``tokens_completion``/``tokens_total``) — all three
    reach the collector chokepoint.
    """
    return NormalizedTokenUsage(
        prompt_tokens=_payload_int(payload, "prompt_tokens", "input_tokens", "tokens_prompt"),
        completion_tokens=_payload_int(payload, "completion_tokens", "output_tokens", "tokens_completion"),
        total_tokens=_payload_int(payload, "total_tokens", "tokens_total"),
        cached_tokens=_payload_opt_int(payload, "cached_tokens", "cache_read_input_tokens"),
        cache_creation_tokens=_payload_opt_int(payload, "cache_creation_tokens", "cache_creation_input_tokens"),
    )


def priced_token_count(usage: NormalizedTokenUsage) -> int:
    """How many tokens :func:`_cost_from_rates` actually applies a rate to.

    Mirrors that formula's four legs EXACTLY — non-cached prompt, cached read,
    cache write, completion — and must be updated in lockstep with it. Pinned
    against the real formula in ``test_pricing.py``.

    The ``max(prompt - cached, 0)`` term is load-bearing, not defensive. A naive
    ``prompt + completion + cached + cache_creation`` double-counts the cached read
    (it is a SUBSET of ``prompt_tokens`` for both OpenAI and Anthropic), while a
    naive ``prompt + completion + cache_creation`` under-counts a turn reported as
    fully cached with ``prompt_tokens=0`` — which is a real shape in this repo's own
    test corpus, and would be reported as a 100% unpriced gap when in fact every
    token was priced through the cached leg.
    """
    cached = usage.cached_tokens or 0
    cache_creation = usage.cache_creation_tokens or 0
    non_cached = max(usage.prompt_tokens - cached, 0)
    return non_cached + cached + cache_creation + usage.completion_tokens


def unpriced_token_count(payload: dict) -> int:
    """Tokens the provider reported as billed that no rate was applied to.

    ``0`` when the priced dimensions account for the reported total, when no total
    was reported, or when the payload carries no usage at all — i.e. silence means
    "nothing detected", never "nothing wrong". Detection only: the caller records
    this, it does NOT feed a price. See :func:`_cost_from_rates` for why reading
    ``reasoning_tokens`` instead would double-bill.

    HONEST LIMIT: this can only see a gap a provider REPORTS. Anthropic sends no
    total at all — ``NormalizedTokenUsage._auto_total`` derives it as
    ``prompt + completion`` — so a token Anthropic billed outside ``output_tokens``
    is invisible here by construction. The detector is not a completeness claim.
    """
    usage = usage_from_payload(payload)
    if not usage.total_tokens:
        return 0
    return max(usage.total_tokens - priced_token_count(usage), 0)


def price_cost_record(payload: dict) -> Optional[float]:
    """Centralized price-on-emit (A1 / LAY-3626): compute ``cost_usd`` for a
    ``cost.record`` payload from its own fields.

    This is the ONE place pricing happens at emit. Called from
    ``TraceCollector.emit`` for every ``cost.record`` regardless of which adapter
    emitted it, so a path that forgot to compute cost still ships a correct
    ``cost_usd`` (A11 fail-closed) and tier (A2) / cache-write (A3) pricing
    applies uniformly. Returns ``None`` for an unpriced model — the caller then
    leaves the payload's ``cost_usd`` untouched (genuinely unpriced models stay
    ``None``; we never fabricate a price).

    Handles both the normalized field names (``cache_creation_tokens``) and the
    raw provider-wire names (``cache_creation_input_tokens`` / ``input_tokens``).

    Returns ``None`` for a token SHAPE that cannot be priced as well as for an
    unpriced model (LAY-3622 / A4b) — see :func:`_cost_from_rates`. Unlike that
    formula, this layer can see whether a token key is ABSENT rather than zero, so
    a payload carrying no usage at all is unpriceable here rather than $0.00.
    """
    model = payload.get("model")
    if not model or not isinstance(model, str):
        return None
    if not any(
        payload.get(k) is not None
        for k in (
            "prompt_tokens",
            "input_tokens",
            "tokens_prompt",
            "completion_tokens",
            "output_tokens",
            "tokens_completion",
            "total_tokens",
            "tokens_total",
            "cached_tokens",
            "cache_read_input_tokens",
            "cache_creation_tokens",
            "cache_creation_input_tokens",
        )
    ):
        # No usage was reported at all. There is nothing to price, and a 0.0 here
        # would present an upstream reporting bug as a real, derived cost.
        return None
    usage = usage_from_payload(payload)
    provider = str(payload.get("provider") or "").lower()
    table = _PROVIDER_TABLES.get(provider, PRICING)
    return calculate_cost(model, usage, table, service_tier=payload.get("service_tier"))


@dataclass
class CostRecord:
    """Result of :meth:`PricingTable.calculate_cost`.

    ``cost_usd`` is ``None`` only when the model isn't priced. Callers can
    forward this object directly into the ``cost.record`` event payload.
    """

    cost_usd: Optional[float]
    model: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0


class PricingTable:
    """Per-model LLM pricing with fuzzy matching and configurable overrides.

    Per LAY-3330 acceptance criteria, callers can:

    * Use ``PricingTable()`` to get the bundled defaults (GPT-4o, GPT-4o-mini,
      GPT-4-turbo, GPT-4, GPT-3.5-turbo, o1, o1-mini, o3, o3-mini, plus Claude,
      Gemini, Llama, Mistral families).
    * Pass an explicit ``table=`` to fully replace the defaults (e.g. for
      pre-release model pricing).
    * Load overrides from a JSON file via :meth:`from_json_file` or via the
      ``LAYERLENS_PRICING_TABLE`` env var (no code changes needed).
    * Call :meth:`calculate_cost` with ``(model, input_tokens, output_tokens)``
      to get a :class:`CostRecord`.

    Fuzzy matching: ``gpt-4o-2024-08-06`` resolves to ``gpt-4o``,
    ``claude-3-5-sonnet-20990101`` resolves to ``claude-3-5-sonnet``. Falls
    back to longest-prefix match for unrecognised dated variants.
    """

    def __init__(
        self,
        table: Optional[dict[str, dict[str, float]]] = None,
        *,
        respect_env_overrides: bool = True,
    ) -> None:
        self._table: dict[str, dict[str, float]] = dict(table) if table is not None else dict(PRICING)
        self._respect_env_overrides = respect_env_overrides

    @classmethod
    def from_default(cls) -> "PricingTable":
        """Build a table populated with the bundled defaults."""
        return cls(table=PRICING)

    @classmethod
    def from_dict(cls, table: dict[str, dict[str, float]]) -> "PricingTable":
        """Build a table from a caller-provided dict (replaces defaults)."""
        return cls(table=table)

    @classmethod
    def from_json_file(cls, path: str) -> "PricingTable":
        """Build a table by loading rates from a JSON file at ``path``."""
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"pricing JSON at {path} must be an object, got {type(data).__name__}")
        return cls(table={k: v for k, v in data.items() if isinstance(v, dict)})

    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        *,
        cached_tokens: int = 0,
    ) -> CostRecord:
        """Compute the USD cost for one model invocation.

        Returns a :class:`CostRecord` with ``cost_usd=None`` for unknown
        models, never raises.
        """
        usage = NormalizedTokenUsage(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cached_tokens=cached_tokens or None,
        )
        cost = calculate_cost(
            model,
            usage,
            self._table if self._respect_env_overrides else self._table,
        )
        # ``calculate_cost`` already applies env overrides at the top of its
        # resolution chain when ``respect_env_overrides`` is True, which is
        # the only mode we currently expose (the flag is reserved for tests
        # that need deterministic isolation).
        if not self._respect_env_overrides:
            # Bypass env: resolve against the local table directly.
            rates = _resolve_rates(model, self._table)
            cost = _compute_cost_from_rates(rates, model, usage) if rates is not None else None
        return CostRecord(
            cost_usd=cost,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
        )

    def models(self) -> list[str]:
        """List the model names that have explicit rates in this table."""
        return list(self._table.keys())

    def has_model(self, model: str) -> bool:
        """True if ``model`` resolves (exact or fuzzy) to a rate in the table."""
        return _resolve_rates(model, self._table) is not None

    def as_dict(self) -> dict[str, dict[str, float]]:
        """Return a copy of the underlying rate dict."""
        return dict(self._table)


def _compute_cost_from_rates(rates: dict[str, float], model: str, usage: NormalizedTokenUsage) -> Optional[float]:
    """Bare cost formula, used by :class:`PricingTable` when bypassing env.

    ``None`` when the token shape cannot be priced — see :func:`_cost_from_rates`.
    ``CostRecord.cost_usd`` is already ``Optional``, so this propagates cleanly.
    """
    return _cost_from_rates(rates, model, usage)
