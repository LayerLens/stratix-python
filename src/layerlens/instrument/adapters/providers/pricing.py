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
}


def _cached_token_discount(model: str) -> float:
    """Cached-token rate as a fraction of the input price.

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


def _resolve_rates(model: str, table: dict[str, dict[str, float]]) -> dict[str, float] | None:
    """Look up rates with fuzzy fallback (LAY-3330 AC).

    Resolution order:
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


def calculate_cost(
    model: str,
    usage: NormalizedTokenUsage,
    pricing_table: dict[str, dict[str, float]] | None = None,
) -> float | None:
    """Return USD cost for a model invocation, or ``None`` if model is unpriced.

    Resolution precedence: env-loaded overrides > caller-supplied
    ``pricing_table`` > bundled ``PRICING``. Each layer supports the same
    fuzzy date-suffix and longest-prefix fallback (LAY-3330).
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

    input_rate = rates.get("input", 0.0)
    output_rate = rates.get("output", 0.0)

    prompt_tokens = usage.prompt_tokens
    cached = usage.cached_tokens or 0

    non_cached = max(prompt_tokens - cached, 0)
    cached_rate = input_rate * _cached_token_discount(model)

    cost = (
        (non_cached * input_rate / 1000)
        + (cached * cached_rate / 1000)
        + (usage.completion_tokens * output_rate / 1000)
    )
    return round(cost, 8)


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


def _compute_cost_from_rates(rates: dict[str, float], model: str, usage: NormalizedTokenUsage) -> float:
    """Bare cost formula, used by :class:`PricingTable` when bypassing env."""
    input_rate = rates.get("input", 0.0)
    output_rate = rates.get("output", 0.0)
    cached = usage.cached_tokens or 0
    non_cached = max(usage.prompt_tokens - cached, 0)
    cached_rate = input_rate * _cached_token_discount(model)
    cost = (
        (non_cached * input_rate / 1000)
        + (cached * cached_rate / 1000)
        + (usage.completion_tokens * output_rate / 1000)
    )
    return round(cost, 8)
