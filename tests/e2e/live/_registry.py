"""Per-provider registry: credentials, scenario runner, and expected-event contract.

Groupings reflect what the adapters actually emit (verified against the source):

- **Tool group** (emit ``tool.call``): anthropic, openai, azure_openai, google_vertex.
  litellm is intentionally NOT here — it doesn't override ``extract_tool_calls``.
- **Chat group** (no ``tool.call``): ollama, bedrock, litellm.

Cost caveats baked into the contract:
- ollama is local: it emits ``cost.record`` but ``cost_usd`` is ``None`` (its models
  aren't in the pricing table), so we don't require a priced cost.
- azure/bedrock price against their own override tables.
"""

from __future__ import annotations

import os
from typing import Tuple, Callable, Optional
from dataclasses import dataclass

from layerlens.instrument.adapters.providers.pricing import PRICING, AZURE_PRICING, BEDROCK_PRICING

from . import _scenarios


@dataclass(frozen=True)
class Contract:
    """What the local trace payload must contain for the default/redaction flow."""

    requires_tool_call: bool
    requires_cost_record: bool
    cost_priced: bool  # if True: every cost.record must have cost_usd not None and > 0
    pricing_table: Optional[dict] = None  # used to recompute & cross-check cost_usd (None -> bundled PRICING)
    min_events: int = 3
    max_events: int = 15
    supports_streaming: bool = False


@dataclass(frozen=True)
class ProviderCase:
    id: str
    import_name: str  # python module to importorskip
    runner: Callable[[str], None]
    contract: Contract
    variants: Tuple[str, ...]
    required_env: Tuple[str, ...] = ()  # all must be present
    any_of_env: Tuple[str, ...] = ()  # at least one must be present


_TOOL_VARIANTS = ("default", "streaming", "error", "redaction")
_CHAT_VARIANTS = ("default", "error", "redaction")
# openai + anthropic additionally drive the canonical tool loop / streamed call
# through their async clients (AsyncOpenAI / AsyncAnthropic) — T1 follow-through
# for the async-routing fix (N5). Same contracts as default / streaming.
_ASYNC_TOOL_VARIANTS = _TOOL_VARIANTS + ("async", "async-streaming")


PROVIDERS: Tuple[ProviderCase, ...] = (
    ProviderCase(
        id="anthropic",
        import_name="anthropic",
        runner=_scenarios.run_anthropic,
        required_env=("ANTHROPIC_API_KEY",),
        variants=_ASYNC_TOOL_VARIANTS,
        contract=Contract(
            requires_tool_call=True,
            requires_cost_record=True,
            cost_priced=True,
            pricing_table=None,
            min_events=4,
            supports_streaming=True,
        ),
    ),
    ProviderCase(
        id="openai",
        import_name="openai",
        runner=_scenarios.run_openai,
        required_env=("OPENAI_API_KEY",),
        variants=_ASYNC_TOOL_VARIANTS,
        contract=Contract(
            requires_tool_call=True,
            requires_cost_record=True,
            cost_priced=True,
            pricing_table=None,
            min_events=4,
            supports_streaming=True,
        ),
    ),
    ProviderCase(
        id="azure_openai",
        import_name="openai",
        runner=_scenarios.run_azure_openai,
        required_env=("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"),
        variants=_TOOL_VARIANTS,
        contract=Contract(
            requires_tool_call=True,
            requires_cost_record=True,
            cost_priced=True,
            pricing_table=AZURE_PRICING,
            min_events=4,
            supports_streaming=True,
        ),
    ),
    ProviderCase(
        id="google_vertex",
        import_name="vertexai",
        runner=_scenarios.run_google_vertex,
        any_of_env=("GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_CLOUD_PROJECT"),
        variants=_TOOL_VARIANTS,
        contract=Contract(
            requires_tool_call=True,
            requires_cost_record=True,
            # Gemini is priced, but the function-call turn's usage shape varies;
            # tolerate None here and rely on the recompute cross-check for non-None costs.
            cost_priced=False,
            pricing_table=None,
            min_events=4,
            supports_streaming=True,
        ),
    ),
    ProviderCase(
        id="ollama",
        import_name="ollama",
        runner=_scenarios.run_ollama,
        required_env=("OLLAMA_HOST",),  # set this (e.g. http://localhost:11434) to opt ollama in
        variants=_CHAT_VARIANTS,
        contract=Contract(
            requires_tool_call=False,
            requires_cost_record=False,  # emits cost.record but cost_usd is None (local, unpriced)
            cost_priced=False,
            pricing_table=None,
            min_events=3,
        ),
    ),
    ProviderCase(
        id="bedrock",
        import_name="boto3",
        runner=_scenarios.run_bedrock,
        any_of_env=("AWS_ACCESS_KEY_ID", "AWS_PROFILE"),
        variants=_CHAT_VARIANTS,
        contract=Contract(
            requires_tool_call=False,
            requires_cost_record=True,
            cost_priced=True,
            pricing_table=BEDROCK_PRICING,
            min_events=3,
        ),
    ),
    ProviderCase(
        id="litellm",
        import_name="litellm",
        runner=_scenarios.run_litellm,
        any_of_env=("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LITELLM_API_KEY"),
        variants=_CHAT_VARIANTS,
        contract=Contract(
            requires_tool_call=False,
            requires_cost_record=True,
            cost_priced=True,
            pricing_table=None,
            min_events=3,
        ),
    ),
)


def resolve_pricing_table(case: ProviderCase) -> dict:
    """The table used to cross-check emitted cost_usd (matches what the adapter used)."""
    return case.contract.pricing_table if case.contract.pricing_table is not None else PRICING


def missing_credentials(case: ProviderCase) -> Optional[str]:
    """Return a skip reason if the provider's credentials aren't present, else None."""
    missing = [key for key in case.required_env if not os.environ.get(key)]
    if missing:
        return f"missing env: {', '.join(missing)}"
    if case.any_of_env and not any(os.environ.get(key) for key in case.any_of_env):
        return f"none of {', '.join(case.any_of_env)} set"
    return None
