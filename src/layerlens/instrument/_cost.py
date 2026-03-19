"""
STRATIX Cost Tracking

From Step 4 specification:
- Cost tracking for tokens and API calls
- Emits cost.record events
- Handles unavailable costs gracefully
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from layerlens.instrument.schema.events.cross_cutting import CostRecordEvent
from layerlens.instrument._context import get_current_context

if TYPE_CHECKING:
    from layerlens.instrument._core import STRATIX


class CostTracker:
    """
    Tracks costs during agent execution.

    Supports multiple cost types:
    - token: LLM token costs
    - api_call: External API call costs
    - compute: Compute resource costs
    - storage: Storage costs
    - other: Miscellaneous costs
    """

    def __init__(self, stratix: "STRATIX"):
        """
        Initialize the cost tracker.

        Args:
            stratix: The STRATIX instance
        """
        self._stratix = stratix
        self._accumulated_costs: dict[str, float] = {}
        self._cost_records: list[CostRecordEvent] = []

    def record(
        self,
        tokens: int | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        api_cost_usd: float | str | None = None,
        infra_cost_usd: float | str | None = None,
        tool_calls: int | None = None,
    ) -> None:
        """
        Record a cost and emit a cost.record event.

        Args:
            tokens: Total tokens consumed
            prompt_tokens: Prompt tokens
            completion_tokens: Completion tokens
            api_cost_usd: API cost in USD (or 'unavailable')
            infra_cost_usd: Infrastructure cost (or 'unavailable')
            tool_calls: Number of tool calls
        """
        ctx = get_current_context()
        if ctx is None:
            return

        # Create cost record event using the existing schema
        event = CostRecordEvent.create(
            tokens=tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            api_cost_usd=api_cost_usd,
            infra_cost_usd=infra_cost_usd,
            tool_calls=tool_calls,
        )

        # Track accumulated costs
        if api_cost_usd is not None and isinstance(api_cost_usd, (int, float)):
            self._accumulated_costs["api:USD"] = self._accumulated_costs.get("api:USD", 0) + api_cost_usd
        if infra_cost_usd is not None and isinstance(infra_cost_usd, (int, float)):
            self._accumulated_costs["infra:USD"] = self._accumulated_costs.get("infra:USD", 0) + infra_cost_usd

        self._cost_records.append(event)
        self._stratix._emit_event(ctx, event)

    def record_tokens(
        self,
        provider: str,
        model: str,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        cost_per_1k_prompt: float | None = None,
        cost_per_1k_completion: float | None = None,
    ) -> None:
        """
        Record token costs from an LLM call.

        Args:
            provider: LLM provider name
            model: Model name
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            total_tokens: Total tokens (calculated if not provided)
            cost_per_1k_prompt: Cost per 1000 prompt tokens
            cost_per_1k_completion: Cost per 1000 completion tokens
        """
        # Calculate total if not provided
        if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens

        # Calculate cost if pricing is available
        api_cost_usd: float | str | None = "unavailable"
        if cost_per_1k_prompt is not None and cost_per_1k_completion is not None:
            prompt_cost = (prompt_tokens or 0) * cost_per_1k_prompt / 1000
            completion_cost = (completion_tokens or 0) * cost_per_1k_completion / 1000
            api_cost_usd = prompt_cost + completion_cost

        self.record(
            tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            api_cost_usd=api_cost_usd,
        )

    def record_tool_call(self) -> None:
        """Record a tool call for cost tracking."""
        self.record(tool_calls=1)

    def get_total(self, currency: str = "USD") -> float:
        """
        Get the total accumulated cost for a currency.

        Args:
            currency: Currency code

        Returns:
            Total accumulated cost
        """
        total = 0.0
        for key, amount in self._accumulated_costs.items():
            if key.endswith(f":{currency}"):
                total += amount
        return total

    def get_breakdown(self) -> dict[str, float]:
        """
        Get a breakdown of costs by type.

        Returns:
            Dictionary mapping cost type to total amount
        """
        return dict(self._accumulated_costs)

    def reset(self) -> None:
        """Reset accumulated costs and records."""
        self._accumulated_costs = {}
        self._cost_records = []


# Module-level convenience functions


def record_cost(
    tokens: int | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    api_cost_usd: float | str | None = None,
    infra_cost_usd: float | str | None = None,
    tool_calls: int | None = None,
) -> None:
    """
    Record a cost using the current context's cost tracker.

    Args:
        tokens: Total tokens consumed
        prompt_tokens: Prompt tokens
        completion_tokens: Completion tokens
        api_cost_usd: API cost in USD (or 'unavailable')
        infra_cost_usd: Infrastructure cost (or 'unavailable')
        tool_calls: Number of tool calls
    """
    ctx = get_current_context()
    if ctx is None:
        raise RuntimeError("No active STRATIX context. Call start_trial() first.")

    # Get or create cost tracker from STRATIX instance
    stratix = ctx.stratix
    if not hasattr(stratix, "_cost_tracker"):
        stratix._cost_tracker = CostTracker(stratix)

    stratix._cost_tracker.record(
        tokens=tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        api_cost_usd=api_cost_usd,
        infra_cost_usd=infra_cost_usd,
        tool_calls=tool_calls,
    )


def record_token_cost(
    provider: str,
    model: str,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    cost_per_1k_prompt: float | None = None,
    cost_per_1k_completion: float | None = None,
) -> None:
    """
    Record token costs from an LLM call.

    Args:
        provider: LLM provider name
        model: Model name
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
        total_tokens: Total tokens
        cost_per_1k_prompt: Cost per 1000 prompt tokens
        cost_per_1k_completion: Cost per 1000 completion tokens
    """
    ctx = get_current_context()
    if ctx is None:
        raise RuntimeError("No active STRATIX context. Call start_trial() first.")

    stratix = ctx.stratix
    if not hasattr(stratix, "_cost_tracker"):
        stratix._cost_tracker = CostTracker(stratix)

    stratix._cost_tracker.record_tokens(
        provider=provider,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost_per_1k_prompt=cost_per_1k_prompt,
        cost_per_1k_completion=cost_per_1k_completion,
    )
