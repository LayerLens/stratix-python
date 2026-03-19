"""Scenario registry — maps scenario names to classes."""

from __future__ import annotations

from .base import BaseScenario
from .customer_service import CustomerServiceScenario
from .it_helpdesk import ITHelpdeskScenario
from .knowledge_faq import KnowledgeFAQScenario
from .order_management import OrderManagementScenario
from .sales import SalesScenario

_SCENARIO_REGISTRY: dict[str, type[BaseScenario]] = {
    "customer_service": CustomerServiceScenario,
    "sales": SalesScenario,
    "order_management": OrderManagementScenario,
    "knowledge_faq": KnowledgeFAQScenario,
    "it_helpdesk": ITHelpdeskScenario,
}


def get_scenario(name: str) -> BaseScenario:
    """Get a scenario instance by name."""
    cls = _SCENARIO_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown scenario: {name}. Available: {list(_SCENARIO_REGISTRY.keys())}"
        )
    return cls()


def list_scenarios() -> list[str]:
    """List all registered scenario names."""
    return sorted(_SCENARIO_REGISTRY.keys())


def register_scenario(name: str, scenario_class: type[BaseScenario]) -> None:
    """Register a custom scenario."""
    _SCENARIO_REGISTRY[name] = scenario_class
