"""Scenario classes for 5 business scenarios."""

from .base import AgentProfile, BaseScenario
from .customer_service import CustomerServiceScenario
from .it_helpdesk import ITHelpdeskScenario
from .knowledge_faq import KnowledgeFAQScenario
from .order_management import OrderManagementScenario
from .registry import get_scenario, list_scenarios, register_scenario
from .sales import SalesScenario

__all__ = [
    "BaseScenario",
    "AgentProfile",
    "CustomerServiceScenario",
    "SalesScenario",
    "OrderManagementScenario",
    "KnowledgeFAQScenario",
    "ITHelpdeskScenario",
    "get_scenario",
    "list_scenarios",
    "register_scenario",
]
