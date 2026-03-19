"""Domain-specific content templates for 5 scenarios."""

from .customer_service import CUSTOMER_SERVICE_TEMPLATES
from .it_helpdesk import IT_HELPDESK_TEMPLATES
from .knowledge_faq import KNOWLEDGE_FAQ_TEMPLATES
from .order_management import ORDER_MANAGEMENT_TEMPLATES
from .sales import SALES_TEMPLATES

SCENARIO_TEMPLATES = {
    "customer_service": CUSTOMER_SERVICE_TEMPLATES,
    "sales": SALES_TEMPLATES,
    "order_management": ORDER_MANAGEMENT_TEMPLATES,
    "knowledge_faq": KNOWLEDGE_FAQ_TEMPLATES,
    "it_helpdesk": IT_HELPDESK_TEMPLATES,
}

__all__ = [
    "SCENARIO_TEMPLATES",
    "CUSTOMER_SERVICE_TEMPLATES",
    "SALES_TEMPLATES",
    "ORDER_MANAGEMENT_TEMPLATES",
    "KNOWLEDGE_FAQ_TEMPLATES",
    "IT_HELPDESK_TEMPLATES",
]
