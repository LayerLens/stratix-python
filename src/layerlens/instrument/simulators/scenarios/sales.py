"""Sales scenario."""

from .base import AgentProfile, BaseScenario


class SalesScenario(BaseScenario):
    @property
    def name(self) -> str:
        return "sales"

    @property
    def topics(self) -> list[str]:
        return [
            "Trial_Extension",
            "Pricing_Inquiry",
            "Demo_Request",
            "Competitor_Comparison",
            "ROI_Analysis",
        ]

    @property
    def agents(self) -> list[AgentProfile]:
        return [
            AgentProfile(
                name="Sales_Assistant_Agent",
                description="AI sales assistant for product evaluation",
                tools=["Get_Trial_Info", "Extend_Trial", "Get_Pricing_Tiers", "Generate_Quote"],
                eval_dimensions=["relevance", "helpfulness"],
            ),
            AgentProfile(
                name="Pricing_Agent",
                description="Pricing specialist agent",
                tools=["Get_Pricing_Tiers", "Generate_Quote"],
            ),
            AgentProfile(
                name="Demo_Coordinator_Agent",
                description="Demo coordination agent",
                tools=["Check_Calendar_Availability", "Schedule_Demo"],
            ),
        ]
