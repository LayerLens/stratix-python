"""Customer service scenario."""

from .base import AgentProfile, BaseScenario


class CustomerServiceScenario(BaseScenario):
    @property
    def name(self) -> str:
        return "customer_service"

    @property
    def topics(self) -> list[str]:
        return [
            "Shipping_Delay",
            "Account_Access",
            "Product_Issue",
            "Billing_Dispute",
            "Escalation",
        ]

    @property
    def agents(self) -> list[AgentProfile]:
        return [
            AgentProfile(
                name="Case_Resolution_Agent",
                description="Customer service agent specializing in case resolution",
                tools=["Get_Order_Details", "Get_Customer_History"],
                eval_dimensions=["factual_accuracy", "helpfulness"],
            ),
            AgentProfile(
                name="Customer_Support_Agent",
                description="Frontline customer support agent",
                tools=["Verify_Customer_Identity", "Unlock_Account"],
            ),
            AgentProfile(
                name="Escalation_Agent",
                description="Senior escalation agent for complex cases",
                tools=["Get_Case_History", "Apply_Account_Credit"],
            ),
        ]
