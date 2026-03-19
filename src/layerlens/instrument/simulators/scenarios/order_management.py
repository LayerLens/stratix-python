"""Order management scenario."""

from .base import AgentProfile, BaseScenario


class OrderManagementScenario(BaseScenario):
    @property
    def name(self) -> str:
        return "order_management"

    @property
    def topics(self) -> list[str]:
        return [
            "Order_Tracking",
            "Payment_Problem",
            "Return_Request",
            "Cancellation",
            "Split_Shipment",
        ]

    @property
    def agents(self) -> list[AgentProfile]:
        return [
            AgentProfile(
                name="Order_Management_Agent",
                description="Order management and tracking agent",
                tools=["Track_Order", "Get_Shipment_Details"],
                eval_dimensions=["factual_accuracy"],
            ),
            AgentProfile(
                name="Payment_Processing_Agent",
                description="Payment processing and refund agent",
                tools=["Check_Payment_Status", "Process_Refund"],
            ),
            AgentProfile(
                name="Returns_Agent",
                description="Returns specialist agent",
                tools=["Check_Return_Eligibility", "Create_Return_Label"],
            ),
        ]
