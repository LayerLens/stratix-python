"""IT helpdesk scenario."""

from .base import AgentProfile, BaseScenario


class ITHelpdeskScenario(BaseScenario):
    @property
    def name(self) -> str:
        return "it_helpdesk"

    @property
    def topics(self) -> list[str]:
        return [
            "Hardware_Issue",
            "Security_Incident",
            "Software_Install",
            "VPN_Access",
            "Password_Reset",
        ]

    @property
    def agents(self) -> list[AgentProfile]:
        return [
            AgentProfile(
                name="IT_Support_Agent",
                description="IT helpdesk agent for hardware and software issues",
                tools=["Get_Asset_Info", "Create_Service_Ticket"],
                eval_dimensions=["factual_accuracy", "compliance"],
            ),
            AgentProfile(
                name="Security_Agent",
                description="IT security agent for incident response",
                tools=["Create_Security_Incident", "Scan_Device"],
            ),
            AgentProfile(
                name="Network_Agent",
                description="Network support agent for VPN and connectivity",
                tools=["Check_VPN_Status", "Update_VPN_Profile"],
            ),
        ]
