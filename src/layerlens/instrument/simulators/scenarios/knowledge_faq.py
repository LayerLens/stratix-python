"""Knowledge FAQ scenario."""

from .base import AgentProfile, BaseScenario


class KnowledgeFAQScenario(BaseScenario):
    @property
    def name(self) -> str:
        return "knowledge_faq"

    @property
    def topics(self) -> list[str]:
        return [
            "Policy_Question",
            "Integration_Question",
            "API_Usage",
            "Best_Practices",
            "Troubleshooting",
        ]

    @property
    def agents(self) -> list[AgentProfile]:
        return [
            AgentProfile(
                name="Knowledge_Base_Agent",
                description="Knowledge base agent for documentation queries",
                tools=["Search_Knowledge_Base", "Search_Documentation"],
                eval_dimensions=["factual_accuracy", "relevance"],
            ),
            AgentProfile(
                name="Technical_Support_Agent",
                description="Technical support for integration and API questions",
                tools=["Get_API_Documentation"],
            ),
            AgentProfile(
                name="Documentation_Agent",
                description="Documentation navigation agent",
                tools=["Search_Best_Practices", "Run_Diagnostics"],
            ),
        ]
