"""Knowledge FAQ scenario templates.

Topics: Policy_Question, Integration_Question, API_Usage, Best_Practices, Troubleshooting
"""

KNOWLEDGE_FAQ_TEMPLATES = {
    "scenario": "knowledge_faq",
    "agent_names": ["Knowledge_Base_Agent", "Technical_Support_Agent", "Documentation_Agent"],
    "system_prompts": {
        "Knowledge_Base_Agent": (
            "You are a knowledge base agent that answers questions using the company's "
            "documentation and FAQ database. Always cite the source document when providing "
            "answers. If the answer isn't in the knowledge base, acknowledge the gap and "
            "suggest contacting support."
        ),
        "Technical_Support_Agent": (
            "You are a technical support agent specializing in integration and API questions. "
            "Provide code examples when helpful and guide users through implementation steps."
        ),
        "Documentation_Agent": (
            "You are a documentation agent that helps users find and understand relevant "
            "documentation pages. Provide direct links and summaries."
        ),
    },
    "topics": {
        "Policy_Question": {
            "user_messages": [
                "What is your data retention policy? How long do you keep our traces?",
                "Does your platform comply with SOC 2 and GDPR requirements?",
                "What happens to our data if we cancel our subscription?",
            ],
            "agent_responses": [
                "Let me look up our data retention policy for you.",
                "Our standard data retention is 90 days for trace data and 1 year for aggregated metrics. Enterprise plans can customize retention up to 3 years. All data is encrypted at rest using AES-256.",
                "Yes, we are SOC 2 Type II certified and GDPR compliant. I can share our compliance documentation and DPA (Data Processing Agreement) for your review.",
            ],
            "tools": {
                "Search_Knowledge_Base": {
                    "input": {"query": "data retention policy", "category": "compliance"},
                    "output": {
                        "results": [
                            {"doc_id": "KB-201", "title": "Data Retention Policy", "relevance": 0.95},
                            {"doc_id": "KB-205", "title": "GDPR Compliance Guide", "relevance": 0.82},
                        ],
                        "answer_snippet": "Standard retention: 90 days for traces, 1 year for metrics.",
                    },
                },
            },
        },
        "Integration_Question": {
            "user_messages": [
                "How do I integrate your SDK with our existing LangChain setup?",
                "Can your platform receive traces from our OpenTelemetry collector?",
                "What's the recommended way to instrument a multi-agent CrewAI application?",
            ],
            "agent_responses": [
                "Let me pull up the integration guide for LangChain.",
                "Integrating with LangChain is straightforward. You need to install our SDK, create an adapter instance, and call connect(). Here's a quick example showing the setup.",
                "Yes, we natively support OTel OTLP ingestion. You can configure your OTel collector to forward traces to our endpoint. We support both gRPC and HTTP protocols.",
            ],
            "tools": {
                "Search_Documentation": {
                    "input": {"query": "langchain integration", "section": "sdk"},
                    "output": {
                        "results": [
                            {"doc_id": "DOC-101", "title": "LangChain Adapter Guide", "url": "/docs/sdk/langchain"},
                            {"doc_id": "DOC-102", "title": "Quick Start", "url": "/docs/quickstart"},
                        ],
                        "code_example": "from stratix import STRATIX\nstratix = STRATIX()\nadapter = stratix.adapters.langchain()",
                    },
                },
            },
        },
        "API_Usage": {
            "user_messages": [
                "What are the rate limits for your REST API?",
                "How do I authenticate API requests? I need to set up programmatic access.",
                "Is there a way to query traces via the API? I need to build a custom dashboard.",
            ],
            "agent_responses": [
                "Let me look up our API rate limits and usage information.",
                "Our API rate limits are: Starter: 100 req/min, Professional: 1000 req/min, Enterprise: 10000 req/min. All requests require an API key in the Authorization header.",
                "Yes, our Trace Query API supports filtering by date range, trace ID, event type, and custom metadata. You can use GraphQL or REST endpoints. Let me show you an example query.",
            ],
            "tools": {
                "Get_API_Documentation": {
                    "input": {"endpoint": "rate_limits"},
                    "output": {
                        "rate_limits": {
                            "starter": {"requests_per_minute": 100, "burst": 20},
                            "professional": {"requests_per_minute": 1000, "burst": 100},
                            "enterprise": {"requests_per_minute": 10000, "burst": 500},
                        },
                        "auth_method": "Bearer token",
                    },
                },
            },
        },
        "Best_Practices": {
            "user_messages": [
                "What's the recommended approach for prompt versioning and testing?",
                "How should we structure our evaluation pipeline for production?",
                "What are the best practices for reducing LLM API costs?",
            ],
            "agent_responses": [
                "Let me look up our best practices guide for prompt management.",
                "We recommend a three-stage approach: (1) Version prompts in your codebase alongside code, (2) Use our A/B replay system to test changes against historical traces, (3) Set up automated evaluation gates using AI judges before deploying prompt changes to production.",
                "For cost optimization, the top three practices are: (1) Implement prompt caching for repeated system prompts, (2) Use our token analysis dashboard to identify verbose prompts, (3) Set up cost alerts per-model and per-team.",
            ],
            "tools": {
                "Search_Best_Practices": {
                    "input": {"topic": "prompt_versioning"},
                    "output": {
                        "guide_id": "BP-301",
                        "title": "Prompt Management Best Practices",
                        "sections": ["Versioning", "Testing", "Deployment", "Rollback"],
                        "key_recommendation": "Use replay-based A/B testing before production deployment.",
                    },
                },
            },
        },
        "Troubleshooting": {
            "user_messages": [
                "My traces aren't showing up in the dashboard. I confirmed the SDK is installed.",
                "I'm getting a 'connection refused' error when trying to export traces.",
                "The latency numbers in the dashboard don't match what I see in my application logs.",
            ],
            "agent_responses": [
                "Let me help you troubleshoot the missing traces. There are a few common causes.",
                "First, let's verify your configuration. The most common cause of missing traces is an incorrect API endpoint or expired API key. Can you check your STRATIX_API_KEY environment variable?",
                "The 'connection refused' error usually indicates the exporter endpoint isn't reachable. Let me walk you through the connectivity check steps.",
            ],
            "tools": {
                "Run_Diagnostics": {
                    "input": {"check_type": "connectivity", "endpoint": "https://api.stratix.example.com"},
                    "output": {
                        "endpoint_reachable": True,
                        "latency_ms": 45,
                        "api_key_valid": False,
                        "error": "API key expired on 2024-12-01",
                        "recommendation": "Generate a new API key from Settings > API Keys",
                    },
                },
            },
        },
    },
}
