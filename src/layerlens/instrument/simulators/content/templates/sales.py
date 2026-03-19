"""Sales scenario templates.

Topics: Trial_Extension, Pricing_Inquiry, Demo_Request, Competitor_Comparison, ROI_Analysis
"""

SALES_TEMPLATES = {
    "scenario": "sales",
    "agent_names": ["Sales_Assistant_Agent", "Pricing_Agent", "Demo_Coordinator_Agent"],
    "system_prompts": {
        "Sales_Assistant_Agent": (
            "You are an AI sales assistant helping potential customers evaluate the product. "
            "Use available tools to look up pricing, feature comparisons, and customer data. "
            "Be consultative rather than pushy. Focus on understanding the customer's needs "
            "and demonstrating value."
        ),
        "Pricing_Agent": (
            "You are a pricing specialist agent. Help customers understand pricing tiers, "
            "volume discounts, and enterprise agreements. You can generate custom quotes "
            "based on the customer's requirements."
        ),
        "Demo_Coordinator_Agent": (
            "You are a demo coordination agent. Schedule product demonstrations, "
            "prepare personalized demo environments, and follow up with prospects."
        ),
    },
    "topics": {
        "Trial_Extension": {
            "user_messages": [
                "Our trial is expiring in 2 days and we haven't had a chance to fully evaluate the platform. Can we get an extension?",
                "We need more time with the trial. Our team has been busy and we've only used about half the features.",
                "Is it possible to extend our free trial by another 2 weeks? We're really interested but need more evaluation time.",
            ],
            "agent_responses": [
                "I'd be happy to help with a trial extension. Let me pull up your account to see your current usage and trial details.",
                "I can see your team has been actively exploring the platform. I've extended your trial by 14 days. I'd also recommend scheduling a guided walkthrough to help you get the most out of the remaining features.",
                "Your trial has been extended. During the extension, you'll also have access to our premium features so you can evaluate the full platform. Would you like me to set up a call with a solutions engineer?",
            ],
            "tools": {
                "Get_Trial_Info": {
                    "input": {"account_id": "ACC-{id}"},
                    "output": {
                        "account_id": "ACC-{id}",
                        "trial_start": "2024-11-15",
                        "trial_end": "2024-12-15",
                        "features_used": 12,
                        "total_features": 25,
                        "active_users": 5,
                        "api_calls": 1247,
                    },
                },
                "Extend_Trial": {
                    "input": {"account_id": "ACC-{id}", "days": 14, "include_premium": True},
                    "output": {"success": True, "new_end_date": "2024-12-29", "premium_enabled": True},
                },
            },
        },
        "Pricing_Inquiry": {
            "user_messages": [
                "Can you walk me through your pricing tiers? We're a team of about 50 people.",
                "What's the difference between your Professional and Enterprise plans?",
                "Do you offer volume discounts for larger teams? We might be looking at 200+ seats.",
            ],
            "agent_responses": [
                "Let me pull up our pricing information for a team your size. We have three main tiers that scale with your needs.",
                "For a team of 50, I'd recommend the Professional plan at $29/user/month. This includes all core features, priority support, and API access. The Enterprise plan at $49/user/month adds SSO, advanced analytics, custom integrations, and a dedicated account manager.",
                "Absolutely, we offer volume discounts starting at 100 seats. For 200+ seats, you'd qualify for our Enterprise tier with a 25% volume discount. Let me generate a custom quote for you.",
            ],
            "tools": {
                "Get_Pricing_Tiers": {
                    "input": {"team_size": 50},
                    "output": {
                        "tiers": [
                            {"name": "Starter", "price_per_user": 9.99, "max_users": 10},
                            {"name": "Professional", "price_per_user": 29.00, "max_users": 100},
                            {"name": "Enterprise", "price_per_user": 49.00, "max_users": None},
                        ],
                        "recommended": "Professional",
                    },
                },
                "Generate_Quote": {
                    "input": {"tier": "Professional", "seats": 50, "billing": "annual"},
                    "output": {
                        "quote_id": "QT-{id}",
                        "monthly_total": 1450.00,
                        "annual_total": 15660.00,
                        "discount": "10% annual billing discount",
                        "valid_until": "2025-01-15",
                    },
                },
            },
        },
        "Demo_Request": {
            "user_messages": [
                "We'd like to schedule a product demo for our engineering team. There will be about 8 people.",
                "Can we see a demo focused on the API integration capabilities?",
                "Our CTO wants to see how the platform handles enterprise-scale deployments.",
            ],
            "agent_responses": [
                "I'd be glad to set up a demo for your team. Let me check available times and prepare a personalized environment.",
                "I've scheduled a demo focused on API integration for next Tuesday at 2 PM ET. I'll prepare a sandbox environment with sample data relevant to your industry.",
                "For your CTO's review, I've prepared an enterprise-scale demo showing multi-tenant architecture, SSO integration, and our 99.99% uptime dashboard. The demo is scheduled for Thursday at 10 AM ET.",
            ],
            "tools": {
                "Check_Calendar_Availability": {
                    "input": {"timezone": "America/New_York", "duration_minutes": 60},
                    "output": {
                        "available_slots": [
                            "2024-12-17T14:00:00-05:00",
                            "2024-12-18T10:00:00-05:00",
                            "2024-12-19T15:00:00-05:00",
                        ],
                    },
                },
                "Schedule_Demo": {
                    "input": {"slot": "2024-12-17T14:00:00-05:00", "attendees": 8, "focus": "api_integration"},
                    "output": {"demo_id": "DEMO-{id}", "calendar_link": "https://cal.example.com/demo-{id}", "sandbox_ready": True},
                },
            },
        },
        "Competitor_Comparison": {
            "user_messages": [
                "How does your platform compare to Datadog for LLM observability?",
                "We're evaluating your tool against Langsmith. What are your key differentiators?",
                "Why should we choose your platform over the open-source alternatives?",
            ],
            "agent_responses": [
                "Great question. Let me pull up a detailed comparison. Our platform differentiates in three key areas compared to Datadog's LLM offering.",
                "Compared to Langsmith, our key differentiators are: (1) bidirectional OTel GenAI support with 12 ingestion sources, (2) AI judges for automated evaluation, and (3) the replay system for A/B testing. Let me show you specific metrics.",
                "While open-source tools provide basic tracing, our platform adds enterprise-grade evaluation, multi-provider normalization, and audit-quality hash chains. The ROI typically comes from reduced debugging time and automated quality gates.",
            ],
            "tools": {
                "Get_Competitor_Analysis": {
                    "input": {"competitor": "datadog", "features": ["llm_observability", "evaluation", "replay"]},
                    "output": {
                        "comparison": {
                            "llm_observability": {"us": "12 sources, OTel native", "them": "Custom SDK, 3 sources"},
                            "evaluation": {"us": "5 AI judges, automated", "them": "Manual review"},
                            "replay": {"us": "Bidirectional, parameterized", "them": "Not available"},
                        },
                    },
                },
            },
        },
        "ROI_Analysis": {
            "user_messages": [
                "Can you help me build a business case for adopting your platform?",
                "What's the typical ROI our team can expect? We spend about $50K/month on LLM APIs.",
                "I need to justify the cost to my CFO. What metrics should I focus on?",
            ],
            "agent_responses": [
                "I'd be happy to help build your business case. Let me analyze your current spending and project the expected savings.",
                "Based on your $50K/month API spend, customers like you typically see 15-25% cost reduction through our optimization recommendations and caching insights. That's $7.5K-$12.5K/month in savings against our $2.5K/month platform cost.",
                "For your CFO presentation, I'd focus on three metrics: (1) API cost reduction through prompt optimization, (2) debugging time saved with root-cause analysis, and (3) quality improvement measured by our AI judges. I've prepared a custom ROI calculator for your team.",
            ],
            "tools": {
                "Calculate_ROI": {
                    "input": {"monthly_api_spend": 50000, "team_size": 15, "current_tools": ["manual_review"]},
                    "output": {
                        "monthly_savings": 10000,
                        "annual_savings": 120000,
                        "platform_cost": 30000,
                        "net_annual_roi": 90000,
                        "payback_months": 4,
                        "efficiency_gain_percent": 35,
                    },
                },
            },
        },
    },
}
