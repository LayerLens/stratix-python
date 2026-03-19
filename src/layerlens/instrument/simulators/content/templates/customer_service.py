"""Customer service scenario templates.

Topics: Shipping_Delay, Account_Access, Product_Issue, Billing_Dispute, Escalation
"""

CUSTOMER_SERVICE_TEMPLATES = {
    "scenario": "customer_service",
    "agent_names": ["Case_Resolution_Agent", "Customer_Support_Agent", "Escalation_Agent"],
    "system_prompts": {
        "Case_Resolution_Agent": (
            "You are a customer service agent specializing in case resolution. "
            "Use available tools to look up order details, account information, and "
            "previous interactions. Provide clear, empathetic responses and resolve "
            "issues efficiently. Escalate to a supervisor when the customer requests "
            "it or the issue exceeds your authorization level."
        ),
        "Customer_Support_Agent": (
            "You are a frontline customer support agent. Help customers with their "
            "inquiries, look up relevant information, and provide accurate solutions. "
            "Always verify the customer's identity before sharing account details."
        ),
        "Escalation_Agent": (
            "You are a senior escalation agent handling complex or sensitive cases. "
            "You have elevated permissions to issue refunds, apply credits, and "
            "override standard policies when warranted."
        ),
    },
    "topics": {
        "Shipping_Delay": {
            "user_messages": [
                "My order #{order_id} was supposed to arrive 3 days ago but the tracking still shows it in transit. Can you help?",
                "I've been waiting over a week for my package. The estimated delivery was last Monday. What's going on?",
                "Where is my shipment? Order #{order_id}. It's been stuck at the distribution center for 4 days.",
            ],
            "agent_responses": [
                "I understand your frustration with the shipping delay. Let me look up your order #{order_id} to see the current status and find out what happened.",
                "I've checked your order and I can see it was delayed at the regional distribution center due to severe weather. The updated estimated delivery is {delivery_date}. I'd like to offer you a $10 credit for the inconvenience.",
                "I apologize for the continued delay. I've escalated this to our shipping team and they'll prioritize getting your package out. You'll receive an updated tracking notification within 2 hours.",
            ],
            "tools": {
                "Get_Order_Details": {
                    "input": {"order_id": "ORD-2024-{id}", "include_tracking": True},
                    "output": {
                        "order_id": "ORD-2024-{id}",
                        "status": "in_transit",
                        "carrier": "FedEx",
                        "tracking_number": "794644790132",
                        "estimated_delivery": "2024-12-15",
                        "shipping_method": "standard",
                        "last_update": "Package at regional distribution center",
                    },
                },
                "Get_Customer_History": {
                    "input": {"customer_id": "CUST-{id}", "limit": 5},
                    "output": {
                        "customer_id": "CUST-{id}",
                        "total_orders": 12,
                        "lifetime_value": 1247.50,
                        "satisfaction_score": 4.2,
                        "recent_cases": [],
                    },
                },
            },
        },
        "Account_Access": {
            "user_messages": [
                "I can't log into my account. I've tried resetting my password three times but I'm not receiving the email.",
                "My account seems to be locked after too many failed login attempts. How do I get back in?",
                "I changed my email address recently and now I can't access my account at all.",
            ],
            "agent_responses": [
                "I'm sorry you're having trouble accessing your account. Let me verify your identity and check the account status right away.",
                "I've verified your identity and unlocked your account. I've also sent a password reset link to your email on file. Please check your spam folder if you don't see it within 5 minutes.",
                "Your account has been successfully updated with your new email address. I've sent a verification link to confirm the change. Once verified, you'll be able to log in normally.",
            ],
            "tools": {
                "Verify_Customer_Identity": {
                    "input": {"email": "user@example.com", "verification_method": "email_otp"},
                    "output": {"verified": True, "customer_id": "CUST-{id}", "account_status": "locked"},
                },
                "Unlock_Account": {
                    "input": {"customer_id": "CUST-{id}", "reason": "customer_request"},
                    "output": {"success": True, "new_status": "active", "reset_link_sent": True},
                },
            },
        },
        "Product_Issue": {
            "user_messages": [
                "The product I received is defective. The screen has a crack and it wasn't like that when I opened the box.",
                "I ordered the blue version but received red instead. I need the correct item.",
                "The item stopped working after just 2 weeks of normal use. This should be covered under warranty.",
            ],
            "agent_responses": [
                "I'm sorry to hear about the defective product. Let me look up your order and arrange a replacement right away.",
                "I can see the order details and confirm the wrong color was shipped. I'll initiate a return and send you the correct item with expedited shipping at no extra charge.",
                "Your product is indeed within the warranty period. I've created a warranty claim and you'll receive a prepaid return label via email. Once we receive the defective unit, we'll ship the replacement.",
            ],
            "tools": {
                "Get_Order_Details": {
                    "input": {"order_id": "ORD-2024-{id}", "include_items": True},
                    "output": {
                        "order_id": "ORD-2024-{id}",
                        "items": [{"sku": "PRD-001", "name": "Widget Pro", "quantity": 1, "price": 79.99}],
                        "delivered_date": "2024-12-10",
                    },
                },
                "Create_Return": {
                    "input": {"order_id": "ORD-2024-{id}", "reason": "defective", "refund_type": "replacement"},
                    "output": {"return_id": "RET-{id}", "label_sent": True, "replacement_order": "ORD-2024-{id}R"},
                },
            },
        },
        "Billing_Dispute": {
            "user_messages": [
                "I was charged twice for the same order. I need the duplicate charge refunded immediately.",
                "There's a charge on my account for $149.99 that I don't recognize. I didn't authorize this purchase.",
                "I cancelled my subscription last month but I was still charged this month. Please refund me.",
            ],
            "agent_responses": [
                "I can see the duplicate charge on your account. I've initiated a refund for the extra charge of ${amount}. It should appear on your statement within 3-5 business days.",
                "I understand your concern about the unrecognized charge. Let me investigate this. I can see it's associated with order #{order_id} placed on {date}. Does that ring a bell?",
                "I apologize for the billing error. Your subscription was cancelled on {date} but the system processed one more charge. I've refunded the full amount and confirmed your cancellation is now complete.",
            ],
            "tools": {
                "Get_Billing_History": {
                    "input": {"customer_id": "CUST-{id}", "months": 3},
                    "output": {
                        "transactions": [
                            {"date": "2024-12-01", "amount": 149.99, "type": "charge", "order_id": "ORD-001"},
                            {"date": "2024-12-01", "amount": 149.99, "type": "charge", "order_id": "ORD-001"},
                        ],
                    },
                },
                "Process_Refund": {
                    "input": {"transaction_id": "TXN-{id}", "amount": 149.99, "reason": "duplicate_charge"},
                    "output": {"refund_id": "REF-{id}", "status": "processing", "estimated_days": 5},
                },
            },
        },
        "Escalation": {
            "user_messages": [
                "I want to speak to a manager. This is the third time I've called about this issue and it's still not resolved.",
                "Your agent couldn't help me. I need someone with more authority to handle my complaint.",
                "I've been a loyal customer for 5 years and this experience is unacceptable. Please escalate my case.",
            ],
            "agent_responses": [
                "I completely understand your frustration, and I sincerely apologize for the repeated inconvenience. Let me pull up the full history of your case and escalate this to our senior resolution team immediately.",
                "I've reviewed your entire case history and I can see this has been ongoing. As a senior agent, I have the authority to resolve this. Here's what I'm going to do: {resolution}",
                "Thank you for your loyalty and patience. I've applied a {credit_amount} account credit, resolved the underlying issue, and set up a follow-up check in 48 hours to ensure everything is working correctly.",
            ],
            "tools": {
                "Get_Case_History": {
                    "input": {"customer_id": "CUST-{id}", "include_interactions": True},
                    "output": {
                        "cases": [
                            {"case_id": "CASE-001", "date": "2024-11-15", "status": "closed_unresolved"},
                            {"case_id": "CASE-002", "date": "2024-11-28", "status": "closed_unresolved"},
                        ],
                        "total_interactions": 7,
                    },
                },
                "Apply_Account_Credit": {
                    "input": {"customer_id": "CUST-{id}", "amount": 50.00, "reason": "escalation_resolution"},
                    "output": {"credit_id": "CRD-{id}", "new_balance": 50.00, "applied": True},
                },
            },
        },
    },
}
