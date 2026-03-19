"""Order management scenario templates.

Topics: Order_Tracking, Payment_Problem, Return_Request, Cancellation, Split_Shipment
"""

ORDER_MANAGEMENT_TEMPLATES = {
    "scenario": "order_management",
    "agent_names": ["Order_Management_Agent", "Payment_Processing_Agent", "Returns_Agent"],
    "system_prompts": {
        "Order_Management_Agent": (
            "You are an order management agent. Help customers track orders, "
            "modify shipments, and resolve delivery issues. Use order lookup tools "
            "to provide accurate, real-time information."
        ),
        "Payment_Processing_Agent": (
            "You are a payment processing agent. Handle payment failures, "
            "retry declined transactions, update payment methods, and process refunds."
        ),
        "Returns_Agent": (
            "You are a returns specialist agent. Process return requests, "
            "generate return labels, and coordinate exchanges or refunds."
        ),
    },
    "topics": {
        "Order_Tracking": {
            "user_messages": [
                "Can you tell me where my order #{order_id} is right now?",
                "I need an update on my shipment. It was supposed to arrive yesterday.",
                "The tracking link you sent me isn't working. Can you check the status?",
            ],
            "agent_responses": [
                "Let me look up the current status of your order right away.",
                "Your order #{order_id} is currently at the local delivery facility and is out for delivery today. You should receive it by end of day.",
                "I apologize for the tracking link issue. I can see your package was delivered to your front porch at 2:47 PM today. I've resent the tracking confirmation to your email.",
            ],
            "tools": {
                "Track_Order": {
                    "input": {"order_id": "ORD-{id}"},
                    "output": {
                        "order_id": "ORD-{id}",
                        "status": "out_for_delivery",
                        "carrier": "UPS",
                        "tracking_number": "1Z999AA10123456784",
                        "estimated_delivery": "2024-12-15",
                        "events": [
                            {"timestamp": "2024-12-15T06:00:00Z", "event": "Out for delivery"},
                            {"timestamp": "2024-12-14T23:00:00Z", "event": "At local facility"},
                        ],
                    },
                },
            },
        },
        "Payment_Problem": {
            "user_messages": [
                "My payment was declined but I have sufficient funds. Can you help?",
                "I keep getting an error when trying to complete checkout with my credit card.",
                "The charge went through twice on my card. I need one of them reversed.",
            ],
            "agent_responses": [
                "I understand how frustrating that must be. Let me check the payment status and see what happened.",
                "I can see the decline was due to your bank's fraud protection. I recommend trying again after contacting your bank, or you can use a different payment method. I've saved your cart so nothing will be lost.",
                "I've confirmed the duplicate charge and initiated a reversal for the second transaction. The refund will appear on your statement within 3-5 business days.",
            ],
            "tools": {
                "Check_Payment_Status": {
                    "input": {"order_id": "ORD-{id}", "payment_method": "credit_card"},
                    "output": {
                        "status": "declined",
                        "decline_code": "fraud_suspected",
                        "amount": 89.99,
                        "last_four": "4242",
                        "attempts": 2,
                    },
                },
                "Process_Refund": {
                    "input": {"transaction_id": "TXN-{id}", "amount": 89.99, "reason": "duplicate_charge"},
                    "output": {"refund_id": "REF-{id}", "status": "processing", "estimated_days": 5},
                },
            },
        },
        "Return_Request": {
            "user_messages": [
                "I'd like to return the jacket I ordered. It doesn't fit right.",
                "How do I start a return? The product isn't what I expected from the description.",
                "I want to exchange this for a different size. Is that possible?",
            ],
            "agent_responses": [
                "I'd be happy to help with your return. Let me check your order and our return policy.",
                "Your order is within our 30-day return window. I've created a return label that will be sent to your email. Once we receive the item, your refund will be processed within 5 business days.",
                "Absolutely, I can arrange an exchange. I'll send you a return label for the current item and place a new order for the correct size with expedited shipping at no extra cost.",
            ],
            "tools": {
                "Check_Return_Eligibility": {
                    "input": {"order_id": "ORD-{id}", "item_sku": "PRD-001"},
                    "output": {
                        "eligible": True,
                        "return_window_remaining_days": 22,
                        "refund_amount": 79.99,
                        "exchange_available": True,
                    },
                },
                "Create_Return_Label": {
                    "input": {"order_id": "ORD-{id}", "return_type": "exchange", "new_size": "M"},
                    "output": {
                        "return_id": "RET-{id}",
                        "label_url": "https://returns.example.com/label/RET-{id}",
                        "exchange_order": "ORD-{id}X",
                    },
                },
            },
        },
        "Cancellation": {
            "user_messages": [
                "I need to cancel my order #{order_id}. I ordered the wrong item.",
                "Can I still cancel? I just placed the order 20 minutes ago.",
                "Please cancel my subscription and all pending orders.",
            ],
            "agent_responses": [
                "Let me check if your order is still eligible for cancellation.",
                "Your order hasn't shipped yet, so I was able to cancel it successfully. The refund of ${amount} will be processed within 2-3 business days.",
                "I've cancelled your subscription effective immediately. Any pending orders have also been cancelled. You'll retain access to your account features until the end of your current billing period on {date}.",
            ],
            "tools": {
                "Cancel_Order": {
                    "input": {"order_id": "ORD-{id}", "reason": "wrong_item"},
                    "output": {
                        "cancelled": True,
                        "refund_amount": 149.99,
                        "refund_method": "original_payment",
                        "estimated_refund_days": 3,
                    },
                },
            },
        },
        "Split_Shipment": {
            "user_messages": [
                "I received part of my order but some items are missing.",
                "Why was my order split into two shipments? I paid for one order.",
                "One package arrived but the tracking shows two shipments. When is the rest coming?",
            ],
            "agent_responses": [
                "Let me check the details of your order. Sometimes orders are split when items ship from different warehouses.",
                "Your order was split into two shipments because the items were in different fulfillment centers. The first package has arrived, and the second is scheduled for delivery on {date}.",
                "I can confirm both shipments. The second package containing your remaining items is currently in transit and will arrive by {date}. No additional shipping charges were applied for the split shipment.",
            ],
            "tools": {
                "Get_Shipment_Details": {
                    "input": {"order_id": "ORD-{id}"},
                    "output": {
                        "shipments": [
                            {"shipment_id": "SHIP-001", "status": "delivered", "items": ["Widget A", "Widget B"]},
                            {"shipment_id": "SHIP-002", "status": "in_transit", "items": ["Widget C"], "eta": "2024-12-17"},
                        ],
                    },
                },
            },
        },
    },
}
