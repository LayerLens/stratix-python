#!/usr/bin/env python3
"""Retail: Customer Service -- LayerLens Python SDK Sample.

Evaluates customer support AI for response accuracy, tone/empathy,
and resolution effectiveness.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python retail_support.py
"""

from __future__ import annotations

import os
import sys
from typing import Any

from layerlens import Stratix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _helpers import upload_trace_dict, poll_evaluation_results, create_judge

SUPPORT_TICKETS: list[dict[str, Any]] = [
    {
        "id": "ticket-001",
        "category": "return_request",
        "customer_message": "I received the wrong item. I ordered a blue jacket size M but got a red one in size L.",
        "agent_response": "I'm sorry about the mix-up. I've initiated a prepaid return label. Once we receive the incorrect item, we'll ship the correct blue jacket in size M with express shipping at no cost. You should have it within 2-3 business days.",
        "policies_applied": ["30_day_return", "free_exchange_shipping", "wrong_item_priority"],
    },
    {
        "id": "ticket-002",
        "category": "price_match",
        "customer_message": "I bought this TV last week for $999 and now it's on sale for $799. Can I get the difference back?",
        "agent_response": "I'd be happy to check our price match policy. Our policy covers price adjustments within 14 days of purchase. Since you bought the TV last week, you're within the window. I've processed a $200 refund to your original payment method.",
        "policies_applied": ["14_day_price_match", "refund_to_original_payment"],
    },
]

_VERDICT_COLORS = {"pass": "\033[92m", "fail": "\033[91m", "uncertain": "\033[93m"}
_RESET = "\033[0m"


def main() -> None:
    """Run customer service evaluation."""
    print("=== LayerLens Retail: Customer Service Quality ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Create judges up front
    judges = {
        "accuracy": create_judge(
            client,
            name="Response Accuracy Judge",
            evaluation_goal="Evaluate whether the customer service response accurately applies company policies and provides correct information.",
        ),
        "empathy": create_judge(
            client,
            name="Empathy Judge",
            evaluation_goal="Evaluate whether the customer service response demonstrates appropriate empathy, tone, and professionalism.",
        ),
        "resolution": create_judge(
            client,
            name="Resolution Judge",
            evaluation_goal="Evaluate whether the customer service response effectively resolves the customer's issue with a clear action plan.",
        ),
    }
    judge_labels = {"accuracy": "Accuracy", "empathy": "Empathy", "resolution": "Resolution"}
    judge_ids = [j.id for j in judges.values()]

    try:
        for ticket in SUPPORT_TICKETS:
            trace_result = upload_trace_dict(client,
                input_text=ticket["customer_message"],
                output_text=ticket["agent_response"],
                metadata={"category": ticket["category"], "policies_applied": ticket["policies_applied"]},
            )
            trace_id = trace_result.trace_ids[0] if trace_result.trace_ids else ticket["id"]

            print(f"Ticket: {ticket['category']} - {ticket['customer_message'][:50]}...")
            for judge_key, judge_obj in judges.items():
                label = judge_labels[judge_key]
                evaluation = client.trace_evaluations.create(trace_id=trace_id, judge_id=judge_obj.id)
                results = poll_evaluation_results(client, evaluation.id)
                score = 0.0
                passed = False
                reasoning = ""
                if results:
                    r = results[0]
                    score = r.score
                    passed = r.passed
                    reasoning = r.reasoning
                verdict = "pass" if passed else "fail"
                color = _VERDICT_COLORS.get(verdict, "")
                print(f"  {label:12s} {color}{verdict.upper()}{_RESET} ({score:.2f}) - {reasoning}")
            print()

    finally:
        for jid in judge_ids:
            try:
                client.judges.delete(jid)
            except Exception:
                pass


if __name__ == "__main__":
    main()
