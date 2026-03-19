#!/usr/bin/env python3
"""
Marvin Text Classification with STRATIX Instrumentation

Demonstrates Marvin's AI-powered classification, extraction, and
transformation functions, with STRATIX tracing of each operation.

Requirements:
    pip install marvin

Set OPENAI_API_KEY in your environment before running.
"""

from __future__ import annotations

import argparse
import os
import sys
from enum import Enum

try:
    import marvin
except ImportError:
    sys.exit(
        "This sample requires marvin. Install with:\n"
        "  pip install marvin"
    )

from pydantic import BaseModel, Field
from layerlens.instrument import STRATIX


# ---------------------------------------------------------------------------
# Classification labels and models
# ---------------------------------------------------------------------------

class TicketPriority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TicketCategory(str, Enum):
    BUG = "bug"
    FEATURE_REQUEST = "feature_request"
    DOCUMENTATION = "documentation"
    PERFORMANCE = "performance"
    SECURITY = "security"
    GENERAL = "general"


class TicketMetadata(BaseModel):
    """Structured metadata extracted from a support ticket."""
    affected_component: str = Field(description="Component or service affected")
    user_sentiment: str = Field(description="User sentiment: frustrated, neutral, or satisfied")
    requires_escalation: bool = Field(description="Whether this needs immediate escalation")
    suggested_assignee: str = Field(description="Team or role best suited to handle this")


# ---------------------------------------------------------------------------
# Sample tickets
# ---------------------------------------------------------------------------

SAMPLE_TICKETS = [
    "The dashboard keeps crashing when I try to view traces with more than 1000 events. "
    "This is blocking our production debugging. We need this fixed ASAP!",

    "It would be great if STRATIX could export traces to Jaeger in addition to OTLP. "
    "We use Jaeger extensively and this would simplify our setup.",

    "The API documentation for the emit_handoff function is missing the parameters "
    "description. Also, the example code has a typo on line 3.",

    "Our agent response times increased by 200ms after enabling STRATIX instrumentation. "
    "Is there a way to reduce the overhead? We're running in a latency-sensitive environment.",

    "We noticed that trace data is being sent over HTTP instead of HTTPS. This is a "
    "compliance issue for our SOC 2 audit. Please advise on secure configuration.",
]


def main():
    parser = argparse.ArgumentParser(description="Marvin text classification with STRATIX tracing")
    parser.add_argument("--agent-id", default="marvin-classification-demo")
    parser.add_argument("--policy", default="stratix-demo@1.0.0")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY environment variable before running this sample.")

    # Initialize STRATIX
    stratix = STRATIX(
        policy_ref=args.policy,
        agent_id=args.agent_id,
        framework="openai",
    )
    ctx = stratix.start_trial()
    stratix.emit_input(f"Classify {len(SAMPLE_TICKETS)} support tickets")

    print(f"Processing {len(SAMPLE_TICKETS)} support tickets...\n")

    results = []
    for i, ticket in enumerate(SAMPLE_TICKETS, 1):
        print(f"--- Ticket {i} ---")
        print(f"  Text: {ticket[:80]}...")

        # Classify priority
        priority = marvin.classify(ticket, labels=TicketPriority)
        print(f"  Priority: {priority.value}")

        # Classify category
        category = marvin.classify(ticket, labels=TicketCategory)
        print(f"  Category: {category.value}")

        # Extract metadata
        metadata = marvin.cast(ticket, target=TicketMetadata)
        print(f"  Component: {metadata.affected_component}")
        print(f"  Sentiment: {metadata.user_sentiment}")
        print(f"  Escalate: {metadata.requires_escalation}")
        print(f"  Assign to: {metadata.suggested_assignee}")
        print()

        results.append({
            "ticket": i,
            "priority": priority.value,
            "category": category.value,
            "escalate": metadata.requires_escalation,
        })

    # Print summary table
    print("=== Classification Summary ===")
    print(f"{'Ticket':<8} {'Priority':<10} {'Category':<18} {'Escalate':<10}")
    print("-" * 46)
    for r in results:
        print(f"{r['ticket']:<8} {r['priority']:<10} {r['category']:<18} {str(r['escalate']):<10}")

    stratix.emit_output(f"Classified {len(results)} tickets")

    # Summary
    summary = stratix.end_trial()
    events = stratix.get_events()
    print(f"\n--- STRATIX Trace Summary ---")
    print(f"Status: {summary.get('status')}")
    print(f"Captured {len(events)} events:")
    for e in events:
        print(f"  {e.get_event_type()}: {str(e.payload)[:80]}")


if __name__ == "__main__":
    main()
