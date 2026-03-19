#!/usr/bin/env python3
"""
DSPy Compiled Prompts with STRATIX Instrumentation (AWS Bedrock)

Demonstrates DSPy signatures and compiled modules using AWS Bedrock
as the LLM backend, with STRATIX tracing of optimization steps.

Requirements:
    pip install dspy-ai boto3

Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_DEFAULT_REGION in
your environment before running.
"""

from __future__ import annotations

import argparse
import os
import sys

try:
    import dspy
except ImportError:
    sys.exit(
        "This sample requires dspy. Install with:\n"
        "  pip install dspy-ai boto3"
    )

from layerlens.instrument import STRATIX


# ---------------------------------------------------------------------------
# DSPy signatures
# ---------------------------------------------------------------------------

class ClassifyIntent(dspy.Signature):
    """Classify the user's intent from their message."""
    message: str = dspy.InputField(desc="User message to classify")
    intent: str = dspy.OutputField(desc="Classified intent: question, command, feedback, or other")
    confidence: str = dspy.OutputField(desc="Confidence level: high, medium, or low")


class GenerateResponse(dspy.Signature):
    """Generate a helpful response given the classified intent."""
    message: str = dspy.InputField(desc="Original user message")
    intent: str = dspy.InputField(desc="Classified intent")
    response: str = dspy.OutputField(desc="Helpful response addressing the user's intent")


class IntentRouter(dspy.Module):
    """A two-step module: classify intent, then generate a response."""

    def __init__(self):
        super().__init__()
        self.classify = dspy.ChainOfThought(ClassifyIntent)
        self.respond = dspy.ChainOfThought(GenerateResponse)

    def forward(self, message: str) -> dspy.Prediction:
        classification = self.classify(message=message)
        response = self.respond(
            message=message,
            intent=classification.intent,
        )
        return dspy.Prediction(
            intent=classification.intent,
            confidence=classification.confidence,
            response=response.response,
        )


# ---------------------------------------------------------------------------
# Sample data for optimization
# ---------------------------------------------------------------------------

TRAINING_EXAMPLES = [
    dspy.Example(message="How do I reset my password?", intent="question", confidence="high").with_inputs("message"),
    dspy.Example(message="Delete all my old logs", intent="command", confidence="high").with_inputs("message"),
    dspy.Example(message="The dashboard is really slow lately", intent="feedback", confidence="medium").with_inputs("message"),
    dspy.Example(message="What frameworks does STRATIX support?", intent="question", confidence="high").with_inputs("message"),
    dspy.Example(message="Enable two-factor authentication for my account", intent="command", confidence="high").with_inputs("message"),
]

TEST_MESSAGES = [
    "Can you explain how hash-chain attestation works?",
    "Archive traces older than 30 days",
    "The new adapter auto-detection is amazing!",
    "What's the difference between cleartext and redacted privacy?",
]


def intent_metric(example, pred, trace=None) -> bool:
    """Simple metric: check if intent matches and confidence is not low."""
    return pred.confidence.lower() != "low"


def main():
    parser = argparse.ArgumentParser(description="DSPy with AWS Bedrock and STRATIX tracing")
    parser.add_argument("--model", default="anthropic.claude-3-haiku-20240307-v1:0")
    parser.add_argument("--region", default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    parser.add_argument("--agent-id", default="dspy-bedrock-demo")
    parser.add_argument("--policy", default="stratix-demo@1.0.0")
    parser.add_argument("--optimize", action="store_true", help="Run BootstrapFewShot optimization")
    args = parser.parse_args()

    for var in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        if not os.environ.get(var):
            sys.exit(f"Set {var} environment variable before running this sample.")

    # Initialize STRATIX
    stratix = STRATIX(
        policy_ref=args.policy,
        agent_id=args.agent_id,
        framework="aws_bedrock",
    )
    ctx = stratix.start_trial()

    # Configure DSPy with Bedrock
    lm = dspy.LM(
        model=f"bedrock/{args.model}",
        aws_region_name=args.region,
    )
    dspy.configure(lm=lm)

    router = IntentRouter()

    # Optionally optimize
    if args.optimize:
        print("Optimizing with BootstrapFewShot...\n")
        optimizer = dspy.BootstrapFewShot(metric=intent_metric, max_bootstrapped_demos=3)
        router = optimizer.compile(router, trainset=TRAINING_EXAMPLES)
        print("Optimization complete.\n")

    # Run on test messages
    for msg in TEST_MESSAGES:
        stratix.emit_input(msg)
        result = router(message=msg)
        print(f"Message: {msg}")
        print(f"  Intent: {result.intent} (confidence: {result.confidence})")
        print(f"  Response: {result.response[:100]}")
        print()
        stratix.emit_output(f"[{result.intent}] {result.response[:200]}")

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
