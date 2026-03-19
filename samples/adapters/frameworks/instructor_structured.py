#!/usr/bin/env python3
"""
Instructor Structured Output Extraction with STRATIX Instrumentation

Demonstrates using Instructor to extract structured data from unstructured
text, with STRATIX tracing of each extraction call and validation step.

Requirements:
    pip install instructor openai pydantic

Set OPENAI_API_KEY in your environment before running.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

try:
    import instructor
    from pydantic import BaseModel, Field
except ImportError:
    sys.exit(
        "This sample requires instructor. Install with:\n"
        "  pip install instructor openai pydantic"
    )

import openai
from layerlens.instrument import STRATIX, emit_tool_call


# ---------------------------------------------------------------------------
# Pydantic models for structured extraction
# ---------------------------------------------------------------------------

class Address(BaseModel):
    street: Optional[str] = None
    city: str
    state: Optional[str] = None
    country: str
    zip_code: Optional[str] = None


class Person(BaseModel):
    name: str = Field(description="Full name of the person")
    age: Optional[int] = Field(default=None, description="Age if mentioned")
    occupation: Optional[str] = Field(default=None, description="Job title or role")
    email: Optional[str] = Field(default=None, description="Email address if present")
    address: Optional[Address] = Field(default=None, description="Address if mentioned")


class ExtractedEntities(BaseModel):
    """Collection of entities extracted from text."""
    people: list[Person] = Field(default_factory=list, description="People mentioned in the text")
    summary: str = Field(description="One-sentence summary of the text")
    sentiment: str = Field(description="Overall sentiment: positive, negative, or neutral")
    key_topics: list[str] = Field(default_factory=list, description="Main topics discussed")


SAMPLE_TEXTS = {
    "email": (
        "Hi team, I'm Sarah Chen (sarah.chen@example.com), the new VP of Engineering "
        "based in San Francisco, CA 94105. I'm 34 and excited to join! I'll be working "
        "closely with James Rivera, our 42-year-old CTO in New York. Looking forward "
        "to our AI observability product launch next quarter."
    ),
    "review": (
        "After using LayerLens for three months, our team lead Mike Thompson in London "
        "says it transformed how we debug agent workflows. The trace visualization is "
        "outstanding, though the documentation could use improvement. Overall very "
        "satisfied with the platform."
    ),
    "report": (
        "Q3 results show Dr. Emily Park, Chief Data Scientist at 29, drove a 40% "
        "improvement in model accuracy from our Austin, TX office. Her colleague "
        "Raj Patel, 37, Senior ML Engineer in Bangalore, India, contributed the "
        "new pipeline architecture. The team morale is exceptionally high."
    ),
}


def main():
    parser = argparse.ArgumentParser(description="Instructor structured extraction with STRATIX tracing")
    parser.add_argument("--text", choices=list(SAMPLE_TEXTS.keys()), default="email",
                        help="Sample text to extract from")
    parser.add_argument("--custom-text", default=None, help="Custom text to extract from (overrides --text)")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--agent-id", default="instructor-extraction-demo")
    parser.add_argument("--policy", default="stratix-demo@1.0.0")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY environment variable before running this sample.")

    text = args.custom_text or SAMPLE_TEXTS[args.text]

    # Initialize STRATIX
    stratix = STRATIX(
        policy_ref=args.policy,
        agent_id=args.agent_id,
        framework="openai",
    )
    ctx = stratix.start_trial()
    stratix.emit_input(f"Extract entities from: {text[:100]}...")

    # Patch OpenAI client with Instructor
    client = instructor.from_openai(openai.OpenAI())

    print(f"Input text:\n  {text}\n")

    # Extract structured data
    entities = client.chat.completions.create(
        model=args.model,
        response_model=ExtractedEntities,
        messages=[
            {"role": "system", "content": "Extract all entities and metadata from the given text."},
            {"role": "user", "content": text},
        ],
        max_retries=2,
    )

    # Display results
    print(f"Summary: {entities.summary}")
    print(f"Sentiment: {entities.sentiment}")
    print(f"Topics: {', '.join(entities.key_topics)}")
    print(f"\nPeople found ({len(entities.people)}):")
    for person in entities.people:
        parts = [person.name]
        if person.age:
            parts.append(f"age {person.age}")
        if person.occupation:
            parts.append(person.occupation)
        if person.address:
            parts.append(f"{person.address.city}, {person.address.country}")
        print(f"  - {', '.join(parts)}")

    stratix.emit_output(entities.model_dump_json(indent=2))

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
