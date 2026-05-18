#!/usr/bin/env python3
"""Brand Evaluation -- LayerLens Python SDK Sample.

Evaluates content against brand guidelines for voice consistency
(tone, vocabulary, reading level) and visual identity compliance
using dedicated judges.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python brand_evaluation.py
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

from layerlens import Stratix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _helpers import create_judge, upload_trace_dict, poll_evaluation_results

# ---------------------------------------------------------------------------
# Brand guidelines and sample data
# ---------------------------------------------------------------------------

VOICE_GUIDELINES: dict[str, Any] = {
    "tone": "professional yet approachable",
    "reading_level": "8th grade",
    "avoid_words": ["synergy", "leverage", "disrupt"],
    "preferred_terms": {"customer": "client", "buy": "invest in"},
}

VISUAL_GUIDELINES: dict[str, Any] = {
    "primary_colors": ["#1a73e8", "#ffffff", "#333333"],
    "fonts": ["Inter", "Roboto"],
    "logo_min_size_px": 48,
}

SAMPLES: list[dict[str, Any]] = [
    {
        "id": "brand-001",
        "name": "Marketing email (on-brand)",
        "content": "We are excited to share our latest product updates. Our team has been working to make your experience even better.",
        "content_type": "text",
    },
    {
        "id": "brand-002",
        "name": "Landing page (mixed content)",
        "content": "Leverage our synergistic platform to disrupt the market!",
        "content_type": "mixed",
        "visual_metadata": {"colors_used": ["#1a73e8", "#ff0000"], "fonts_used": ["Arial"], "logo_size_px": 32},
    },
]

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

_PASS_COLOR = "\033[92m"
_FAIL_COLOR = "\033[91m"
_RESET = "\033[0m"


def display_result(label: str, score: float | None, passed: bool | None, reasoning: str) -> None:
    """Display a single evaluation result."""
    score_str = f"{score:.2f}" if score is not None else "N/A"
    if passed is not None:
        color = _PASS_COLOR if passed else _FAIL_COLOR
        status = "PASS" if passed else "FAIL"
    else:
        color = ""
        status = "PEND"
    detail = (reasoning[:60] + "...") if reasoning else ""
    print(f"  {label:10s} {color}{status:6s}{_RESET}  ({score_str}) - {detail}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run brand evaluation on all content samples."""
    print("=== LayerLens Brand Evaluation Sample ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Create judges up front
    print("Creating brand evaluation judges...")
    voice_judge = create_judge(
        client,
        name=f"Brand Voice Judge {int(time.time())}",
        evaluation_goal=(
            "Evaluate whether the content follows brand voice guidelines: "
            "professional yet approachable tone, 8th grade reading level, "
            "avoids banned words (synergy, leverage, disrupt), and uses "
            "preferred terminology."
        ),
    )
    print(f"  Created: {voice_judge.name} (id={voice_judge.id})")

    visual_judge = create_judge(
        client,
        name=f"Brand Visual Judge {int(time.time())}",
        evaluation_goal=(
            "Evaluate whether the content follows brand visual identity guidelines: "
            "uses approved colors (#1a73e8, #ffffff, #333333), approved fonts "
            "(Inter, Roboto), and meets minimum logo size requirements (48px)."
        ),
    )
    print(f"  Created: {visual_judge.name} (id={visual_judge.id})")
    print()

    print(f"Evaluating {len(SAMPLES)} content pieces against brand guidelines...\n")
    passed_all = 0

    try:
        for sample in SAMPLES:
            print(f"Sample: {sample['name']}")

            trace_result = upload_trace_dict(
                client,
                input_text="Brand compliance check",
                output_text=sample["content"],
                metadata={
                    "content_type": sample["content_type"],
                    "visual_metadata": sample.get("visual_metadata"),
                },
            )
            trace_id = trace_result.trace_ids[0] if trace_result.trace_ids else sample["id"]

            all_passed = True

            # Voice evaluation
            voice_eval = client.trace_evaluations.create(
                trace_id=trace_id,
                judge_id=voice_judge.id,
            )
            if voice_eval:
                results = poll_evaluation_results(client, voice_eval.id)
                if results:
                    r = results[0]
                    display_result("Voice", r.score, r.passed, r.reasoning or "")
                    if not r.passed:
                        all_passed = False
                else:
                    display_result("Voice", None, None, "(results pending)")
                    all_passed = False
            else:
                display_result("Voice", None, None, "Failed to create evaluation")
                all_passed = False

            # Visual evaluation (for mixed content)
            if sample["content_type"] == "mixed" and sample.get("visual_metadata"):
                visual_eval = client.trace_evaluations.create(
                    trace_id=trace_id,
                    judge_id=visual_judge.id,
                )
                if visual_eval:
                    results = poll_evaluation_results(client, visual_eval.id)
                    if results:
                        r = results[0]
                        display_result("Visual", r.score, r.passed, r.reasoning or "")
                        if not r.passed:
                            all_passed = False
                    else:
                        display_result("Visual", None, None, "(results pending)")
                        all_passed = False
                else:
                    display_result("Visual", None, None, "Failed to create evaluation")
                    all_passed = False
            else:
                print("  Visual:    N/A   - Text only content")

            if all_passed:
                passed_all += 1
            print()

        print(f"Overall: {passed_all}/{len(SAMPLES)} pieces fully compliant with brand guidelines")

    finally:
        # Clean up judges
        print("\nCleaning up judges...")
        for judge in [voice_judge, visual_judge]:
            try:
                client.judges.delete(judge.id)
                print(f"  Deleted: {judge.name}")
            except Exception:
                print(f"  WARNING: Could not delete judge {judge.id}")


if __name__ == "__main__":
    main()
