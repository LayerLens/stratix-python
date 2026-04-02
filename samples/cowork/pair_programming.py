#!/usr/bin/env python3
"""Co-Work: Pair Programming for Judge Refinement -- LayerLens Python SDK Sample.

Demonstrates an iterative pair-programming pattern where a Rubric Writer agent
creates and refines a custom judge, and a Rubric Tester agent validates it by
uploading sample traces and running evaluations. The two agents collaborate in
a create-test-refine loop until the judge produces satisfactory results.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python pair_programming.py
"""

from __future__ import annotations

import os
import sys
from typing import Any

from layerlens import Stratix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _helpers import create_judge, upload_trace_dict, poll_evaluation_results

# ---------------------------------------------------------------------------
# Test cases: pairs of prompts and responses with expected quality
# ---------------------------------------------------------------------------

TEST_CASES: list[dict[str, Any]] = [
    {
        "label": "Good: concise and helpful",
        "input": "How do I reverse a list in Python?",
        "output": (
            "You can reverse a list in Python using `my_list[::-1]` for a new "
            "reversed list, or `my_list.reverse()` to reverse in place. Both "
            "approaches are O(n)."
        ),
        "expected_quality": "high",
    },
    {
        "label": "Poor: vague and unhelpful",
        "input": "How do I reverse a list in Python?",
        "output": "You can do it with a function. Look it up.",
        "expected_quality": "low",
    },
    {
        "label": "Good: thorough explanation",
        "input": "Explain the difference between a list and a tuple in Python.",
        "output": (
            "Lists are mutable sequences created with [], while tuples are "
            "immutable sequences created with (). Tuples are hashable (can be "
            "dict keys), use less memory, and signal that the data should not "
            "change. Lists are better when you need to add or remove items."
        ),
        "expected_quality": "high",
    },
    {
        "label": "Poor: incorrect information",
        "input": "Explain the difference between a list and a tuple in Python.",
        "output": ("Lists and tuples are the same thing in Python. They both use square brackets and are mutable."),
        "expected_quality": "low",
    },
]

QUALITY_THRESHOLD = 0.6
MAX_REFINEMENT_ROUNDS = 2


def run_test_suite(
    client: Stratix,
    judge_id: str,
    round_num: int,
) -> list[dict[str, Any]]:
    """Upload test traces and evaluate them with the given judge.

    Returns a list of result dicts with scores and expected quality.
    """
    results: list[dict[str, Any]] = []

    for case in TEST_CASES:
        trace_result = upload_trace_dict(
            client,
            input_text=case["input"],
            output_text=case["output"],
            metadata={
                "test_label": case["label"],
                "round": round_num,
                "channel": "co-work-pair-programming",
            },
        )
        tid = trace_result.trace_ids[0] if trace_result.trace_ids else "unknown"

        evaluation = client.trace_evaluations.create(
            trace_id=tid,
            judge_id=judge_id,
        )

        # Fetch detailed results
        eval_results = poll_evaluation_results(client, evaluation.id)
        score = 0.0
        if eval_results:
            score = eval_results[0].score

        results.append(
            {
                "label": case["label"],
                "trace_id": tid,
                "expected_quality": case["expected_quality"],
                "score": score,
                "detail": eval_results,
            }
        )

    return results


def check_alignment(results: list[dict[str, Any]]) -> tuple[bool, float]:
    """Check whether judge scores align with expected quality labels.

    Returns (is_aligned, alignment_rate).
    """
    aligned = 0
    for r in results:
        score_is_high = r["score"] >= QUALITY_THRESHOLD
        expected_high = r["expected_quality"] == "high"
        if score_is_high == expected_high:
            aligned += 1

    rate = aligned / len(results) if results else 0.0
    return rate >= 0.75, rate


def main() -> None:
    """Run the pair-programming judge refinement Co-Work Channel demo."""
    print("=== LayerLens Co-Work: Pair Programming (Judge Refinement) ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Phase 1 -- Rubric Writer: create an initial custom judge
    # ------------------------------------------------------------------
    initial_goal = (
        "Evaluate whether an AI assistant response is helpful, accurate, "
        "and sufficiently detailed for the user's question."
    )
    print("[RubricWriter] Creating initial judge...")
    judge = create_judge(
        client,
        name="PairProg-ResponseQuality",
        evaluation_goal=initial_goal,
    )
    judge_id = judge.id
    print(f"[RubricWriter] Judge created: {judge_id}")
    print(f'[RubricWriter] Goal: "{initial_goal[:80]}..."\n')

    # ------------------------------------------------------------------
    # Iterative refinement loop
    # ------------------------------------------------------------------
    refined_goals = [
        (
            "Evaluate whether an AI assistant response is helpful, accurate, "
            "and detailed. Penalize vague or dismissive answers that do not "
            "address the question. Penalize factually incorrect statements. "
            "Reward responses that include concrete examples or code."
        ),
        (
            "Evaluate AI assistant response quality on three axes: "
            "(1) correctness -- all facts must be accurate, "
            "(2) helpfulness -- the answer must directly address the question "
            "with actionable detail, "
            "(3) completeness -- cover the key aspects without being verbose. "
            "Score below 0.5 if the answer is vague, dismissive, or incorrect."
        ),
    ]

    try:
        for round_num in range(1, MAX_REFINEMENT_ROUNDS + 2):
            print(f"--- Round {round_num} ---\n")

            # Rubric Tester: run test suite
            print(f"[RubricTester] Testing judge {judge_id} with {len(TEST_CASES)} cases...")
            results = run_test_suite(client, judge_id, round_num)

            for r in results:
                marker = "PASS" if ((r["score"] >= QUALITY_THRESHOLD) == (r["expected_quality"] == "high")) else "MISS"
                print(
                    f'[RubricTester]   {marker} "{r["label"]}" '
                    f"score={r['score']:.2f} (expected={r['expected_quality']})"
                )

            is_aligned, alignment_rate = check_alignment(results)
            print(f"\n[RubricTester] Alignment rate: {alignment_rate:.0%}")

            if is_aligned:
                print("[RubricTester] Judge meets quality threshold. Done.\n")
                break

            if round_num - 1 < len(refined_goals):
                new_goal = refined_goals[round_num - 1]
                print(f"\n[RubricWriter] Refining judge goal (round {round_num + 1})...")
                client.judges.update(judge_id, evaluation_goal=new_goal)
                print(f'[RubricWriter] Updated goal: "{new_goal[:80]}..."\n')
            else:
                print("[RubricWriter] Max refinement rounds reached.\n")
                break

        # ------------------------------------------------------------------
        # Final report
        # ------------------------------------------------------------------
        print("=" * 64)
        print("[PairProgramming] Session Summary")
        print("=" * 64)

        final_judge = client.judges.get(judge_id)
        print(f"  Judge ID: {judge_id}")
        print(f"  Judge name: {final_judge.name}")
        print(f"  Final alignment rate: {alignment_rate:.0%}")
        print(f"  Refinement rounds: {round_num}")
        print(f"  Test cases used: {len(TEST_CASES)}")
        print(f"  Threshold: {QUALITY_THRESHOLD}")
        print("  Judge and all evaluations stored in LayerLens.")

    finally:
        # Cleanup: delete the judge we created for the demo
        try:
            client.judges.delete(judge_id)
        except Exception:
            pass


if __name__ == "__main__":
    main()
