#!/usr/bin/env python
"""
Compound Failure Calculator -- LayerLens Python SDK Sample
==========================================================

Proves a critical insight for AI agent reliability: an agent that
passes 85% of individual steps drops to roughly 20% end-to-end
accuracy over 10 steps. The compound failure effect is the single
biggest reason multi-step agents fail silently in production.

This tool takes multi-step agent traces, evaluates each step
independently with Stratix judges, then computes and visualizes
the compound failure probability using real evaluation data.

The math is simple: P(all_pass) = p^n, where p is per-step
accuracy and n is the number of steps. The consequences are not.

Prerequisites
-------------
* ``pip install layerlens --index-url https://sdk.layerlens.ai/package``
* Set ``LAYERLENS_STRATIX_API_KEY`` environment variable
* (Optional) ``pip install matplotlib`` for PNG chart export

Usage
-----
::

    export LAYERLENS_STRATIX_API_KEY=your-api-key

    # Simulate a 7-step agent and show compound failure curve
    python compound_failure_calculator.py --simulate 7

    # Custom per-step accuracy
    python compound_failure_calculator.py --simulate 10 --per-step-accuracy 0.90

    # Evaluate a real multi-step trace file
    python compound_failure_calculator.py --trace-file ../data/traces/multi_step_agent_workflow.json

    # Show compound failure across a custom range of step counts
    python compound_failure_calculator.py --simulate 7 --steps-range 1 25

    # Export as JSON
    python compound_failure_calculator.py --simulate 7 --json

    # Save visualization as PNG (requires matplotlib)
    python compound_failure_calculator.py --simulate 7 --output reliability_curve.png
"""

from __future__ import annotations

import os
import sys
import json
import math
import time
import logging
import argparse
from typing import Any, Dict, List, Optional, Tuple

from layerlens import Stratix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _helpers import create_judge, poll_evaluation_results, upload_trace_dict

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("layerlens.samples.compound_failure_calculator")


# ---------------------------------------------------------------------------
# Pre-computed compound scenario table
# ---------------------------------------------------------------------------

COMPOUND_SCENARIOS: Dict[str, Dict[int, float]] = {}

for _acc in (0.99, 0.95, 0.90, 0.85, 0.80, 0.75):
    _key = f"{_acc:.2f}"
    COMPOUND_SCENARIOS[_key] = {n: round(_acc**n, 6) for n in range(1, 21)}


# ---------------------------------------------------------------------------
# Sample agent steps (embedded for --simulate mode)
# ---------------------------------------------------------------------------

SAMPLE_AGENT_STEPS: List[Dict[str, str]] = [
    {
        "step_name": "parse_customer_intent",
        "input": (
            "Hi, I ordered a wireless keyboard (order #WK-20260508-4471) last "
            "Tuesday and it arrived yesterday with a stuck spacebar. I already "
            "tried the troubleshooting steps on your website and nothing fixed "
            "it. I want a full refund including the expedited shipping I paid "
            "for."
        ),
        "output": (
            "Intent parsed. Customer requests a full refund for order "
            "#WK-20260508-4471 (wireless keyboard). Reason: hardware defect "
            "(stuck spacebar). Customer has already attempted self-service "
            "troubleshooting. Refund scope requested: product cost plus "
            "expedited shipping fee. Sentiment: frustrated but cooperative."
        ),
        "evaluation_goal": (
            "Evaluate whether the agent correctly identified the customer "
            "intent, extracted all relevant order details, and classified "
            "the sentiment accurately."
        ),
    },
    {
        "step_name": "order_lookup",
        "input": (
            "Look up order #WK-20260508-4471 in the database and retrieve "
            "all order details including product, pricing, shipping, tax, "
            "delivery date, and return eligibility window."
        ),
        "output": (
            "Order retrieved. Product: TechFlow Pro Wireless Keyboard "
            "(SKU: TF-KB-PRO-BLK). Ordered: 2026-05-08. Delivered: "
            "2026-05-11. Product price: $89.99. Expedited shipping: $12.99. "
            "Tax (CA 9.5%): $9.78. Order total: $112.76. Return window: "
            "30 days from delivery (expires 2026-06-10). No prior returns "
            "on this account."
        ),
        "evaluation_goal": (
            "Evaluate whether the agent retrieved complete and accurate "
            "order information including all pricing components, dates, "
            "and return eligibility status."
        ),
    },
    {
        "step_name": "check_refund_eligibility",
        "input": (
            "Check whether order #WK-20260508-4471 qualifies for a refund "
            "under current return policies. The product has a hardware "
            "defect (stuck spacebar) and the customer attempted "
            "troubleshooting."
        ),
        "output": (
            "Eligibility verified against refund policy RP-2024-003. "
            "Return window: ELIGIBLE (within 30 days). Defect category: "
            "ELIGIBLE (hardware manufacturing defect, Section 4.2). "
            "Troubleshooting requirement: SATISFIED. Shipping refund: "
            "ELIGIBLE (defective returns, Section 6.1). Restocking fee: "
            "WAIVED (defective items exempt). All six checks passed."
        ),
        "evaluation_goal": (
            "Evaluate whether the agent correctly applied the refund "
            "policy rules, checked all required eligibility criteria, "
            "and reached the correct eligibility determination."
        ),
    },
    {
        "step_name": "calculate_refund_amount",
        "input": (
            "Calculate the exact refund amount for order #WK-20260508-4471. "
            "Product: $89.99. Shipping: $12.99. Original tax: $9.78. "
            "Apply California tax rules for the refund calculation."
        ),
        "output": (
            "Refund calculated. Product $89.99 plus shipping $12.99, "
            "subtotal $102.98. California sales tax at 9.5% on product "
            "only (shipping is tax-exempt in CA per Revenue and Taxation "
            "Code Section 6012): $8.55. Original tax of $9.78 included "
            "tax on shipping in error. Total refund: $111.53."
        ),
        "evaluation_goal": (
            "Evaluate the mathematical accuracy of the refund calculation, "
            "including correct tax treatment under California tax law and "
            "proper handling of the tax overcharge on the original order."
        ),
    },
    {
        "step_name": "draft_customer_response",
        "input": (
            "Draft a customer-facing response for ticket TKT-2026-0512-3347 "
            "communicating the approved refund of $111.53, return "
            "instructions, and timeline. Tone: empathetic and professional."
        ),
        "output": (
            "Hi there, thank you for reaching out about your TechFlow Pro "
            "Wireless Keyboard. I am sorry the spacebar is giving you "
            "trouble. Your order qualifies for a full refund of $111.53 "
            "(product $89.99 + shipping $12.99 + tax $8.55). The refund "
            "will go to your Visa ending in 3892 within 5 to 7 business "
            "days. I am generating a prepaid UPS return label now. Just "
            "drop the keyboard at any UPS location. Would you prefer a "
            "replacement instead?"
        ),
        "evaluation_goal": (
            "Evaluate the customer response for accuracy, empathetic tone, "
            "completeness of information (amount, timeline, return "
            "instructions), and appropriate offering of alternatives."
        ),
    },
    {
        "step_name": "verify_tone_and_compliance",
        "input": (
            "Review the drafted customer response for PII safety, refund "
            "amount accuracy, timeline accuracy, tone appropriateness, "
            "legal compliance, and absence of inappropriate content."
        ),
        "output": (
            "Compliance review complete. PII handling: PASS (only last "
            "four card digits shown). Amount accuracy: PASS ($111.53 "
            "matches calculation). Timeline: PASS (5-7 business days "
            "within Visa policy). Return label: PASS (generation "
            "triggered). Tone: empathetic and professional, no defensive "
            "language. Legal: PASS. Competitor mentions: PASS (none). "
            "Upsell pressure: PASS (replacement offer is optional). "
            "All 8 checks passed."
        ),
        "evaluation_goal": (
            "Evaluate whether the compliance review was thorough, checked "
            "all required categories, and correctly identified any issues "
            "or confirmed compliance across all dimensions."
        ),
    },
    {
        "step_name": "submit_refund",
        "input": (
            "Submit refund of $111.53 to payment gateway for Visa ending "
            "3892 on order #WK-20260508-4471. Generate prepaid UPS return "
            "label and update ticket status."
        ),
        "output": (
            "Refund submitted. Refund ID: RF-2026-0512-8834. Amount: "
            "$111.53 to Visa ending 3892. Gateway: APPROVED (auth code "
            "VR-449821). Processing: 5-7 business days. Return label: "
            "UPS 1Z999AA10123456784, emailed to customer. Ticket "
            "TKT-2026-0512-3347 updated to RESOLVED."
        ),
        "evaluation_goal": (
            "Evaluate whether the refund submission was executed correctly, "
            "all confirmation details were captured, the return label was "
            "generated, and the ticket was properly closed."
        ),
    },
]


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_compound_reliability(
    per_step_accuracy: float,
    max_steps: int,
) -> List[Dict[str, Any]]:
    """Compute compound reliability for each step count from 1 to max_steps.

    Args:
        per_step_accuracy: Probability of any single step passing (0.0 to 1.0).
        max_steps: Maximum number of steps to compute.

    Returns:
        List of dicts with keys: steps, compound_reliability, failure_probability.
    """
    results = []
    for n in range(1, max_steps + 1):
        compound = per_step_accuracy**n
        results.append(
            {
                "steps": n,
                "compound_reliability": round(compound, 6),
                "failure_probability": round(1 - compound, 6),
            }
        )
    return results


def find_reliability_cliff(
    per_step_accuracy: float,
    threshold: float = 0.50,
) -> int:
    """Find the step count where compound reliability drops below a threshold.

    Args:
        per_step_accuracy: Per-step pass rate.
        threshold: Reliability threshold (default 0.50).

    Returns:
        Step count where reliability first drops below the threshold.
    """
    if per_step_accuracy <= 0 or per_step_accuracy >= 1:
        return 1
    return math.ceil(math.log(threshold) / math.log(per_step_accuracy))


def expected_steps_before_failure(per_step_accuracy: float) -> float:
    """Compute expected number of steps before the first failure.

    Formula: E[steps] = 1 / (1 - p) where p is per-step accuracy.

    Args:
        per_step_accuracy: Per-step pass rate.

    Returns:
        Expected step count before failure.
    """
    q = 1.0 - per_step_accuracy
    if q <= 0:
        return float("inf")
    return round(1.0 / q, 2)


# ---------------------------------------------------------------------------
# Trace parsing
# ---------------------------------------------------------------------------


def parse_trace_steps(trace_path: str) -> List[Dict[str, str]]:
    """Extract evaluable steps from a multi-step agent trace JSON file.

    Looks for events with type "agent.output" that have a step_name field,
    paired with the preceding agent.input or model.invoke context.

    Args:
        trace_path: Path to a trace JSON file.

    Returns:
        List of step dicts with keys: step_name, input, output, evaluation_goal.
    """
    with open(trace_path, "r") as f:
        trace = json.load(f)

    events = trace.get("events", [])
    steps: List[Dict[str, str]] = []
    pending_input = ""

    for event in events:
        etype = event.get("type", "")
        payload = event.get("payload", {})

        if etype == "agent.input":
            pending_input = payload.get("input", "")

        elif etype == "agent.output" and payload.get("step"):
            step_name = payload.get("step_name", f"step_{payload['step']}")
            output_text = payload.get("output", "")

            steps.append(
                {
                    "step_name": step_name,
                    "input": pending_input or f"Execute {step_name}",
                    "output": output_text,
                    "evaluation_goal": (
                        f"Evaluate the quality, accuracy, and completeness "
                        f"of the '{step_name}' step in a multi-step agent "
                        f"workflow. Assess whether the output correctly "
                        f"addresses the input requirements."
                    ),
                }
            )
            pending_input = ""

    if not steps:
        logger.warning(
            "No step-tagged agent.output events found. "
            "Falling back to all agent.output events."
        )
        for event in events:
            if event.get("type") == "agent.output":
                payload = event.get("payload", {})
                agent = payload.get("agent_name", "unknown")
                output_text = payload.get("output", "")
                steps.append(
                    {
                        "step_name": agent,
                        "input": f"Agent '{agent}' task execution",
                        "output": output_text,
                        "evaluation_goal": (
                            f"Evaluate the quality and accuracy of the "
                            f"output produced by agent '{agent}'."
                        ),
                    }
                )

    return steps


# ---------------------------------------------------------------------------
# Stratix evaluation of individual steps
# ---------------------------------------------------------------------------


def evaluate_steps_with_stratix(
    client: Stratix,
    steps: List[Dict[str, str]],
    skip_cleanup: bool = False,
) -> List[Dict[str, Any]]:
    """Upload each step as a trace, create a judge, evaluate, and collect results.

    Args:
        client: An initialized Stratix client.
        steps: List of step dicts from parse_trace_steps or SAMPLE_AGENT_STEPS.
        skip_cleanup: If True, keep created resources after evaluation.

    Returns:
        List of per-step result dicts with keys: step_name, step_number,
        score, passed, reasoning, trace_id, judge_id.
    """
    created_trace_ids: List[str] = []
    created_judge_ids: List[str] = []
    step_results: List[Dict[str, Any]] = []

    try:
        models = client.models.get(type="public")
        if not models:
            logger.error("No public models available for judge creation.")
            sys.exit(1)
        model_id = models[0].id

        for i, step in enumerate(steps, 1):
            step_name = step["step_name"]
            logger.info(
                "Step %d/%d: Evaluating '%s'",
                i,
                len(steps),
                step_name,
            )

            # Upload the step as a trace
            result = upload_trace_dict(
                client,
                input_text=step["input"],
                output_text=step["output"],
                metadata={
                    "source": "compound-failure-calculator",
                    "step_name": step_name,
                    "step_number": i,
                },
            )
            if not result or not result.trace_ids:
                logger.error("  Failed to upload trace for step '%s'", step_name)
                step_results.append(
                    {
                        "step_name": step_name,
                        "step_number": i,
                        "score": None,
                        "passed": None,
                        "reasoning": "Trace upload failed",
                        "trace_id": None,
                        "judge_id": None,
                    }
                )
                continue

            trace_id = result.trace_ids[0]
            created_trace_ids.append(trace_id)

            # Create a judge for this step type
            judge = create_judge(
                client,
                name=f"Compound Calc Step Judge {step_name} {int(time.time())}",
                evaluation_goal=step["evaluation_goal"],
                model_id=model_id,
            )
            if not judge:
                logger.error("  Failed to create judge for step '%s'", step_name)
                step_results.append(
                    {
                        "step_name": step_name,
                        "step_number": i,
                        "score": None,
                        "passed": None,
                        "reasoning": "Judge creation failed",
                        "trace_id": trace_id,
                        "judge_id": None,
                    }
                )
                continue

            created_judge_ids.append(judge.id)

            # Run the evaluation
            trace_eval = client.trace_evaluations.create(
                trace_id=trace_id,
                judge_id=judge.id,
            )
            if not trace_eval:
                logger.error("  Failed to create evaluation for step '%s'", step_name)
                step_results.append(
                    {
                        "step_name": step_name,
                        "step_number": i,
                        "score": None,
                        "passed": None,
                        "reasoning": "Evaluation creation failed",
                        "trace_id": trace_id,
                        "judge_id": judge.id,
                    }
                )
                continue

            # Poll for results
            eval_results = poll_evaluation_results(client, trace_eval.id)
            if eval_results and len(eval_results) > 0:
                r = eval_results[0]
                reasoning_text = (r.reasoning or "")[:200]
                step_results.append(
                    {
                        "step_name": step_name,
                        "step_number": i,
                        "score": r.score,
                        "passed": r.passed,
                        "reasoning": reasoning_text,
                        "trace_id": trace_id,
                        "judge_id": judge.id,
                    }
                )
                status = "PASS" if r.passed else "FAIL"
                logger.info(
                    "  Result: %s (score=%s) %s",
                    status,
                    r.score,
                    reasoning_text[:80],
                )
            else:
                logger.warning("  No results returned for step '%s'", step_name)
                step_results.append(
                    {
                        "step_name": step_name,
                        "step_number": i,
                        "score": None,
                        "passed": None,
                        "reasoning": "Evaluation timed out",
                        "trace_id": trace_id,
                        "judge_id": judge.id,
                    }
                )

    finally:
        if not skip_cleanup:
            logger.info(
                "Cleaning up %d traces and %d judges...",
                len(created_trace_ids),
                len(created_judge_ids),
            )
            for jid in created_judge_ids:
                try:
                    client.judges.delete(jid)
                except Exception:
                    pass
            for tid in created_trace_ids:
                try:
                    client.traces.delete(tid)
                except Exception:
                    pass

    return step_results


# ---------------------------------------------------------------------------
# ASCII visualization
# ---------------------------------------------------------------------------


def render_ascii_chart(
    per_step_accuracy: float,
    steps_range: Tuple[int, int],
    actual_results: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Render an ASCII reliability decay chart.

    Args:
        per_step_accuracy: Per-step pass rate for the theoretical curve.
        steps_range: (start, end) range of step counts to display.
        actual_results: Optional list of per-step evaluation results to
            overlay as data points on the curve.

    Returns:
        Multi-line string containing the ASCII chart.
    """
    start, end = steps_range
    chart_width = 60
    chart_height = 20
    lines: List[str] = []

    lines.append("")
    lines.append("  COMPOUND RELIABILITY DECAY")
    lines.append(
        f"  Per-step accuracy: {per_step_accuracy:.1%}" f"    Steps: {start} to {end}"
    )
    lines.append("")

    # Compute values for each step count
    values = []
    for n in range(start, end + 1):
        values.append((n, per_step_accuracy**n))

    # Build actual results lookup
    actual_map: Dict[int, bool] = {}
    if actual_results:
        for r in actual_results:
            if r.get("passed") is not None:
                actual_map[r["step_number"]] = r["passed"]

    # Y-axis: 0% to 100%
    for row in range(chart_height, -1, -1):
        y_val = row / chart_height
        if row % 5 == 0:
            label = f"{y_val:5.0%} |"
        else:
            label = "      |"

        bar = []
        for n, compound in values:
            col_y = compound
            # Normalize to row position
            if abs(col_y - y_val) < (0.5 / chart_height):
                # Check if we have actual data for this step
                if n in actual_map:
                    bar.append("@" if actual_map[n] else "X")
                else:
                    bar.append("*")
            elif row == 0:
                bar.append("_")
            else:
                # Mark threshold lines
                if abs(y_val - 0.50) < (0.5 / chart_height):
                    bar.append("-")
                elif abs(y_val - 0.20) < (0.5 / chart_height):
                    bar.append(".")
                else:
                    bar.append(" ")

        lines.append(label + "".join(f" {c} " for c in bar))

    # X-axis
    x_labels = "       "
    for n in range(start, end + 1):
        x_labels += f"{n:>3}"
    lines.append(x_labels)
    lines.append("       " + " " * (len(values) * 3))
    lines.append("       " + "Steps in agent workflow".center(len(values) * 3))

    # Legend
    lines.append("")
    lines.append("  Legend:")
    lines.append("    *  Theoretical compound reliability (p^n)")
    if actual_results:
        lines.append("    @  Actual step evaluation: PASS")
        lines.append("    X  Actual step evaluation: FAIL")
    lines.append("    -  50% reliability threshold")
    lines.append("    .  20% reliability threshold")

    # Key thresholds
    cliff_50 = find_reliability_cliff(per_step_accuracy, 0.50)
    cliff_20 = find_reliability_cliff(per_step_accuracy, 0.20)
    expected = expected_steps_before_failure(per_step_accuracy)

    lines.append("")
    lines.append("  Key thresholds:")
    lines.append(f"    Reliability drops below 50% at step {cliff_50}")
    lines.append(f"    Reliability drops below 20% at step {cliff_20}")
    lines.append(f"    Expected steps before first failure: {expected}")

    return "\n".join(lines)


def render_scenario_table() -> str:
    """Render a table of pre-computed compound scenarios.

    Returns:
        Multi-line string with the scenario comparison table.
    """
    lines: List[str] = []
    lines.append("")
    lines.append("  COMPOUND RELIABILITY TABLE")
    lines.append("  Compound success probability (p^n) at selected step counts")
    lines.append("")

    # Header
    step_counts = [1, 3, 5, 7, 10, 15, 20]
    header = "  Accuracy |"
    for n in step_counts:
        header += f" {n:>5} |"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for acc_key in sorted(COMPOUND_SCENARIOS.keys(), reverse=True):
        row = f"    {acc_key}  |"
        for n in step_counts:
            val = COMPOUND_SCENARIOS[acc_key][n]
            row += f" {val:5.1%} |"
        lines.append(row)

    lines.append("")
    lines.append("  Reading: at 85% per-step accuracy over 10 steps,")
    lines.append("  compound reliability is only 19.7%. Over 20 steps: 3.9%.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Matplotlib visualization (optional)
# ---------------------------------------------------------------------------


def save_matplotlib_chart(
    per_step_accuracy: float,
    steps_range: Tuple[int, int],
    output_path: str,
    actual_results: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Save a matplotlib PNG chart of the reliability decay curve.

    Args:
        per_step_accuracy: Per-step pass rate.
        steps_range: (start, end) range of step counts.
        output_path: File path for the saved PNG.
        actual_results: Optional per-step results to overlay.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        logger.error(
            "matplotlib is required for PNG output. "
            "Install with: pip install matplotlib"
        )
        sys.exit(1)

    start, end = steps_range
    steps = list(range(start, end + 1))
    compound = [per_step_accuracy**n for n in steps]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(
        steps,
        compound,
        "b-o",
        linewidth=2,
        markersize=6,
        label=f"Compound reliability (p={per_step_accuracy:.0%})",
    )

    # Overlay actual results if available
    if actual_results:
        pass_steps = [
            r["step_number"] for r in actual_results if r.get("passed") is True
        ]
        fail_steps = [
            r["step_number"] for r in actual_results if r.get("passed") is False
        ]
        if pass_steps:
            pass_y = [per_step_accuracy**n for n in pass_steps]
            ax.scatter(
                pass_steps,
                pass_y,
                c="green",
                s=120,
                zorder=5,
                label="Actual: PASS",
                marker="^",
            )
        if fail_steps:
            fail_y = [per_step_accuracy**n for n in fail_steps]
            ax.scatter(
                fail_steps,
                fail_y,
                c="red",
                s=120,
                zorder=5,
                label="Actual: FAIL",
                marker="v",
            )

    # Threshold lines
    ax.axhline(
        y=0.50, color="orange", linestyle="--", alpha=0.7, label="50% reliability"
    )
    ax.axhline(y=0.20, color="red", linestyle="--", alpha=0.7, label="20% reliability")

    # Annotations
    cliff_50 = find_reliability_cliff(per_step_accuracy, 0.50)
    cliff_20 = find_reliability_cliff(per_step_accuracy, 0.20)
    if start <= cliff_50 <= end:
        ax.annotate(
            f"50% cliff at step {cliff_50}",
            xy=(cliff_50, 0.50),
            xytext=(cliff_50 + 1, 0.60),
            arrowprops=dict(arrowstyle="->", color="orange"),
            fontsize=10,
            color="orange",
        )
    if start <= cliff_20 <= end:
        ax.annotate(
            f"20% cliff at step {cliff_20}",
            xy=(cliff_20, 0.20),
            xytext=(cliff_20 + 1, 0.30),
            arrowprops=dict(arrowstyle="->", color="red"),
            fontsize=10,
            color="red",
        )

    ax.set_xlabel("Number of Agent Steps", fontsize=12)
    ax.set_ylabel("Compound Success Probability", fontsize=12)
    ax.set_title(
        f"Compound Failure: {per_step_accuracy:.0%} Per-Step Accuracy "
        f"Over {start} to {end} Steps",
        fontsize=14,
    )
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(start - 0.5, end + 0.5)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Chart saved to: %s", output_path)


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------


def build_summary(
    per_step_accuracy: float,
    num_steps: int,
    step_results: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Build a summary report of compound failure analysis.

    Args:
        per_step_accuracy: Per-step accuracy used for computation.
        num_steps: Number of steps in the workflow.
        step_results: Optional list of actual evaluation results.

    Returns:
        Dict containing all summary statistics and per-step details.
    """
    compound = per_step_accuracy**num_steps
    cliff_50 = find_reliability_cliff(per_step_accuracy, 0.50)
    cliff_20 = find_reliability_cliff(per_step_accuracy, 0.20)
    expected = expected_steps_before_failure(per_step_accuracy)

    summary: Dict[str, Any] = {
        "per_step_accuracy": per_step_accuracy,
        "num_steps": num_steps,
        "compound_reliability": round(compound, 6),
        "compound_failure_rate": round(1 - compound, 6),
        "reliability_cliff_50pct": cliff_50,
        "reliability_cliff_20pct": cliff_20,
        "expected_steps_before_failure": expected,
        "reliability_curve": compute_compound_reliability(
            per_step_accuracy,
            max(num_steps, 15),
        ),
    }

    if step_results:
        scored = [r for r in step_results if r.get("score") is not None]
        passed = [r for r in step_results if r.get("passed") is True]
        failed = [r for r in step_results if r.get("passed") is False]

        actual_pass_rate = len(passed) / len(scored) if scored else 0.0
        actual_compound = actual_pass_rate**num_steps if scored else 0.0

        summary["actual_results"] = {
            "steps_evaluated": len(scored),
            "steps_passed": len(passed),
            "steps_failed": len(failed),
            "actual_per_step_pass_rate": round(actual_pass_rate, 4),
            "actual_compound_reliability": round(actual_compound, 6),
            "per_step_details": step_results,
        }

    return summary


def print_summary(summary: Dict[str, Any]) -> None:
    """Print a formatted summary to the terminal.

    Args:
        summary: Summary dict from build_summary.
    """
    p = summary["per_step_accuracy"]
    n = summary["num_steps"]
    compound = summary["compound_reliability"]

    print("\n" + "=" * 64)
    print("  COMPOUND FAILURE ANALYSIS")
    print("=" * 64)
    print()
    print(f"  Configuration:")
    print(f"    Per-step accuracy:  {p:.1%}")
    print(f"    Number of steps:    {n}")
    print()
    print(f"  Compound reliability: {compound:.1%}")
    print(f"  Compound failure rate: {summary['compound_failure_rate']:.1%}")
    print()
    print(f"  50% reliability cliff: step {summary['reliability_cliff_50pct']}")
    print(f"  20% reliability cliff: step {summary['reliability_cliff_20pct']}")
    print(
        f"  Expected steps before first failure: "
        f"{summary['expected_steps_before_failure']}"
    )

    if "actual_results" in summary:
        actual = summary["actual_results"]
        print()
        print("  Actual Stratix evaluation results:")
        print(f"    Steps evaluated:    {actual['steps_evaluated']}")
        print(f"    Steps passed:       {actual['steps_passed']}")
        print(f"    Steps failed:       {actual['steps_failed']}")
        print(f"    Actual pass rate:   " f"{actual['actual_per_step_pass_rate']:.1%}")
        print(
            f"    Actual compound:    " f"{actual['actual_compound_reliability']:.1%}"
        )
        print()
        print("  Per-step breakdown:")
        for detail in actual["per_step_details"]:
            status = "PASS" if detail.get("passed") else "FAIL"
            if detail.get("passed") is None:
                status = "N/A "
            score = detail.get("score", "n/a")
            name = detail["step_name"]
            reasoning = (detail.get("reasoning") or "")[:60]
            print(f"    [{status}] Step {detail['step_number']}: {name}")
            print(f"           Score: {score}  {reasoning}")

    print()
    print("=" * 64)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compound Failure Calculator. Evaluates multi-step AI agent "
            "traces with Stratix judges and computes compound failure "
            "probability."
        ),
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--trace-file",
        type=str,
        metavar="PATH",
        help="Path to a multi-step agent trace JSON file.",
    )
    input_group.add_argument(
        "--simulate",
        type=int,
        metavar="N",
        help=(
            "Simulate an N-step agent workflow using embedded sample data "
            "and compute compound failure mathematically."
        ),
    )

    parser.add_argument(
        "--per-step-accuracy",
        type=float,
        default=0.85,
        metavar="FLOAT",
        help="Per-step pass rate for simulation mode (default: 0.85).",
    )
    parser.add_argument(
        "--steps-range",
        type=int,
        nargs=2,
        default=[1, 15],
        metavar=("START", "END"),
        help=("Range of step counts for the visualization " "(default: 1 15)."),
    )
    parser.add_argument(
        "--output",
        type=str,
        metavar="PATH",
        help=(
            "Save visualization to a PNG file (requires matplotlib). "
            "If omitted, prints an ASCII chart to the terminal."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output results as JSON instead of formatted text.",
    )
    parser.add_argument(
        "--skip-cleanup",
        action="store_true",
        default=False,
        help="Keep created Stratix resources after evaluation.",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        default=False,
        help=(
            "Skip Stratix evaluation and only compute the mathematical "
            "compound failure curve. Useful for quick visualization."
        ),
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    steps_range = (args.steps_range[0], args.steps_range[1])
    per_step_accuracy = args.per_step_accuracy
    step_results: Optional[List[Dict[str, Any]]] = None

    if args.trace_file:
        # --- Real trace mode ---
        logger.info("Loading trace from: %s", args.trace_file)
        steps = parse_trace_steps(args.trace_file)
        if not steps:
            logger.error("No evaluable steps found in trace file.")
            sys.exit(1)
        logger.info("Found %d evaluable steps.", len(steps))
        num_steps = len(steps)

        if not args.skip_evaluation:
            try:
                client = Stratix()
            except Exception as exc:
                logger.error("Failed to initialize Stratix client: %s", exc)
                sys.exit(1)
            logger.info(
                "Connected to LayerLens (org=%s, project=%s)",
                client.organization_id,
                client.project_id,
            )
            step_results = evaluate_steps_with_stratix(
                client,
                steps,
                skip_cleanup=args.skip_cleanup,
            )

            # Compute actual pass rate from results
            scored = [r for r in step_results if r.get("passed") is not None]
            if scored:
                actual_rate = sum(1 for r in scored if r["passed"]) / len(scored)
                per_step_accuracy = actual_rate
                logger.info(
                    "Actual per-step pass rate from evaluation: %.1f%%",
                    actual_rate * 100,
                )
        else:
            logger.info("Skipping Stratix evaluation (math-only mode).")

    elif args.simulate is not None:
        # --- Simulation mode ---
        num_steps = args.simulate
        logger.info(
            "Simulating %d-step agent at %.1f%% per-step accuracy.",
            num_steps,
            per_step_accuracy * 100,
        )

        # Use embedded sample steps (up to num_steps)
        steps = SAMPLE_AGENT_STEPS[:num_steps]
        if len(steps) < num_steps:
            logger.info(
                "Sample data has %d steps; using those for evaluation, "
                "computing compound curve up to step %d mathematically.",
                len(steps),
                num_steps,
            )

        if not args.skip_evaluation:
            try:
                client = Stratix()
            except Exception as exc:
                logger.error("Failed to initialize Stratix client: %s", exc)
                sys.exit(1)
            logger.info(
                "Connected to LayerLens (org=%s, project=%s)",
                client.organization_id,
                client.project_id,
            )
            step_results = evaluate_steps_with_stratix(
                client,
                steps,
                skip_cleanup=args.skip_cleanup,
            )
        else:
            logger.info("Skipping Stratix evaluation (math-only mode).")

    else:
        parser.print_help()
        sys.exit(1)

    # --- Build summary ---
    summary = build_summary(per_step_accuracy, num_steps, step_results)

    # --- Output ---
    if args.json:
        print(json.dumps(summary, indent=2, default=str))
    else:
        print_summary(summary)
        print(
            render_ascii_chart(
                per_step_accuracy,
                steps_range,
                step_results,
            )
        )
        print(render_scenario_table())

    # --- Save chart if requested ---
    if args.output:
        save_matplotlib_chart(
            per_step_accuracy,
            steps_range,
            args.output,
            step_results,
        )

    logger.info("Compound failure analysis complete.")


if __name__ == "__main__":
    main()
