#!/usr/bin/env python
"""
Evaluation Pipeline -- LayerLens Python SDK Sample
===================================================

Port of ateam's "Demo 02 -- Evaluation Pipeline".  Replaces all raw
urllib HTTP calls with SDK calls to demonstrate a complete
judge-based evaluation workflow against production traces.

Workflow
--------
  1. List available judges.
  2. List recent traces to evaluate.
  3. Create a trace evaluation (judge a trace).
  4. Fetch evaluation results.
  5. Print a formatted evaluation report.

Prerequisites
-------------
* ``pip install layerlens --index-url https://sdk.layerlens.ai/package``
* Set ``LAYERLENS_STRATIX_API_KEY`` environment variable
* At least one judge configured in the project (create one with
  ``create_judge.py`` if needed).
* At least one trace uploaded (use ``basic_trace.py`` if needed).

Usage
-----
::

    export LAYERLENS_STRATIX_API_KEY=your-api-key
    python evaluation_pipeline.py
    python evaluation_pipeline.py --judge-id <id> --trace-id <id>
    python evaluation_pipeline.py --poll-interval 10 --poll-timeout 120
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from layerlens import Stratix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _helpers import poll_evaluation_results

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("layerlens.samples.evaluation_pipeline")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an evaluation pipeline with the LayerLens Python SDK.",
    )
    parser.add_argument(
        "--judge-id",
        default="",
        help="Judge ID to use. If omitted, the first available judge is used.",
    )
    parser.add_argument(
        "--trace-id",
        default="",
        help="Trace ID to evaluate. If omitted, the most recent trace is used.",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=5,
        help="Seconds between result polls (default: 5).",
    )
    parser.add_argument(
        "--poll-timeout",
        type=int,
        default=120,
        help="Maximum seconds to wait for results (default: 120).",
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # --- Initialize SDK client ---
    try:
        client = Stratix()
    except Exception as exc:
        logger.error("Failed to initialize client: %s", exc)
        sys.exit(1)

    logger.info(
        "Connected to LayerLens (org=%s, project=%s)",
        client.organization_id,
        client.project_id,
    )

    # ------------------------------------------------------------------
    # Step 1: List available judges
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 1: List available judges")
    logger.info("=" * 60)

    judges_response = client.judges.get_many()
    if not judges_response or not judges_response.judges:
        logger.error(
            "No judges found. Create one first (see create_judge.py)."
        )
        sys.exit(1)

    logger.info(
        "Found %d judge(s) (total=%d)",
        len(judges_response.judges),
        judges_response.total_count,
    )
    for j in judges_response.judges[:5]:
        logger.info(
            "  - %s  id=%s  version=%s",
            j.name,
            j.id,
            getattr(j, "version", "N/A"),
        )

    # Select the judge to use
    if args.judge_id:
        judge = client.judges.get(args.judge_id)
        if not judge:
            logger.error("Judge with ID '%s' not found", args.judge_id)
            sys.exit(1)
    else:
        judge = judges_response.judges[0]

    logger.info("Selected judge: %s (id=%s)", judge.name, judge.id)

    # ------------------------------------------------------------------
    # Step 2: List recent traces
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 2: List recent traces")
    logger.info("=" * 60)

    traces_response = client.traces.get_many(
        page_size=10,
        sort_by="created_at",
        sort_order="desc",
    )

    if not traces_response or not traces_response.traces:
        logger.error(
            "No traces found. Upload some first (see basic_trace.py)."
        )
        sys.exit(1)

    logger.info(
        "Fetched %d trace(s) (total available: %d)",
        traces_response.count,
        traces_response.total_count,
    )
    for t in traces_response.traces[:5]:
        logger.info(
            "  - id=%s  created=%s",
            t.id,
            getattr(t, "created_at", "N/A"),
        )

    # Select the trace to evaluate
    if args.trace_id:
        trace = client.traces.get(args.trace_id)
        if not trace:
            logger.error("Trace with ID '%s' not found", args.trace_id)
            sys.exit(1)
    else:
        trace = traces_response.traces[0]

    logger.info("Selected trace: %s", trace.id)

    # ------------------------------------------------------------------
    # Step 3: Create trace evaluation
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 3: Create trace evaluation")
    logger.info("=" * 60)

    trace_eval = client.trace_evaluations.create(
        trace_id=trace.id,
        judge_id=judge.id,
    )

    if not trace_eval:
        logger.error("Failed to create trace evaluation")
        sys.exit(1)

    logger.info(
        "Trace evaluation created: id=%s  status=%s",
        trace_eval.id,
        getattr(trace_eval, "status", "unknown"),
    )

    # ------------------------------------------------------------------
    # Step 4: Poll for and fetch evaluation results
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 4: Fetch evaluation results")
    logger.info("=" * 60)

    eval_results = poll_evaluation_results(client, trace_eval.id)
    if eval_results:
        logger.info("  Results ready.")
    else:
        logger.warning(
            "Timed out waiting for results. "
            "The evaluation may still be processing -- check later with "
            "trace_evaluation ID: %s",
            trace_eval.id,
        )

    # ------------------------------------------------------------------
    # Step 5: Evaluation report
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 5: Evaluation report")
    logger.info("=" * 60)

    print()
    print("=" * 70)
    print("  EVALUATION PIPELINE REPORT")
    print("=" * 70)
    print(f"  Judge:              {judge.name} (id={judge.id})")
    print(f"  Judge version:      {getattr(judge, 'version', 'N/A')}")
    print(f"  Trace ID:           {trace.id}")
    print(f"  Evaluation ID:      {trace_eval.id}")
    print(
        f"  Evaluation status:  "
        f"{getattr(trace_eval, 'status', 'unknown')}"
    )
    print("-" * 70)

    if eval_results:
        print(f"  Results ({len(eval_results)} item(s)):")
        print()
        for i, result in enumerate(eval_results, 1):
            score = result.score
            score_str = (
                f"{score:.2f}" if isinstance(score, (int, float)) else "N/A"
            )
            passed = result.passed
            reasoning = result.reasoning or ""
            reasoning_preview = (
                reasoning[:100] + "..." if reasoning else "N/A"
            )

            print(f"  [{i}] Score:     {score_str}")
            print(f"      Passed:    {passed}")
            print(f"      Reasoning: {reasoning_preview}")
            print()
    else:
        print("  No results available yet.")
        print(
            f"  Re-run with --trace-id and check evaluation {trace_eval.id}"
        )
        print()

    print("=" * 70)
    print()

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
