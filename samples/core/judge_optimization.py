#!/usr/bin/env python
"""
Judge Optimization -- LayerLens Python SDK Sample
==================================================

Demonstrates the judge optimization workflow using the SDK:

  1. Create a judge.
  2. Estimate optimization cost.
  3. Start an optimization run.
  4. Poll for optimization completion.
  5. List optimization runs.
  6. Apply optimization results.
  7. Clean up.

This sample demonstrates SDK features that correspond to the
judge_optimizations.py example in the existing SDK examples.

Prerequisites
-------------
* ``pip install layerlens --index-url https://sdk.layerlens.ai/package``
* Set ``LAYERLENS_STRATIX_API_KEY`` environment variable
* At least one judge with trace evaluations completed

Usage
-----
::

    export LAYERLENS_STRATIX_API_KEY=your-api-key
    python judge_optimization.py --judge-id <JUDGE_ID>
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

from layerlens import Stratix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _helpers import create_judge

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("layerlens.samples.judge_optimization")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Judge optimization with the LayerLens Python SDK.",
    )
    parser.add_argument(
        "--judge-id",
        default="",
        help="ID of an existing judge to optimize. If omitted, creates one.",
    )
    parser.add_argument(
        "--budget",
        choices=["low", "medium", "high"],
        default="medium",
        help="Optimization budget (default: medium).",
    )
    parser.add_argument(
        "--skip-apply",
        action="store_true",
        default=False,
        help="Skip applying the optimization results.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        client = Stratix()
    except Exception as exc:
        logger.error("Failed to initialize client: %s", exc)
        sys.exit(1)

    logger.info("Connected to LayerLens (org=%s, project=%s)", client.organization_id, client.project_id)

    # --- Get or create judge ---
    if args.judge_id:
        judge = client.judges.get(args.judge_id)
        if not judge:
            logger.error("Judge %s not found", args.judge_id)
            sys.exit(1)
        logger.info("Using existing judge: %s (%s)", judge.name, judge.id)
        judge_id = judge.id
    else:
        # Find a model for the judge
        models = client.models.get(type="public")
        if not models:
            logger.error("No public models available")
            sys.exit(1)

        judge = create_judge(
            client,
            name=f"Optimization Sample Judge {int(time.time())}",
            evaluation_goal="Evaluate AI response quality for accuracy and completeness.",
            model_id=models[0].id,
        )
        if not judge:
            logger.error("Failed to create judge")
            sys.exit(1)
        judge_id = judge.id
        logger.info("Created judge: %s (%s)", judge.name, judge_id)

    # --- Step 1: Estimate cost ---
    logger.info("=" * 60)
    logger.info("Step 1: Estimate optimization cost")
    logger.info("=" * 60)

    estimate = client.judge_optimizations.estimate(
        judge_id=judge_id,
        budget=args.budget,
    )
    if estimate:
        logger.info("Cost estimate: %s", estimate)
    else:
        logger.info("Cost estimation not available")

    # --- Step 2: Create optimization run ---
    logger.info("=" * 60)
    logger.info("Step 2: Create optimization run")
    logger.info("=" * 60)

    run = client.judge_optimizations.create(
        judge_id=judge_id,
        budget=args.budget,
    )
    if not run:
        logger.error("Failed to create optimization run")
        sys.exit(1)
    logger.info("Optimization run created: %s", run.id)

    # --- Step 3: Poll for completion ---
    logger.info("=" * 60)
    logger.info("Step 3: Poll for completion")
    logger.info("=" * 60)

    max_attempts = 30
    poll_delay = 5.0
    max_delay = 60.0
    backoff_factor = 1.5
    for attempt in range(1, max_attempts + 1):
        run_status = client.judge_optimizations.get(run.id)
        if not run_status:
            logger.warning("Could not fetch run status (attempt %d/%d)", attempt, max_attempts)
            time.sleep(poll_delay)
            poll_delay = min(poll_delay * backoff_factor, max_delay)
            continue

        status = getattr(run_status, "status", "unknown")
        logger.info("  Run %s: status=%s (attempt %d/%d)", run.id, status, attempt, max_attempts)

        if status in ("completed", "failed", "cancelled"):
            break

        time.sleep(poll_delay)
        poll_delay = min(poll_delay * backoff_factor, max_delay)
    else:
        logger.warning("Optimization did not complete within %d attempts", max_attempts)

    # --- Step 4: List runs ---
    logger.info("=" * 60)
    logger.info("Step 4: List optimization runs")
    logger.info("=" * 60)

    runs_resp = client.judge_optimizations.get_many(judge_id=judge_id)
    if runs_resp:
        logger.info("Found %d optimization run(s)", runs_resp.count)
        for r in runs_resp.optimization_runs:
            logger.info("  - %s: status=%s", r.id, getattr(r, "status", "unknown"))
    else:
        logger.info("No optimization runs found")

    # --- Step 5: Apply results ---
    if not args.skip_apply and run_status and getattr(run_status, "status", "") == "completed":
        logger.info("=" * 60)
        logger.info("Step 5: Apply optimization results")
        logger.info("=" * 60)

        applied = client.judge_optimizations.apply(run.id)
        if applied:
            logger.info("Optimization results applied: %s", applied)
        else:
            logger.warning("Failed to apply optimization results")
    else:
        logger.info("Skipping apply step")

    logger.info("Sample complete.")


if __name__ == "__main__":
    main()
