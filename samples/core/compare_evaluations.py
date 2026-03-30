#!/usr/bin/env python
"""
Evaluation Comparison -- LayerLens Python SDK Sample
====================================================

Demonstrates evaluation comparison using the SDK:

  1. List evaluations with filtering and sorting.
  2. Compare two evaluations side-by-side.
  3. Compare two models on the same benchmark.

This sample ports concepts from the existing SDK example
compare_evaluations.py into a standalone runnable sample.

Prerequisites
-------------
* ``pip install layerlens --index-url https://sdk.layerlens.ai/package``
* Set ``LAYERLENS_STRATIX_API_KEY`` environment variable
* At least two completed evaluations on the same benchmark

Usage
-----
::

    export LAYERLENS_STRATIX_API_KEY=your-api-key
    python compare_evaluations.py
    python compare_evaluations.py --eval-id-1 <ID1> --eval-id-2 <ID2>
"""

from __future__ import annotations

import argparse
import logging
import sys

from layerlens import Stratix

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("layerlens.samples.compare_evaluations")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare evaluations with the LayerLens Python SDK.",
    )
    parser.add_argument(
        "--eval-id-1",
        default="",
        help="First evaluation ID to compare.",
    )
    parser.add_argument(
        "--eval-id-2",
        default="",
        help="Second evaluation ID to compare.",
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

    # --- Step 1: List evaluations ---
    logger.info("=" * 60)
    logger.info("Step 1: List evaluations")
    logger.info("=" * 60)

    evals_resp = client.evaluations.get_many(
        sort_by="submittedAt",
        order="desc",
        page_size=10,
    )

    if not evals_resp or not evals_resp.evaluations:
        logger.error("No evaluations found. Run evaluations first.")
        sys.exit(1)

    logger.info("Found %d evaluation(s)", len(evals_resp.evaluations))
    for e in evals_resp.evaluations[:5]:
        accuracy = getattr(e, "accuracy", None)
        accuracy_str = f"{accuracy:.2%}" if isinstance(accuracy, (int, float)) else "N/A"
        logger.info("  - %s: status=%s accuracy=%s", e.id, e.status, accuracy_str)

    # --- Step 2: Compare evaluations ---
    logger.info("=" * 60)
    logger.info("Step 2: Compare evaluations")
    logger.info("=" * 60)

    if args.eval_id_1 and args.eval_id_2:
        eval_id_1 = args.eval_id_1
        eval_id_2 = args.eval_id_2
    elif len(evals_resp.evaluations) >= 2:
        eval_id_1 = str(evals_resp.evaluations[0].id)
        eval_id_2 = str(evals_resp.evaluations[1].id)
        logger.info("Using two most recent evaluations for comparison")
    else:
        logger.error("Need at least 2 evaluations. Only found %d.", len(evals_resp.evaluations))
        sys.exit(1)

    logger.info("Comparing: %s vs %s", eval_id_1, eval_id_2)

    comparison = client.public.comparisons.compare(
        evaluation_id_1=eval_id_1,
        evaluation_id_2=eval_id_2,
    )

    if comparison:
        logger.info("Comparison results:")
        if hasattr(comparison, "results") and comparison.results:
            for i, r in enumerate(comparison.results[:10], 1):
                logger.info("  [%d] %s", i, r)
        else:
            logger.info("  %s", comparison)
    else:
        logger.warning("Comparison returned no results (evaluations may use different benchmarks)")

    logger.info("Sample complete.")


if __name__ == "__main__":
    main()
