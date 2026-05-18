#!/usr/bin/env python
"""
Quality Gate -- LayerLens Python SDK CI/CD Sample
=================================================

Evaluates recent traces against all configured judges and blocks the
pipeline if the overall pass rate falls below a threshold.

Designed to be called from a GitHub Actions workflow (see
``github_actions_gate.yml``) but works in any CI environment where
``LAYERLENS_STRATIX_API_KEY`` is set.

Flow
----
1. Initialize ``Stratix()`` client (reads API key from env automatically).
2. Fetch recent traces via ``client.traces.get_many()``.
3. Fetch judges via ``client.judges.get_many()``.
4. Create a trace evaluation for each (trace, judge) pair.
5. Poll for results and compute a pass rate.
6. Print a formatted report.
7. Exit 1 if the pass rate is below the threshold.

Prerequisites
-------------
* ``pip install layerlens --index-url https://sdk.layerlens.ai/package``
* Set ``LAYERLENS_STRATIX_API_KEY`` environment variable

Usage
-----
::

    export LAYERLENS_STRATIX_API_KEY=your-api-key
    python quality_gate.py --threshold 0.85
"""

from __future__ import annotations

import os
import sys
import logging
import argparse

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
logger = logging.getLogger("layerlens.cicd.quality_gate")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLD = 0.85
TRACE_PAGE_SIZE = 50
# Bound the total number of (trace, judge) evaluation pairs to avoid
# excessive API calls and long CI runtimes.  Evaluations are created in
# round-robin order until the cap is reached.
MAX_EVALUATIONS = 200


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AI quality gate using the LayerLens Python SDK.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Minimum pass rate to succeed (default: {DEFAULT_THRESHOLD}).",
    )
    parser.add_argument(
        "--max-traces",
        type=int,
        default=TRACE_PAGE_SIZE,
        help=f"Maximum number of recent traces to evaluate (default: {TRACE_PAGE_SIZE}).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    threshold: float = args.threshold
    max_traces: int = args.max_traces

    # ------------------------------------------------------------------
    # Step 1: Connect
    # ------------------------------------------------------------------
    try:
        client = Stratix()
    except Exception as exc:
        logger.error("Failed to initialize LayerLens client: %s", exc)
        sys.exit(1)

    logger.info(
        "Connected to LayerLens (org=%s, project=%s)",
        client.organization_id,
        client.project_id,
    )

    # ------------------------------------------------------------------
    # Step 2: Fetch recent traces
    # ------------------------------------------------------------------
    logger.info("Fetching up to %d recent traces...", max_traces)
    traces_resp = client.traces.get_many(page_size=max_traces)
    if not traces_resp or not traces_resp.traces:
        logger.error("No traces found -- nothing to evaluate.")
        sys.exit(1)

    traces = traces_resp.traces
    logger.info("Found %d trace(s) (total in project: %d)", len(traces), traces_resp.total_count)

    # ------------------------------------------------------------------
    # Step 3: Fetch judges
    # ------------------------------------------------------------------
    logger.info("Fetching judges...")
    judges_resp = client.judges.get_many()
    if not judges_resp or not judges_resp.judges:
        logger.error("No judges configured -- create at least one judge first.")
        sys.exit(1)

    judges = judges_resp.judges
    logger.info("Found %d judge(s)", len(judges))

    # ------------------------------------------------------------------
    # Step 4: Create trace evaluations
    # ------------------------------------------------------------------
    # NOTE: Rate limiting -- the loop below creates evaluations sequentially.
    # For large batches, consider adding a small delay between requests to
    # avoid hitting API rate limits.
    logger.info("Creating trace evaluations (max %d)...", MAX_EVALUATIONS)
    eval_ids: list[str] = []
    for trace in traces:
        for judge in judges:
            if len(eval_ids) >= MAX_EVALUATIONS:
                logger.info("  Reached MAX_EVALUATIONS cap (%d). Stopping.", MAX_EVALUATIONS)
                break
            te = client.trace_evaluations.create(
                trace_id=trace.id,
                judge_id=judge.id,
            )
            if te:
                eval_ids.append(te.id)
                logger.info("  Created evaluation %s (trace=%s, judge=%s)", te.id, trace.id, judge.id)
            else:
                logger.warning("  Failed to create evaluation (trace=%s, judge=%s)", trace.id, judge.id)
        if len(eval_ids) >= MAX_EVALUATIONS:
            break

    if not eval_ids:
        logger.error("No evaluations could be created.")
        sys.exit(1)

    logger.info("Created %d evaluation(s). Waiting for results...", len(eval_ids))

    # ------------------------------------------------------------------
    # Step 5: Poll for results and compute pass rate
    # ------------------------------------------------------------------
    passed = 0
    failed = 0
    results_detail: list[dict] = []
    pending_count = 0

    for eval_id in eval_ids:
        results = poll_evaluation_results(client, eval_id)
        if results:
            for r in results:
                results_detail.append(
                    {
                        "eval_id": eval_id,
                        "score": r.score,
                        "passed": r.passed,
                        "reasoning": r.reasoning,
                    }
                )
                if r.passed:
                    passed += 1
                else:
                    failed += 1
        else:
            failed += 1
            pending_count += 1
            logger.warning("  Evaluation %s did not return results in time", eval_id)

    if pending_count:
        logger.warning("%d evaluation(s) did not complete in time", pending_count)

    # ------------------------------------------------------------------
    # Step 6: Report
    # ------------------------------------------------------------------
    total = passed + failed
    pass_rate = passed / total if total > 0 else 0.0

    print()
    print("=" * 60)
    print("  AI Quality Gate Report")
    print("=" * 60)
    print(f"  Traces evaluated : {len(traces)}")
    print(f"  Judges used      : {len(judges)}")
    print(f"  Total results    : {total}")
    print(f"  Passed           : {passed}")
    print(f"  Failed           : {failed}")
    print(f"  Pass rate        : {pass_rate:.1%}")
    print(f"  Threshold        : {threshold:.1%}")
    print("-" * 60)

    if results_detail:
        print("  Detailed Results:")
        for rd in results_detail:
            status = "PASS" if rd["passed"] else "FAIL"
            print(f"    [{status}] score={rd['score']:.2f}  eval={rd['eval_id'][:12]}...")
        print("-" * 60)

    # ------------------------------------------------------------------
    # Step 7: Gate decision
    # ------------------------------------------------------------------
    if pass_rate >= threshold:
        print(f"  RESULT: PASSED (pass rate {pass_rate:.1%} >= {threshold:.1%})")
        print("=" * 60)
        print()
    else:
        print(f"  RESULT: FAILED (pass rate {pass_rate:.1%} < {threshold:.1%})")
        print("=" * 60)
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()
