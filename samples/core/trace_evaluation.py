#!/usr/bin/env python
"""
Trace Evaluation -- LayerLens Python SDK Sample
================================================

Demonstrates the trace evaluation workflow using the SDK:

  1. Upload traces from a JSONL file.
  2. Create a judge.
  3. Estimate the cost of evaluating traces with the judge.
  4. Run a trace evaluation (judge against trace).
  5. Poll for results.
  6. Fetch and display evaluation results.
  7. Clean up (delete judge and traces).

This combines concepts from the ateam core/run_evaluation.py and
core/create_judge.py samples, adapted for the SDK's trace evaluation
resource.

Prerequisites
-------------
* ``pip install layerlens --index-url https://sdk.layerlens.ai/package``
* Set ``LAYERLENS_STRATIX_API_KEY`` environment variable

Usage
-----
::

    export LAYERLENS_STRATIX_API_KEY=your-api-key
    python trace_evaluation.py
    python trace_evaluation.py --skip-cleanup
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time

from layerlens import Stratix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _helpers import poll_evaluation_results, create_judge

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("layerlens.samples.trace_evaluation")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trace evaluation with the LayerLens Python SDK.",
    )
    parser.add_argument(
        "--skip-cleanup",
        action="store_true",
        default=False,
        help="Keep created resources after the sample completes.",
    )
    return parser


def generate_sample_traces() -> str:
    """Generate a temporary JSONL file with sample trace data."""
    traces = [
        {
            "input": [{"role": "user", "content": "What is the capital of France?"}],
            "output": "The capital of France is Paris.",
            "metadata": {"model": "gpt-4o", "source": "trace-eval-sample"},
        },
        {
            "input": [{"role": "user", "content": "Explain quantum computing in simple terms."}],
            "output": "Quantum computing uses quantum bits (qubits) that can exist in multiple states simultaneously, enabling certain calculations to be performed much faster than classical computers.",
            "metadata": {"model": "gpt-4o", "source": "trace-eval-sample"},
        },
    ]
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as f:
        for trace in traces:
            f.write(json.dumps(trace) + "\n")
    return path


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        client = Stratix()
    except Exception as exc:
        logger.error("Failed to initialize client: %s", exc)
        sys.exit(1)

    logger.info("Connected to LayerLens (org=%s, project=%s)", client.organization_id, client.project_id)

    created_trace_ids = []
    created_judge_id = None
    temp_file = None

    try:
        # --- Step 1: Upload traces ---
        logger.info("Step 1: Upload traces")
        temp_file = generate_sample_traces()
        result = client.traces.upload(temp_file)
        if not result or not result.trace_ids:
            logger.error("Upload failed or returned no trace IDs")
            sys.exit(1)
        created_trace_ids = result.trace_ids
        logger.info("Uploaded %d trace(s)", len(created_trace_ids))

        # --- Step 2: Create a judge ---
        logger.info("Step 2: Create a judge")
        models = client.models.get(type="public")
        if not models:
            logger.error("No public models available")
            sys.exit(1)

        judge = create_judge(
            client,
            name=f"Trace Eval Sample Judge {int(time.time())}",
            evaluation_goal="Evaluate the factual accuracy and completeness of AI responses.",
            model_id=models[0].id,
        )
        if not judge:
            logger.error("Failed to create judge")
            sys.exit(1)
        created_judge_id = judge.id
        logger.info("Judge created: %s (%s)", judge.name, judge.id)

        # --- Step 3: Estimate cost ---
        logger.info("Step 3: Estimate evaluation cost")
        estimate = client.trace_evaluations.estimate_cost(
            trace_ids=created_trace_ids,
            judge_id=judge.id,
        )
        if estimate:
            logger.info("Cost estimate: %s", estimate)
        else:
            logger.info("Cost estimation not available (proceeding anyway)")

        # --- Step 4: Create trace evaluation ---
        logger.info("Step 4: Create trace evaluation")
        trace_eval = client.trace_evaluations.create(
            trace_id=created_trace_ids[0],
            judge_id=judge.id,
        )
        if not trace_eval:
            logger.error("Failed to create trace evaluation")
            sys.exit(1)
        logger.info("Trace evaluation created: %s (status=%s)",
                     trace_eval.id, getattr(trace_eval, "status", "unknown"))

        # --- Step 5: Poll and fetch results ---
        logger.info("Step 5: Fetch results")

        eval_results = poll_evaluation_results(client, trace_eval.id)
        if eval_results:
            logger.info("Got %d result(s)", len(eval_results))
            for r in eval_results:
                logger.info("  Score: %s  Passed: %s  Reasoning: %s",
                            r.score,
                            r.passed,
                            (r.reasoning or "")[:80])
        else:
            logger.info("No results yet (evaluation may still be processing)")

        # --- Step 6: List trace evaluations ---
        logger.info("Step 6: List trace evaluations")
        evals_resp = client.trace_evaluations.get_many(judge_id=judge.id)
        if evals_resp:
            logger.info("Found %d trace evaluation(s)", evals_resp.count)
        else:
            logger.info("No trace evaluations found")

    finally:
        # --- Cleanup ---
        if not args.skip_cleanup:
            logger.info("Cleaning up...")
            if created_judge_id:
                client.judges.delete(created_judge_id)
                logger.info("  Deleted judge %s", created_judge_id)
            for tid in created_trace_ids:
                client.traces.delete(tid)
                logger.info("  Deleted trace %s", tid)
        else:
            logger.info("Skipping cleanup (--skip-cleanup)")
            if created_judge_id:
                logger.info("  Judge ID: %s", created_judge_id)
            if created_trace_ids:
                logger.info("  Trace IDs: %s", ", ".join(created_trace_ids))

        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)

    logger.info("Sample complete.")


if __name__ == "__main__":
    main()
