#!/usr/bin/env python
"""
Pre-Commit Hook -- LayerLens Python SDK CI/CD Sample
=====================================================

A Git pre-commit hook that runs a quick safety evaluation against
recent traces before allowing a commit.  If the pass rate drops below
a configurable threshold the commit is blocked.

Installation
------------
Option A -- symlink::

    ln -sf "$(pwd)/samples/cicd/pre_commit_hook.py" .git/hooks/pre-commit
    chmod +x .git/hooks/pre-commit

Option B -- using the ``pre-commit`` framework (``.pre-commit-config.yaml``)::

    repos:
      - repo: local
        hooks:
          - id: layerlens-safety
            name: LayerLens Safety Gate
            entry: python samples/cicd/pre_commit_hook.py
            language: python
            additional_dependencies: ["layerlens"]
            pass_filenames: false

Prerequisites
-------------
* ``pip install layerlens --index-url https://sdk.layerlens.ai/package``
* Set ``LAYERLENS_STRATIX_API_KEY`` environment variable

Usage
-----
::

    export LAYERLENS_STRATIX_API_KEY=your-api-key
    python pre_commit_hook.py
"""

from __future__ import annotations

import os
import sys
import logging
import subprocess

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
logger = logging.getLogger("layerlens.cicd.pre_commit_hook")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RELEVANT_EXTENSIONS = {".py", ".yaml", ".yml", ".json", ".jsonl", ".txt", ".md"}
RELEVANT_PATHS = {"src/", "prompts/", "agents/", "config/"}
SAFETY_THRESHOLD = 0.90
SMOKE_TRACE_COUNT = 10


def get_staged_files() -> list[str]:
    """Return list of staged file paths using ``git diff --cached``."""
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
            capture_output=True,
            text=True,
            check=True,
        )
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except subprocess.CalledProcessError:
        return []


def has_relevant_changes(files: list[str]) -> bool:
    """Check whether any staged files are in a path or have an extension we care about."""
    for f in files:
        _, ext = os.path.splitext(f)
        if ext.lower() in RELEVANT_EXTENSIONS:
            for prefix in RELEVANT_PATHS:
                if f.startswith(prefix):
                    return True
    return False


def main() -> None:
    # ------------------------------------------------------------------
    # Check staged files
    # ------------------------------------------------------------------
    staged = get_staged_files()
    if not staged:
        logger.info("No staged files -- skipping safety check.")
        sys.exit(0)

    if not has_relevant_changes(staged):
        logger.info("No relevant file changes -- skipping safety check.")
        sys.exit(0)

    logger.info("Relevant files staged (%d). Running safety smoke test...", len(staged))

    # ------------------------------------------------------------------
    # Initialize client
    # ------------------------------------------------------------------
    try:
        client = Stratix()
    except Exception as exc:
        logger.warning("LayerLens client init failed (%s) -- allowing commit.", exc)
        sys.exit(0)

    logger.info(
        "Connected to LayerLens (org=%s, project=%s)",
        client.organization_id,
        client.project_id,
    )

    # ------------------------------------------------------------------
    # Fetch a small set of recent traces
    # ------------------------------------------------------------------
    traces_resp = client.traces.get_many(page_size=SMOKE_TRACE_COUNT)
    if not traces_resp or not traces_resp.traces:
        logger.info("No traces available -- skipping safety check.")
        sys.exit(0)

    traces = traces_resp.traces
    logger.info("Fetched %d recent trace(s) for smoke test.", len(traces))

    # ------------------------------------------------------------------
    # Find a safety-oriented judge
    # ------------------------------------------------------------------
    judges_resp = client.judges.get_many()
    if not judges_resp or not judges_resp.judges:
        logger.info("No judges configured -- skipping safety check.")
        sys.exit(0)

    # Prefer a judge whose name contains "safety"; fall back to first available
    safety_judge = None
    for judge in judges_resp.judges:
        if "safety" in (judge.name or "").lower():
            safety_judge = judge
            break
    if safety_judge is None:
        safety_judge = judges_resp.judges[0]

    logger.info("Using judge: %s (%s)", safety_judge.name, safety_judge.id)

    # ------------------------------------------------------------------
    # Create trace evaluations
    # ------------------------------------------------------------------
    eval_ids: list[str] = []
    for trace in traces:
        te = client.trace_evaluations.create(
            trace_id=trace.id,
            judge_id=safety_judge.id,
        )
        if te:
            eval_ids.append(te.id)

    if not eval_ids:
        logger.warning("Could not create evaluations -- allowing commit.")
        sys.exit(0)

    logger.info("Created %d evaluation(s). Polling for results...", len(eval_ids))

    # ------------------------------------------------------------------
    # Poll for results
    # ------------------------------------------------------------------
    passed = 0
    failed = 0

    for eval_id in eval_ids:
        results = poll_evaluation_results(client, eval_id)
        if results:
            for r in results:
                if r.passed:
                    passed += 1
                else:
                    failed += 1
        else:
            failed += 1

    # ------------------------------------------------------------------
    # Gate decision
    # ------------------------------------------------------------------
    total = passed + failed
    pass_rate = passed / total if total > 0 else 0.0

    print()
    print("-" * 50)
    print("  LayerLens Pre-Commit Safety Check")
    print("-" * 50)
    print(f"  Traces   : {len(traces)}")
    print(f"  Judge    : {safety_judge.name}")
    print(f"  Results  : {total} ({passed} passed, {failed} failed)")
    print(f"  Pass rate: {pass_rate:.1%}")
    print(f"  Threshold: {SAFETY_THRESHOLD:.1%}")
    print("-" * 50)

    if pass_rate >= SAFETY_THRESHOLD:
        print("  COMMIT ALLOWED")
        print("-" * 50)
        print()
    else:
        print("  COMMIT BLOCKED -- safety check failed")
        print("-" * 50)
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()
