#!/usr/bin/env python
"""
Judge CRUD -- LayerLens Python SDK Sample
==========================================

Demonstrates the full judge lifecycle using the SDK:

  1. List available models to pick a backing LLM.
  2. Create a judge with a name and evaluation goal.
  3. Get the judge by ID.
  4. List all judges with pagination.
  5. Update the judge.
  6. Delete the judge.

This sample ports the ateam core/create_judge.py sample to use the
layerlens SDK client instead of raw httpx calls.

Prerequisites
-------------
* ``pip install layerlens --index-url https://sdk.layerlens.ai/package``
* Set ``LAYERLENS_STRATIX_API_KEY`` environment variable

Usage
-----
::

    export LAYERLENS_STRATIX_API_KEY=your-api-key
    python create_judge.py
    python create_judge.py --model-name gpt-4o --skip-delete
"""

from __future__ import annotations

import os
import sys
import time
import logging
import argparse

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
logger = logging.getLogger("layerlens.samples.create_judge")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Judge CRUD operations with the LayerLens Python SDK.",
    )
    parser.add_argument(
        "--model-name",
        default="gpt-4o",
        help="Model name to use as the judge's LLM (default: gpt-4o).",
    )
    parser.add_argument(
        "--skip-delete",
        action="store_true",
        default=False,
        help="Keep the judge after the sample completes.",
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

    logger.info("Connected to LayerLens (org=%s, project=%s)", client.organization_id, client.project_id)

    # --- Step 1: Find a model for the judge ---
    logger.info("=" * 60)
    logger.info("Step 1: Find a model for the judge")
    logger.info("=" * 60)

    models = client.models.get(type="public", name=args.model_name)
    if not models:
        logger.warning("No models found matching '%s', trying all public models...", args.model_name)
        models = client.models.get(type="public")

    if not models:
        logger.error("No models available. Cannot create a judge without a backing model.")
        sys.exit(1)

    model = models[0]
    logger.info("Using model: %s (id=%s)", model.name, model.id)

    # --- Step 2: Create a judge ---
    logger.info("=" * 60)
    logger.info("Step 2: Create a judge")
    logger.info("=" * 60)

    judge_name = f"SDK Sample Judge {int(time.time())}"
    judge = create_judge(
        client,
        name=judge_name,
        evaluation_goal="Evaluate the quality and accuracy of AI-generated responses, checking for correctness, completeness, and clarity.",
        model_id=model.id,
    )

    if not judge:
        logger.error("Failed to create judge")
        sys.exit(1)

    logger.info("Judge created:")
    logger.info("  ID:      %s", judge.id)
    logger.info("  Name:    %s", judge.name)
    logger.info("  Version: %s", getattr(judge, "version", "N/A"))

    # --- Step 3: Get the judge by ID ---
    logger.info("=" * 60)
    logger.info("Step 3: Get judge by ID")
    logger.info("=" * 60)

    fetched = client.judges.get(judge.id)
    if fetched:
        logger.info("Judge retrieved: %s (version=%s)", fetched.name, getattr(fetched, "version", "N/A"))
    else:
        logger.warning("Could not retrieve judge %s", judge.id)

    # --- Step 4: List all judges ---
    logger.info("=" * 60)
    logger.info("Step 4: List all judges")
    logger.info("=" * 60)

    response = client.judges.get_many()
    if response:
        logger.info("Found %d judge(s) (total=%d)", len(response.judges), response.total_count)
        for j in response.judges[:5]:
            logger.info("  - %s (v%s, %d runs)", j.name, getattr(j, "version", "?"), getattr(j, "run_count", 0))
    else:
        logger.warning("No judges found")

    # --- Step 5: Update the judge ---
    logger.info("=" * 60)
    logger.info("Step 5: Update the judge")
    logger.info("=" * 60)

    updated = client.judges.update(
        judge.id,
        name=f"Updated {judge_name}",
        evaluation_goal="Evaluate AI responses for correctness, completeness, clarity, and safety compliance.",
    )

    if updated:
        logger.info("Judge updated: id=%s", updated.id)
    else:
        logger.warning("Judge update returned no confirmation")

    # --- Step 6: Delete the judge ---
    if not args.skip_delete:
        logger.info("=" * 60)
        logger.info("Step 6: Delete the judge")
        logger.info("=" * 60)

        deleted = client.judges.delete(judge.id)
        if deleted:
            logger.info("Judge %s deleted (id=%s)", judge_name, deleted.id)
        else:
            logger.warning("Judge deletion returned no confirmation")
    else:
        logger.info("Skipping deletion (--skip-delete). Judge ID: %s", judge.id)

    logger.info("Sample complete.")


if __name__ == "__main__":
    main()
