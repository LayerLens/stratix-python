#!/usr/bin/env python
"""
Basic Trace Operations -- LayerLens Python SDK Sample
=====================================================

Demonstrates trace operations using the LayerLens Python SDK:

  1. Upload traces from a JSONL file.
  2. List traces with filtering and pagination.
  3. Get a single trace by ID.
  4. Get available trace sources.
  5. Delete a trace.

This sample ports the ateam core/basic_trace.py sample to use the
layerlens SDK client instead of raw httpx calls.

Prerequisites
-------------
* ``pip install layerlens --index-url https://sdk.layerlens.ai/package``
* Set ``LAYERLENS_STRATIX_API_KEY`` environment variable
* A traces.jsonl file (see samples/data/traces/ for format)

Usage
-----
::

    export LAYERLENS_STRATIX_API_KEY=your-api-key
    python basic_trace.py
    python basic_trace.py --file /path/to/traces.jsonl
"""

from __future__ import annotations

import os
import sys
import json
import logging
import argparse
import tempfile

from layerlens import Stratix

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("layerlens.samples.basic_trace")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trace CRUD operations with the LayerLens Python SDK.",
    )
    parser.add_argument(
        "--file",
        default="",
        help="Path to a JSONL trace file to upload. If omitted, sample data is generated.",
    )
    parser.add_argument(
        "--skip-delete",
        action="store_true",
        default=False,
        help="Keep traces on the platform after the sample completes.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=10,
        help="Number of traces to list per page (default: 10).",
    )
    return parser


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------


def generate_sample_traces() -> str:
    """Generate a temporary JSONL file with sample trace data.

    Returns the path to the temporary file.
    """
    traces = [
        {
            "input": [{"role": "user", "content": "What is the capital of France?"}],
            "output": "The capital of France is Paris.",
            "metadata": {"model": "gpt-4o", "temperature": 0.7, "source": "sdk-sample"},
        },
        {
            "input": [{"role": "user", "content": "Explain photosynthesis briefly."}],
            "output": "Photosynthesis is the process by which plants convert sunlight, water, and CO2 into glucose and oxygen.",
            "metadata": {"model": "gpt-4o", "temperature": 0.7, "source": "sdk-sample"},
        },
        {
            "input": [{"role": "user", "content": "What is binary search?"}],
            "output": "Binary search is an efficient algorithm that finds a target value in a sorted array by repeatedly dividing the search interval in half, achieving O(log n) time complexity.",
            "metadata": {"model": "gpt-4o", "temperature": 0.7, "source": "sdk-sample"},
        },
    ]

    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as f:
        for trace in traces:
            f.write(json.dumps(trace) + "\n")
    return path


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

    # --- Step 1: Upload traces ---
    logger.info("=" * 60)
    logger.info("Step 1: Upload traces")
    logger.info("=" * 60)

    temp_file = None
    if args.file:
        if not os.path.isfile(args.file):
            logger.error("File not found: %s", args.file)
            sys.exit(1)
        file_path = args.file
    else:
        file_path = generate_sample_traces()
        temp_file = file_path
        logger.info("Generated sample traces at %s", file_path)

    try:
        result = client.traces.upload(file_path)
        if result and result.trace_ids:
            logger.info("Uploaded %d trace(s)", len(result.trace_ids))
            for tid in result.trace_ids:
                logger.info("  trace_id=%s", tid)
        else:
            logger.warning("Upload returned no trace IDs")
            sys.exit(1)
    except Exception as exc:
        logger.error("Upload failed: %s", exc)
        sys.exit(1)
    finally:
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)

    uploaded_ids = result.trace_ids

    # --- Step 2: List traces ---
    logger.info("=" * 60)
    logger.info("Step 2: List traces")
    logger.info("=" * 60)

    response = client.traces.get_many(page_size=args.page_size, sort_by="created_at", sort_order="desc")
    if response:
        logger.info("Found %d trace(s) (total=%d)", response.count, response.total_count)
        for trace in response.traces[:5]:
            logger.info("  - %s: %s", trace.id, getattr(trace, "filename", "N/A"))
    else:
        logger.warning("No traces found")

    # --- Step 3: Get a single trace ---
    logger.info("=" * 60)
    logger.info("Step 3: Get a single trace")
    logger.info("=" * 60)

    trace = client.traces.get(uploaded_ids[0])
    if trace:
        logger.info("Trace %s retrieved successfully", trace.id)
        logger.info("  Data keys: %s", list(trace.data.keys()) if hasattr(trace, "data") and trace.data else "N/A")
    else:
        logger.warning("Could not retrieve trace %s", uploaded_ids[0])

    # --- Step 4: Get sources ---
    logger.info("=" * 60)
    logger.info("Step 4: Get trace sources")
    logger.info("=" * 60)

    sources = client.traces.get_sources()
    logger.info("Available sources: %s", sources if sources else "(none)")

    # --- Step 5: Delete traces ---
    if not args.skip_delete:
        logger.info("=" * 60)
        logger.info("Step 5: Delete uploaded traces")
        logger.info("=" * 60)

        for tid in uploaded_ids:
            deleted = client.traces.delete(tid)
            logger.info("  Deleted %s: %s", tid, deleted)
    else:
        logger.info("Skipping deletion (--skip-delete). Trace IDs: %s", ", ".join(uploaded_ids))

    logger.info("Sample complete.")


if __name__ == "__main__":
    main()
