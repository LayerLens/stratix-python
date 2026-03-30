#!/usr/bin/env python
"""
Trace Investigation -- LayerLens Python SDK Sample
===================================================

Port of ateam's "Demo 01 -- Trace Investigation".  Replaces all raw
urllib HTTP calls with SDK calls to demonstrate investigating production
traces for latency and error issues.

Workflow
--------
  1. List recent traces sorted by creation time.
  2. Filter the results for high-latency traces.
  3. Inspect a specific trace in detail.
  4. Print an investigation summary.

Prerequisites
-------------
* ``pip install layerlens --index-url https://sdk.layerlens.ai/package``
* Set ``LAYERLENS_STRATIX_API_KEY`` environment variable
* At least a few traces already present in the project (upload some
  with ``basic_trace.py`` if needed).

Usage
-----
::

    export LAYERLENS_STRATIX_API_KEY=your-api-key
    python trace_investigation.py
    python trace_investigation.py --page-size 25 --latency-threshold 2.0
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from layerlens import Stratix

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("layerlens.samples.trace_investigation")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Investigate production traces with the LayerLens Python SDK.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=20,
        help="Number of traces to fetch per page (default: 20).",
    )
    parser.add_argument(
        "--latency-threshold",
        type=float,
        default=2.0,
        help="Seconds above which a trace is considered high-latency (default: 2.0).",
    )
    parser.add_argument(
        "--trace-id",
        default="",
        help="Specific trace ID to inspect. If omitted, the first trace from the listing is used.",
    )
    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_latency(trace) -> float | None:
    """Attempt to extract a latency value (in seconds) from trace metadata."""
    # The trace data structure may carry latency in several places depending
    # on the ingestion source.  Try common locations.
    for attr in ("latency", "duration", "elapsed"):
        val = getattr(trace, attr, None)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass

    # Check inside nested data / metadata dicts
    data = getattr(trace, "data", None) or {}
    if isinstance(data, dict):
        metadata = data.get("metadata", data)
        if isinstance(metadata, dict):
            for key in ("latency", "duration", "elapsed", "response_time"):
                val = metadata.get(key)
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        pass
    return None


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
    # Step 1: List recent traces
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 1: List recent traces")
    logger.info("=" * 60)

    response = client.traces.get_many(
        page_size=args.page_size,
        sort_by="created_at",
        sort_order="desc",
    )

    if not response or not response.traces:
        logger.error(
            "No traces found. Upload some traces first (see basic_trace.py)."
        )
        sys.exit(1)

    logger.info(
        "Fetched %d trace(s) (total available: %d)",
        response.count,
        response.total_count,
    )
    for trace in response.traces[:5]:
        logger.info(
            "  - id=%s  created=%s  source=%s",
            trace.id,
            getattr(trace, "created_at", "N/A"),
            getattr(trace, "source", "N/A"),
        )

    # ------------------------------------------------------------------
    # Step 2: Filter for high-latency traces
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info(
        "Step 2: Filter for high-latency traces (threshold=%.1fs)",
        args.latency_threshold,
    )
    logger.info("=" * 60)

    high_latency_traces = []
    for trace in response.traces:
        latency = extract_latency(trace)
        if latency is not None and latency > args.latency_threshold:
            high_latency_traces.append((trace, latency))

    if high_latency_traces:
        logger.info(
            "Found %d high-latency trace(s) out of %d:",
            len(high_latency_traces),
            response.count,
        )
        for trace, latency in high_latency_traces[:10]:
            logger.info("  - id=%s  latency=%.2fs", trace.id, latency)
    else:
        logger.info(
            "No traces exceeded the %.1fs latency threshold "
            "(latency metadata may not be present on all traces).",
            args.latency_threshold,
        )

    # ------------------------------------------------------------------
    # Step 3: Inspect a specific trace
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 3: Inspect a specific trace")
    logger.info("=" * 60)

    if args.trace_id:
        inspect_id = args.trace_id
    elif high_latency_traces:
        inspect_id = high_latency_traces[0][0].id
        logger.info("Using highest-latency trace for inspection.")
    else:
        inspect_id = response.traces[0].id
        logger.info("Using most recent trace for inspection.")

    trace_detail = client.traces.get(inspect_id)
    if not trace_detail:
        logger.error("Could not retrieve trace %s", inspect_id)
        sys.exit(1)

    logger.info("Trace detail for %s:", trace_detail.id)
    logger.info("  Created:  %s", getattr(trace_detail, "created_at", "N/A"))
    logger.info("  Source:   %s", getattr(trace_detail, "source", "N/A"))
    logger.info("  Filename: %s", getattr(trace_detail, "filename", "N/A"))

    data = getattr(trace_detail, "data", None)
    if data and isinstance(data, dict):
        logger.info("  Data keys: %s", list(data.keys()))
        metadata = data.get("metadata")
        if metadata and isinstance(metadata, dict):
            logger.info("  Metadata: %s", metadata)
        input_data = data.get("input")
        if input_data:
            preview = str(input_data)[:120]
            logger.info("  Input preview: %s...", preview)
        output_data = data.get("output")
        if output_data:
            preview = str(output_data)[:120]
            logger.info("  Output preview: %s...", preview)
    else:
        logger.info("  (no structured data available)")

    # ------------------------------------------------------------------
    # Step 4: Investigation summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Step 4: Investigation summary")
    logger.info("=" * 60)

    print()
    print("=" * 70)
    print("  TRACE INVESTIGATION SUMMARY")
    print("=" * 70)
    print(f"  Total traces examined:   {response.count}")
    print(f"  Total traces in project: {response.total_count}")
    print(f"  Latency threshold:       {args.latency_threshold:.1f}s")
    print(f"  High-latency traces:     {len(high_latency_traces)}")
    print(f"  Inspected trace ID:      {inspect_id}")
    if high_latency_traces:
        avg_latency = sum(l for _, l in high_latency_traces) / len(high_latency_traces)
        max_latency = max(l for _, l in high_latency_traces)
        print(f"  Avg high-latency:        {avg_latency:.2f}s")
        print(f"  Max latency observed:    {max_latency:.2f}s")
    print("=" * 70)
    print()

    logger.info("Investigation complete.")


if __name__ == "__main__":
    main()
