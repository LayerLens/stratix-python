#!/usr/bin/env python3
"""
LayerLens Evaluation Script for OpenClaw
=========================================
Called by the LayerLens OpenClaw skill to upload a trace and run an
evaluation. Accepts input via command-line arguments or JSON on stdin.
Prints results as JSON for OpenClaw to consume.

Usage:
    # Via arguments:
    python evaluate.py --input "prompt" --output "response" --goal "accuracy"

    # Via stdin (JSON):
    echo '{"input": "prompt", "output": "response", "goal": "accuracy"}' | python evaluate.py

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from typing import Any, Optional

try:
    from layerlens import Stratix
except ImportError:
    print(json.dumps({
        "error": "layerlens package not installed. Run: pip install layerlens --index-url https://sdk.layerlens.ai/package",
        "success": False,
    }))
    sys.exit(1)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from _helpers import create_judge, poll_evaluation_results


def _parse_stdin() -> Optional[dict[str, Any]]:
    """Try to read JSON input from stdin if available."""
    if sys.stdin.isatty():
        return None
    try:
        raw = sys.stdin.read().strip()
        if raw:
            return json.loads(raw)
    except (json.JSONDecodeError, IOError):
        pass
    return None


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Upload a trace and evaluate it with LayerLens.",
    )
    parser.add_argument(
        "--input", "-i",
        dest="input_text",
        help="The input/prompt text for the trace.",
    )
    parser.add_argument(
        "--output", "-o",
        dest="output_text",
        help="The output/response text for the trace.",
    )
    parser.add_argument(
        "--goal", "-g",
        dest="evaluation_goal",
        default="Evaluate whether the response is accurate, helpful, and safe.",
        help="The evaluation goal for the judge.",
    )
    parser.add_argument(
        "--judge-name",
        default="OpenClaw Skill Judge",
        help="Name for the evaluation judge.",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="JSON string of additional metadata to attach to the trace.",
    )
    return parser.parse_args()


def _upload_trace(
    client: Stratix,
    input_text: str,
    output_text: str,
    metadata: Optional[dict[str, Any]] = None,
) -> str:
    """Upload a single trace and return its ID."""
    trace_data: dict[str, Any] = {
        "input": [{"role": "user", "content": input_text}],
        "output": output_text,
    }
    if metadata:
        trace_data["metadata"] = metadata

    fd, path = tempfile.mkstemp(suffix=".jsonl")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(trace_data) + "\n")
        result = client.traces.upload(path)
    finally:
        if os.path.exists(path):
            os.unlink(path)

    if not result or not result.trace_ids:
        raise RuntimeError("Trace upload returned no trace IDs")
    return result.trace_ids[0]


def _poll_results(
    client: Stratix,
    evaluation_id: str,
    max_attempts: int = 30,
) -> Optional[dict[str, Any]]:
    """Poll for evaluation results using the shared helper."""
    results = poll_evaluation_results(client, evaluation_id, max_attempts=max_attempts)
    if results:
        r = results[0]
        return {
            "score": r.score,
            "passed": r.passed,
            "reasoning": r.reasoning,
        }
    return None


def main() -> None:
    """Run evaluation and print results as JSON."""
    # Resolve input from stdin or args
    stdin_data = _parse_stdin()
    args = _parse_args()

    input_text = (stdin_data or {}).get("input") or args.input_text
    output_text = (stdin_data or {}).get("output") or args.output_text
    evaluation_goal = (stdin_data or {}).get("goal") or args.evaluation_goal
    judge_name = (stdin_data or {}).get("judge_name") or args.judge_name

    extra_metadata = None
    metadata_raw = (stdin_data or {}).get("metadata") or args.metadata
    if isinstance(metadata_raw, str):
        try:
            extra_metadata = json.loads(metadata_raw)
        except json.JSONDecodeError:
            pass
    elif isinstance(metadata_raw, dict):
        extra_metadata = metadata_raw

    if not input_text or not output_text:
        print(json.dumps({
            "error": "Both --input and --output are required.",
            "success": False,
        }))
        sys.exit(1)

    # Initialize client
    try:
        client = Stratix()
    except Exception as exc:
        print(json.dumps({
            "error": f"Failed to initialize LayerLens client: {exc}",
            "success": False,
        }))
        sys.exit(1)

    # Upload trace
    try:
        metadata = {"source": "openclaw-skill"}
        if extra_metadata:
            metadata.update(extra_metadata)
        trace_id = _upload_trace(client, input_text, output_text, metadata)
    except Exception as exc:
        print(json.dumps({
            "error": f"Failed to upload trace: {exc}",
            "success": False,
        }))
        sys.exit(1)

    # Create judge
    try:
        judge = create_judge(
            client,
            name=judge_name,
            evaluation_goal=evaluation_goal,
        )
    except Exception as exc:
        print(json.dumps({
            "error": f"Failed to create judge: {exc}",
            "trace_id": trace_id,
            "success": False,
        }))
        sys.exit(1)

    # Run evaluation
    try:
        evaluation = client.trace_evaluations.create(
            trace_id=trace_id,
            judge_id=judge.id,
        )
    except Exception as exc:
        print(json.dumps({
            "error": f"Failed to create evaluation: {exc}",
            "trace_id": trace_id,
            "judge_id": judge.id,
            "success": False,
        }))
        sys.exit(1)

    # Poll for results
    result = _poll_results(client, evaluation.id)
    if result:
        output = {
            "success": True,
            "trace_id": trace_id,
            "judge_id": judge.id,
            "evaluation_id": evaluation.id,
            "score": result["score"],
            "passed": result["passed"],
            "reasoning": result["reasoning"],
        }
    else:
        output = {
            "success": False,
            "status": "pending",
            "trace_id": trace_id,
            "judge_id": judge.id,
            "evaluation_id": evaluation.id,
            "score": None,
            "passed": None,
            "reasoning": "Evaluation still processing. Poll again later.",
        }

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
