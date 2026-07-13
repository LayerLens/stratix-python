#!/usr/bin/env python3
"""Co-Work: Automated Code Review -- LayerLens Python SDK Sample.

Demonstrates a multi-agent code review pattern where an Instrumentor agent
generates code trace data (simulated code snippets paired with their outputs)
and uploads them, while a Reviewer agent evaluates each trace across multiple
quality dimensions using LayerLens judge types.

Prerequisites:
    pip install layerlens --index-url https://sdk.layerlens.ai/package
    export LAYERLENS_STRATIX_API_KEY=your-api-key

Usage:
    python code_review.py
"""

from __future__ import annotations

import os
import sys
from typing import Any

from layerlens import Stratix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _helpers import (
    create_judge,
    poll_evaluation_results,
    recorded_trace_path,
    upload_recorded_trace,
)

# This sample uploads RECORDED REAL traces: each was captured from a genuine
# instrumented ``code-review-agent`` run over the code snippets below (see
# ``samples/data/_generate_fixtures.py``), so the LayerLens UI renders the
# Agent, Framework, and Status columns from real data. The snippets remain
# here as documentation of what was reviewed and to label the evaluation output.
SAMPLE = "code_review"
FIXTURE = recorded_trace_path("cowork", "code_review.jsonl")

# ---------------------------------------------------------------------------
# Code snippets and their outputs (labels for the recorded traces above)
# ---------------------------------------------------------------------------

CODE_SAMPLES: list[dict[str, Any]] = [
    {
        "description": "SQL query builder",
        "input": (
            "Write a Python function that builds a SQL query from user input to search a products table by name."
        ),
        "output": (
            "def search_products(user_input: str) -> str:\n"
            "    query = f\"SELECT * FROM products WHERE name LIKE '%{user_input}%'\"\n"
            "    return query\n"
        ),
        "language": "python",
    },
    {
        "description": "Password hashing utility",
        "input": (
            "Write a function that hashes a password for storage using a secure algorithm."
        ),
        "output": (
            "import hashlib\n"
            "import secrets\n\n"
            "def hash_password(password: str) -> tuple[str, str]:\n"
            "    salt = secrets.token_hex(16)\n"
            "    hashed = hashlib.pbkdf2_hmac(\n"
            '        "sha256", password.encode(), salt.encode(), 100_000\n'
            "    )\n"
            "    return hashed.hex(), salt\n"
        ),
        "language": "python",
    },
    {
        "description": "REST API endpoint",
        "input": (
            "Write a FastAPI endpoint that returns a user profile by ID, including proper error handling."
        ),
        "output": (
            "from fastapi import FastAPI, HTTPException\n\n"
            "app = FastAPI()\n\n"
            "@app.get('/users/{user_id}')\n"
            "async def get_user(user_id: int):\n"
            "    user = await db.get_user(user_id)\n"
            "    if not user:\n"
            "        raise HTTPException(status_code=404, detail='User not found')\n"
            "    return {'id': user.id, 'name': user.name, 'email': user.email}\n"
        ),
        "language": "python",
    },
    {
        "description": "File reader with error handling",
        "input": (
            "Write a function that reads a JSON configuration file and "
            "returns the parsed contents with proper error handling."
        ),
        "output": (
            "import json\n\n"
            "def read_config(path: str) -> dict:\n"
            "    try:\n"
            "        with open(path) as f:\n"
            "            return json.load(f)\n"
            "    except FileNotFoundError:\n"
            "        raise SystemExit(f'Config file not found: {path}')\n"
            "    except json.JSONDecodeError as e:\n"
            "        raise SystemExit(f'Invalid JSON in {path}: {e}')\n"
        ),
        "language": "python",
    },
]


def main() -> None:
    """Run the automated code review Co-Work Channel demo."""
    print("=== LayerLens Co-Work: Automated Code Review ===\n")

    try:
        client = Stratix()
    except Exception as exc:
        print(f"ERROR: Failed to initialize LayerLens client: {exc}")
        sys.exit(1)

    # Upload the recorded real traces first. Doing this before judge creation
    # means the traces always land even if the org has no evaluation model yet.
    print(f"[Instrumentor] Uploading {len(CODE_SAMPLES)} recorded code traces...\n")
    trace_ids = upload_recorded_trace(client, FIXTURE)
    if not trace_ids:
        print("ERROR: no traces uploaded (fixture missing or rejected).")
        sys.exit(1)

    # Create judges. If the org has no models available, judge creation raises
    # RuntimeError -- we skip the evaluations (the traces are already uploaded)
    # rather than crash.
    judge_ids: list[str] = []
    try:
        review_judges = {
            "Execution": create_judge(
                client,
                name="Code Execution Judge",
                evaluation_goal="Evaluate whether the generated code would execute correctly without errors.",
                namespace=SAMPLE,
            ),
            "Security": create_judge(
                client,
                name="Code Security Judge",
                evaluation_goal="Evaluate whether the generated code follows security best practices and is free from vulnerabilities.",
                namespace=SAMPLE,
            ),
            "Metrics": create_judge(
                client,
                name="Code Metrics Judge",
                evaluation_goal="Evaluate the code quality metrics including readability, maintainability, and adherence to best practices.",
                namespace=SAMPLE,
            ),
        }
        judge_ids = [j.id for j in review_judges.values()]

        # ------------------------------------------------------------------
        # Phase 1 -- Instrumentor agent: map recorded traces to snippets
        # ------------------------------------------------------------------
        trace_map: list[dict[str, Any]] = []

        for sample, tid in zip(CODE_SAMPLES, trace_ids):
            trace_map.append(
                {
                    "trace_id": tid,
                    "description": sample["description"],
                }
            )
            print(f'[Instrumentor] Snippet "{sample["description"]}" -> trace {tid}')

        # ------------------------------------------------------------------
        # Phase 2 -- Reviewer agent: evaluate each trace
        # ------------------------------------------------------------------
        print(f"\n[Reviewer] Evaluating {len(trace_map)} code traces...\n")

        review_results: list[dict[str, Any]] = []

        for entry in trace_map:
            tid = entry["trace_id"]
            desc = entry["description"]
            print(f'[Reviewer] Reviewing: "{desc}" (trace {tid})')

            scores: dict[str, float] = {}
            for label, judge_obj in review_judges.items():
                evaluation = client.trace_evaluations.create(
                    trace_id=tid,
                    judge_id=judge_obj.id,
                )
                results = poll_evaluation_results(client, evaluation.id)
                score = 0.0
                if results:
                    score = results[0].score
                scores[label] = score
                status = "PASS" if score >= 0.7 else "WARN" if score >= 0.4 else "FAIL"
                print(f"[Reviewer]   {label:12s} {status} ({score:.2f})")

            avg_score = sum(scores.values()) / len(scores) if scores else 0.0
            review_results.append(
                {
                    "trace_id": tid,
                    "description": desc,
                    "scores": scores,
                    "average": avg_score,
                }
            )
            print()

        # ------------------------------------------------------------------
        # Phase 3 -- Summary report
        # ------------------------------------------------------------------
        print("=" * 64)
        print("[ReviewReport] Code Review Summary")
        print("=" * 64)

        total_avg = 0.0
        for result in review_results:
            quality = (
                "HIGH"
                if result["average"] >= 0.7
                else "MEDIUM" if result["average"] >= 0.4 else "LOW"
            )
            print(
                f"  {result['description']:30s}  avg={result['average']:.2f}  quality={quality}"
            )
            total_avg += result["average"]

        overall = total_avg / len(review_results) if review_results else 0.0
        print(f"\n  Overall quality score: {overall:.2f}")
        print(f"  Traces reviewed: {len(review_results)}")
        print(f"  Judge dimensions: {', '.join(review_judges.keys())}")

        high_count = sum(1 for r in review_results if r["average"] >= 0.7)
        print(f"  High quality: {high_count}/{len(review_results)}")
        print("  All evaluations stored in LayerLens.")

    except RuntimeError as exc:
        print(f"\nNOTE: evaluations skipped -- {exc}")
        print("  Traces are uploaded; add a project/public model to enable judges.")
    finally:
        for jid in judge_ids:
            try:
                client.judges.delete(jid)
            except Exception:
                pass


if __name__ == "__main__":
    main()
