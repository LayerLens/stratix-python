#!/usr/bin/env python
"""
Quality Gate v2 -- Stratix CI/CD, shipped as the Stratix Quality Gate Action
============================================================================

Differences from the samples/cicd v1 gate (all were acceptance-test defects):

1. Installs from the LayerLens SDK index, not the unrelated PyPI ``layerlens``
   project (v1 workflow line 36 did ``pip install layerlens``).
2. Reads a PINNED trace set (--trace-set) or a baseline diff (--baseline with
   --mode diff) instead of scoring "the newest traces, whatever they are"
   (v1 quality_gate.py line 114). Unpinned runs are still possible but are
   labeled ``unpinned`` in the report and in the PR comment.
3. Attributes the gate to the pull request: the report and PR comment carry
   the changed files (from GITHUB_*_PULL_REQUEST_* env vars) so reviewers see
   which diff the failed traces belong to.
4. Writes failed judges WITH their reasoning to a machine-readable report
   (``--report`` JSON) that the composite action posts as a PR comment. The v1
   workflow had no comment step at all.
5. The evaluation cap and the timeout are consistent: the action defaults
   max-evaluations=30 against a 20-minute timeout instead of 200 evaluations
   at 5-60 s each against a 15-minute timeout (v1 MAX_EVALUATIONS=200 at line
   67 vs timeout-minutes: 15 at workflow line 29).

Usage
-----

::

    export LAYERLENS_STRATIX_API_KEY=your-api-key
    python quality_gate.py --threshold 0.85 --trace-set traces.json --report report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import logging
from datetime import datetime, timezone

from layerlens import Stratix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "samples"))
from _helpers import poll_evaluation_results  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("layerlens.cicd.quality_gate")

# Defaults sized so that worst-case evaluation latency (5-60 s each) fits the
# action's default 20-minute timeout with head room.
DEFAULT_THRESHOLD = 0.85
DEFAULT_MAX_TRACES = 10
DEFAULT_MAX_EVALUATIONS = 30


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AI quality gate using the Stratix SDK (action edition).",
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--max-traces", type=int, default=DEFAULT_MAX_TRACES)
    parser.add_argument("--max-evaluations", type=int, default=DEFAULT_MAX_EVALUATIONS)
    parser.add_argument(
        "--trace-set",
        default=None,
        help="JSON file of pinned trace IDs (array, or {\"trace_ids\": [...]}) to evaluate.",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="JSON file mapping trace ID to last-known pass rate for regression diffing.",
    )
    parser.add_argument(
        "--mode",
        choices=["threshold", "diff"],
        default="threshold",
        help="threshold: absolute pass rate. diff: fail on regression vs baseline.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Write a machine-readable report JSON here (consumed by the PR-comment step).",
    )
    return parser


def load_trace_set(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        ids = data
    elif isinstance(data, dict):
        ids = data.get("trace_ids", [])
    else:
        raise ValueError(f"Unsupported trace-set format in {path}")
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        raise ValueError(f"trace-set must be a list of trace ID strings: {path}")
    return ids


def load_baseline(path: str) -> dict[str, float]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"baseline must map trace ID to pass rate: {path}")
    return {str(k): float(v) for k, v in data.items()}


def changed_files_from_env() -> list[str]:
    """Files changed by the pull request, from GitHub's default event env vars."""
    raw = os.environ.get("GITHUB_EVENT_PATH")
    if not raw or not os.path.exists(raw):
        return []
    try:
        with open(raw, encoding="utf-8") as f:
            event = json.load(f)
        files = [f["filename"] for f in event.get("pull_request", {}).get("files", [])]
        return files
    except Exception as exc:  # pragma: no cover - event payload is untrusted
        logger.warning("Could not read changed files from event payload: %s", exc)
        return []


def fetch_traces(client: Stratix, trace_set: list[str] | None, max_traces: int):
    """Resolve the trace list from a pinned set, or fall back to newest traces."""
    if trace_set:
        traces = []
        for tid in trace_set[:max_traces]:
            try:
                trace = client.traces.get(tid)
                if trace:
                    traces.append(trace)
            except Exception as exc:
                logger.warning("Pinned trace %s could not be fetched: %s", tid, exc)
        source = "pinned"
        if not traces:
            logger.error("None of the pinned traces could be fetched.")
            sys.exit(1)
        return traces, source

    logger.warning(
        "No --trace-set given: scoring the NEWEST traces. Run-to-run results "
        "are not comparable; pin a trace set for CI use."
    )
    resp = client.traces.get_many(page_size=max_traces)
    if not resp or not resp.traces:
        logger.error("No traces found -- nothing to evaluate.")
        sys.exit(1)
    return resp.traces, "unpinned"


def main() -> None:
    args = build_parser().parse_args()
    threshold: float = args.threshold
    mode: str = args.mode

    api_key = os.environ.get("STRATIX_API_KEY") or os.environ.get("LAYERLENS_STRATIX_API_KEY")
    if api_key:
        os.environ.setdefault("LAYERLENS_STRATIX_API_KEY", api_key)

    try:
        client = Stratix()
    except Exception as exc:
        logger.error("Failed to initialize Stratix client: %s", exc)
        sys.exit(1)

    logger.info("Connected to LayerLens (org=%s, project=%s)", client.organization_id, client.project_id)

    trace_ids_pinned = load_trace_set(args.trace_set) if args.trace_set else None
    traces, trace_source = fetch_traces(client, trace_ids_pinned, args.max_traces)
    logger.info("Trace source: %s (%d trace(s))", trace_source, len(traces))

    judges_resp = client.judges.get_many()
    if not judges_resp or not judges_resp.judges:
        logger.error("No judges configured -- create at least one judge first.")
        sys.exit(1)
    judges = judges_resp.judges
    logger.info("Found %d judge(s)", len(judges))

    # ------------------------------------------------------------------
    # Create evaluations, capped
    # ------------------------------------------------------------------
    eval_pairs: list[tuple[str, str, str]] = []  # (eval_id, trace_id, judge_name)
    for trace in traces:
        for judge in judges:
            if len(eval_pairs) >= args.max_evaluations:
                logger.info("Reached max-evaluations cap (%d). Stopping.", args.max_evaluations)
                break
            te = client.trace_evaluations.create(trace_id=trace.id, judge_id=judge.id)
            if te:
                eval_pairs.append((te.id, trace.id, judge.name))
            else:
                logger.warning("Failed to create evaluation (trace=%s, judge=%s)", trace.id, judge.id)
        if len(eval_pairs) >= args.max_evaluations:
            break

    if not eval_pairs:
        logger.error("No evaluations could be created.")
        sys.exit(1)

    logger.info("Created %d evaluation(s). Waiting for results...", len(eval_pairs))

    # ------------------------------------------------------------------
    # Poll and collect per-judge results
    # ------------------------------------------------------------------
    passed = 0
    failed = 0
    pending = 0
    failures: list[dict] = []
    pending_failures: list[dict] = []
    per_trace: dict[str, dict] = {}

    for eval_id, trace_id, judge_name in eval_pairs:
        results = poll_evaluation_results(client, eval_id)
        entry = per_trace.setdefault(trace_id, {"passed": 0, "failed": 0, "pending": 0})
        if results:
            for r in results:
                if r.passed:
                    passed += 1
                    entry["passed"] += 1
                else:
                    failed += 1
                    entry["failed"] += 1
                    failures.append(
                        {
                            "trace": trace_id,
                            "judge": judge_name,
                            "score": r.score,
                            "reasoning": getattr(r, "reasoning", None) or "(no reasoning returned)",
                        }
                    )
        else:
            pending += 1
            entry["pending"] += 1
            pending_failures.append({"trace": trace_id, "judge": judge_name})

    total = passed + failed
    pass_rate = passed / total if total > 0 else 0.0

    # ------------------------------------------------------------------
    # Gate decision
    # ------------------------------------------------------------------
    if mode == "diff":
        if not args.baseline:
            logger.error("--mode diff requires --baseline.")
            sys.exit(1)
        baseline = load_baseline(args.baseline)
        regressions = []
        for trace_id, entry in per_trace.items():
            base = baseline.get(trace_id)
            if base is None:
                continue
            rate = entry["passed"] / (entry["passed"] + entry["failed"]) if (entry["passed"] + entry["failed"]) else 0.0
            if rate < base:
                regressions.append({"trace": trace_id, "baseline": base, "now": rate})
        verdict = "pass" if not regressions else "fail"
        reason = (
            f"no regressions vs baseline ({len(per_trace)} matched traces)"
            if not regressions
            else f"{len(regressions)} trace(s) regressed below baseline"
        )
        if regressions:
            logger.error("Regression detected: %s", json.dumps(regressions, indent=2))
    else:
        verdict = "pass" if pass_rate >= threshold else "fail"
        reason = f"pass rate {pass_rate:.1%} vs threshold {threshold:.1%}"

    changed_files = changed_files_from_env()
    if changed_files:
        logger.info("PR changes under gate: %s", ", ".join(changed_files[:20]))

    report = {
        "verdict": verdict,
        "reason": reason,
        "mode": mode,
        "pass_rate": round(pass_rate, 4),
        "pass_rate_pct": round(pass_rate * 100, 1),
        "threshold": threshold,
        "threshold_pct": round(threshold * 100, 1),
        "trace_source": trace_source,
        "traces_evaluated": len(traces),
        "evaluations_completed": total,
        "evaluations_pending": pending,
        "pending": pending,
        "pending_failures": pending_failures,
        "failures": failures,
        "changed_files": changed_files,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    print()
    print("=" * 60)
    print("  Stratix Quality Gate Report")
    print("=" * 60)
    print(f"  Trace source     : {trace_source} ({len(traces)} traces)")
    print(f"  Changed files    : {len(changed_files)}")
    print(f"  Total results    : {total} ({pending} pending)")
    print(f"  Pass rate        : {pass_rate:.1%}")
    if mode == "diff":
        print(f"  Baseline         : {args.baseline}")
    print(f"  Threshold        : {threshold:.1%}" if mode == "threshold" else f"  Reason           : {reason}")
    if failures:
        print("-" * 60)
        print("  Failed evaluations:")
        for f in failures:
            print(f"    [{f['judge']}] trace={f['trace'][:12]}... score={f['score']}")
    print("-" * 60)
    print(f"  RESULT: {verdict.upper()} ({reason})")
    print("=" * 60)

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("Report written to %s", args.report)

    if verdict == "fail":
        sys.exit(1)


if __name__ == "__main__":
    main()
