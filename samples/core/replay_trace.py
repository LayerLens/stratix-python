"""Replay a trace with a model swap and compare original vs. replay.

Demonstrates:
- Fetching an existing trace by ID
- Triggering a replay with a different model (model override)
- Showing a side-by-side comparison of original and replayed outputs

This uses the LayerLens Stratix API client for trace retrieval. The replay
API triggers server-side re-execution of the trace with the specified model.

Requires:
    LAYERLENS_STRATIX_API_KEY  - LayerLens platform API key
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import json
import textwrap


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        print(f"ERROR: environment variable {name} is not set.", file=sys.stderr)
        sys.exit(1)
    return val


def fetch_trace(client, trace_id: str):
    """Fetch a trace and display its summary."""
    print(f"[fetch]   Retrieving trace {trace_id}...")
    trace = client.traces.get(trace_id)
    if trace is None:
        print(f"ERROR: Trace {trace_id} not found.", file=sys.stderr)
        sys.exit(1)
    print(f"[fetch]   Trace found.")
    _print_trace_summary(trace, label="Original")
    return trace


def _print_trace_summary(trace, label: str = "Trace") -> None:
    """Print a compact trace summary."""
    sep = "-" * 72
    print(f"\n{sep}")
    print(f"  {label} Trace Summary")
    print(sep)
    print(f"  ID        : {trace.id}")
    if hasattr(trace, "source") and trace.source:
        print(f"  Source    : {trace.source}")
    if hasattr(trace, "model_name") and trace.model_name:
        print(f"  Model     : {trace.model_name}")
    if hasattr(trace, "status") and trace.status:
        print(f"  Status    : {trace.status}")
    if hasattr(trace, "created_at") and trace.created_at:
        print(f"  Created   : {trace.created_at}")
    if hasattr(trace, "input") and trace.input:
        wrapped = textwrap.shorten(str(trace.input), width=60, placeholder="...")
        print(f"  Input     : {wrapped}")
    if hasattr(trace, "output") and trace.output:
        wrapped = textwrap.shorten(str(trace.output), width=60, placeholder="...")
        print(f"  Output    : {wrapped}")
    print(sep)


def trigger_replay(client, trace_id: str, model_override: str):
    """Trigger a trace replay with a model swap.

    The replay API may not be available yet on all platform versions.
    This function uses a try/except pattern so the sample remains
    forward-compatible.
    """
    print(f"[replay]  Triggering replay of {trace_id} with model={model_override}...")

    # The traces resource may expose a replay method in the future.
    # For now, we attempt a POST to the replay endpoint directly.
    try:
        base = f"/organizations/{client.organization_id}/projects/{client.project_id}/traces"
        resp = client._post(
            f"{base}/{trace_id}/replay",
            body={"model_override": model_override},
            timeout=60,
            cast_to=dict,
        )
        if isinstance(resp, dict) and "data" in resp and "status" in resp:
            resp = resp["data"]
        return resp
    except Exception as exc:
        print(f"[replay]  Replay API call failed: {exc}")
        print("[replay]  The replay endpoint may not be available on this platform version.")
        print("[replay]  Falling back to a local comparison workflow.")
        return None


def poll_replay(client, replay_data: dict, timeout_sec: int = 300) -> dict | None:
    """Poll a replay job until completion."""
    replay_id = replay_data.get("replay_id") or replay_data.get("id")
    if not replay_id:
        print("[poll]    No replay_id in response; replay may have completed synchronously.")
        return replay_data

    print(f"[poll]    Polling replay {replay_id}...")
    start = time.time()
    interval = 5
    while time.time() - start < timeout_sec:
        try:
            base = f"/organizations/{client.organization_id}/projects/{client.project_id}/traces"
            resp = client._get(
                f"{base}/replays/{replay_id}",
                timeout=30,
                cast_to=dict,
            )
            if isinstance(resp, dict) and "data" in resp and "status" in resp:
                resp = resp["data"]
            status = resp.get("status", "unknown") if isinstance(resp, dict) else "unknown"
            print(f"[poll]    status={status}  elapsed={time.time() - start:.0f}s")
            if status in ("completed", "success", "failed", "error"):
                return resp
        except Exception:
            pass
        time.sleep(interval)
        interval = min(interval * 2, 30)

    print("[poll]    Timeout waiting for replay.", file=sys.stderr)
    return None


def show_diff(original_trace, replay_result: dict | None) -> None:
    """Show a side-by-side diff of original vs. replayed outputs."""
    sep = "=" * 72
    print(f"\n{sep}")
    print("  SIDE-BY-SIDE COMPARISON")
    print(sep)

    orig_output = ""
    if hasattr(original_trace, "output") and original_trace.output:
        orig_output = str(original_trace.output)

    replay_output = ""
    if replay_result and isinstance(replay_result, dict):
        replay_output = str(
            replay_result.get("output", replay_result.get("result", "(no output)"))
        )

    col_width = 34
    print(f"  {'ORIGINAL':<{col_width}}  {'REPLAY':<{col_width}}")
    print(f"  {'-' * col_width}  {'-' * col_width}")

    orig_lines = textwrap.wrap(orig_output or "(empty)", width=col_width) or ["(empty)"]
    replay_lines = textwrap.wrap(replay_output or "(empty)", width=col_width) or ["(empty)"]
    max_lines = max(len(orig_lines), len(replay_lines))

    for i in range(max_lines):
        left = orig_lines[i] if i < len(orig_lines) else ""
        right = replay_lines[i] if i < len(replay_lines) else ""
        marker = " " if left == right else "*"
        print(f" {marker}{left:<{col_width}}  {right:<{col_width}}")

    print(sep)
    if orig_output == replay_output:
        print("  Outputs are identical.")
    else:
        print("  Outputs differ (lines marked with * are different).")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a trace with a model swap and compare outputs."
    )
    parser.add_argument(
        "trace_id",
        help="ID of the trace to replay.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name to use for the replay (e.g. 'gpt-4o', 'claude-3-opus').",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Max seconds to wait for replay completion (default: 300).",
    )
    args = parser.parse_args()

    from layerlens import Stratix

    api_key = _require_env("LAYERLENS_STRATIX_API_KEY")
    client = Stratix(api_key=api_key)
    print(f"[init]    Connected to LayerLens (org={client.organization_id})")

    # Fetch original trace
    original = fetch_trace(client, args.trace_id)

    # Trigger replay with model swap
    replay_data = trigger_replay(client, args.trace_id, args.model)

    replay_result = None
    if replay_data is not None:
        replay_result = poll_replay(client, replay_data, timeout_sec=args.timeout)

    # Compare
    show_diff(original, replay_result)


if __name__ == "__main__":
    main()
