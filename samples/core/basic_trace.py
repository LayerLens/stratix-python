"""Capture LLM traces with STRATIX decorator and context-manager patterns.

Demonstrates:
- Initializing the STRATIX instrumentation SDK
- Using @stratix.trace_tool to auto-capture tool I/O and latency
- Using stratix.context() as a context manager for scoped tracing
- Calling OpenAI GPT and recording the model invocation event
- Uploading the trace file via the LayerLens Stratix API client
- Polling to confirm the trace was ingested

Requires:
    LAYERLENS_STRATIX_API_KEY  - LayerLens platform API key
    OPENAI_API_KEY             - OpenAI API key
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

from openai import OpenAI

from layerlens import Stratix
from layerlens.instrument import STRATIX, emit_input, emit_output, emit_model_invoke


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        print(f"ERROR: environment variable {name} is not set.", file=sys.stderr)
        sys.exit(1)
    return val


def build_stratix(policy_ref: str, agent_id: str) -> STRATIX:
    """One-liner STRATIX initialization."""
    return STRATIX(
        policy_ref=policy_ref,
        agent_id=agent_id,
        framework="openai",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture an LLM trace with STRATIX decorators and context managers."
    )
    parser.add_argument(
        "--prompt",
        default="Explain gradient descent in two sentences.",
        help="Prompt to send to the LLM (default: gradient descent explanation).",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--policy-ref",
        default="stratix-policy-cs-v1@1.0.0",
        help="STRATIX policy reference.",
    )
    parser.add_argument(
        "--agent-id",
        default="sample_trace_agent",
        help="Agent identifier for the trace.",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip uploading the trace to the platform.",
    )
    args = parser.parse_args()

    openai_key = _require_env("OPENAI_API_KEY")

    # -- 1. Initialize STRATIX instrumentation --
    stratix = build_stratix(args.policy_ref, args.agent_id)
    print(f"[STRATIX] Initialized  policy={args.policy_ref}  agent={args.agent_id}")

    # -- 2. Define a traced tool --
    @stratix.trace_tool(name="format_response", version="1.0.0")
    def format_response(raw: str) -> dict:
        """Trim and wrap the LLM output."""
        return {"answer": raw.strip(), "length": len(raw.strip())}

    # -- 3. Run a trial inside a context scope --
    ctx = stratix.start_trial()
    print(f"[STRATIX] Trial started  trace_id={ctx.trace_id}")

    with stratix.context():
        # Emit user input
        emit_input(args.prompt, role="human")
        print(f"[trace]   Input recorded: {args.prompt[:60]}...")

        # Call OpenAI
        oai = OpenAI(api_key=openai_key)
        t0 = time.perf_counter()
        response = oai.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": args.prompt}],
            max_tokens=256,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        answer = response.choices[0].message.content or ""
        usage = response.usage

        # Emit model invocation event
        emit_model_invoke(
            provider="openai",
            name=args.model,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            total_tokens=usage.total_tokens if usage else None,
            latency_ms=latency_ms,
        )
        print(f"[trace]   Model invoked: {args.model}  latency={latency_ms:.0f}ms")

        # Use the traced tool
        result = format_response(answer)
        print(f"[trace]   Tool output: {json.dumps(result)[:80]}...")

        # Emit agent output
        emit_output(result["answer"])

    # -- 4. End trial and inspect attestation --
    summary = stratix.end_trial()
    events = stratix.get_events()
    print(f"[STRATIX] Trial ended   status={summary.get('status')}  events={len(events)}")

    if args.skip_upload:
        print("[skip]    Upload skipped (--skip-upload)")
        return

    # -- 5. Upload trace via API client --
    ll_key = _require_env("LAYERLENS_STRATIX_API_KEY")
    client = Stratix(api_key=ll_key)

    # Serialize events to a temp JSONL file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for ev in events:
            f.write(json.dumps(ev.to_dict()) + "\n")
        trace_path = f.name

    try:
        print(f"[upload]  Uploading {len(events)} events from {trace_path} ...")
        resp = client.traces.upload(trace_path)
        if resp and resp.trace_ids:
            print(f"[upload]  Success! trace_ids={resp.trace_ids}")

            # Poll to confirm ingestion
            for tid in resp.trace_ids[:1]:
                trace = client.traces.get(tid)
                if trace:
                    print(f"[verify]  Trace {tid} confirmed on platform.")
                else:
                    print(f"[verify]  Trace {tid} not yet visible (eventual consistency).")
        else:
            print("[upload]  Upload returned no trace IDs. Check API logs.")
    except Exception as exc:
        print(f"[upload]  Upload failed: {exc}", file=sys.stderr)
    finally:
        os.unlink(trace_path)


if __name__ == "__main__":
    main()
