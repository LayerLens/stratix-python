"""AG-UI (CopilotKit Agent-User Interface) streaming protocol demo.

Demonstrates:
- Run lifecycle events (RUN_STARTED, TEXT_MESSAGE_*, TOOL_CALL_*, RUN_FINISHED)
- Frontend state management via STATE_SNAPSHOT events
- User feedback capture and full SSE stream instrumented by STRATIX

If ``ag_ui`` is not installed the protocol is simulated via STRATIX emit functions.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid

from layerlens.instrument import (
    STRATIX,
    emit,
    emit_input,
    emit_output,
    emit_tool_call,
)

# ---------------------------------------------------------------------------
# Try importing the real AG-UI SDK; fall back to simulation
# ---------------------------------------------------------------------------

try:
    from ag_ui.core import RunStarted, TextMessageStart, RunFinished  # type: ignore[import-untyped]

    HAS_AGUI_SDK = True
except ImportError:
    HAS_AGUI_SDK = False
    print(
        "[ag-ui] SDK not installed. Run: pip install ag-ui-protocol\n"
        "        Continuing with simulated streaming flow.\n"
    )


# ---------------------------------------------------------------------------
# Simulated AG-UI event stream
# ---------------------------------------------------------------------------

def _simulated_agui_stream(run_id: str) -> list[dict]:
    """Build a realistic AG-UI SSE event sequence."""
    msg_id = str(uuid.uuid4())
    return [
        {"type": "RUN_STARTED", "threadId": "t-1", "runId": run_id},
        {"type": "TEXT_MESSAGE_START", "messageId": msg_id, "role": "assistant"},
        {"type": "TEXT_MESSAGE_CONTENT", "messageId": msg_id, "delta": "Based on the data, "},
        {"type": "TEXT_MESSAGE_CONTENT", "messageId": msg_id, "delta": "the optimal strategy is to increase allocation by 15%."},
        {"type": "TEXT_MESSAGE_END", "messageId": msg_id},
        {"type": "TOOL_CALL_START", "toolCallId": "tc-1", "toolCallName": "update_portfolio"},
        {"type": "TOOL_CALL_ARGS", "toolCallId": "tc-1", "delta": json.dumps({"allocation_pct": 15})},
        {"type": "TOOL_CALL_END", "toolCallId": "tc-1"},
        {"type": "STATE_SNAPSHOT", "snapshot": {"portfolio_updated": True, "allocation_pct": 15}},
        {"type": "RUN_FINISHED", "runId": run_id},
    ]


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="AG-UI streaming protocol instrumentation demo")
    parser.add_argument("--policy-ref", default="stratix-policy-cs-v1@1.0.0")
    parser.add_argument("--agent-id", default="copilot_agent")
    parser.add_argument("--user-message", default="Rebalance my portfolio for growth.")
    args = parser.parse_args()

    stratix = STRATIX(policy_ref=args.policy_ref, agent_id=args.agent_id, framework="agui")
    ctx = stratix.start_trial()
    print(f"[STRATIX] Trial started  trace_id={ctx.trace_id}")

    run_id = str(uuid.uuid4())
    text_chunks: list[str] = []
    tool_args_buffer: str = ""

    with stratix.context():
        emit_input(args.user_message, role="human")
        print(f"[ag-ui] User: {args.user_message}")

        # -- Process simulated SSE stream --
        for event in _simulated_agui_stream(run_id):
            etype = event["type"]
            t0 = time.perf_counter()

            if etype == "RUN_STARTED":
                emit_tool_call(
                    name="agui.lifecycle/run_started",
                    input_data={"runId": event["runId"], "threadId": event["threadId"]},
                    output_data={"status": "started"},
                    integration="service",
                )
                print(f"[ag-ui] RUN_STARTED  run={run_id[:8]}...")

            elif etype == "TEXT_MESSAGE_CONTENT":
                text_chunks.append(event["delta"])
                # Stream deltas are not emitted individually to avoid noise

            elif etype == "TEXT_MESSAGE_END":
                full_text = "".join(text_chunks)
                emit_tool_call(
                    name="agui.stream/text_message",
                    input_data={"messageId": event["messageId"]},
                    output_data={"text": full_text, "chunks": len(text_chunks)},
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    integration="service",
                )
                print(f"[ag-ui] TEXT_MESSAGE  ({len(text_chunks)} chunks): {full_text[:60]}...")

            elif etype == "TOOL_CALL_START":
                tool_args_buffer = ""
                print(f"[ag-ui] TOOL_CALL_START  tool={event['toolCallName']}")

            elif etype == "TOOL_CALL_ARGS":
                tool_args_buffer += event["delta"]

            elif etype == "TOOL_CALL_END":
                parsed_args = json.loads(tool_args_buffer) if tool_args_buffer else {}
                emit_tool_call(
                    name="update_portfolio",
                    input_data=parsed_args,
                    output_data={"applied": True},
                    latency_ms=35.0,
                    integration="service",
                )
                print(f"[ag-ui] TOOL_CALL_END  args={tool_args_buffer}")

            elif etype == "STATE_SNAPSHOT":
                emit_tool_call(
                    name="agui.state/snapshot",
                    input_data={},
                    output_data=event["snapshot"],
                    integration="service",
                )
                print(f"[ag-ui] STATE_SNAPSHOT  {json.dumps(event['snapshot'])}")

            elif etype == "RUN_FINISHED":
                emit_tool_call(
                    name="agui.lifecycle/run_finished",
                    input_data={"runId": event["runId"]},
                    output_data={"status": "finished"},
                    integration="service",
                )
                print(f"[ag-ui] RUN_FINISHED")

            time.sleep(0.02)

        # -- Emit final output --
        final_text = "".join(text_chunks)
        emit_output(final_text)

        # -- Simulate user feedback --
        emit_tool_call(
            name="agui.feedback/thumbs_up",
            input_data={"runId": run_id, "rating": "positive"},
            output_data={"recorded": True},
            integration="service",
        )
        print("[ag-ui] User feedback: thumbs_up")

    # -- Summary --
    summary = stratix.end_trial()
    events = stratix.get_events()
    print(f"\n{'=' * 60}")
    print(f"[STRATIX] Trial ended   status={summary.get('status')}  events={len(events)}")
    print(f"[STRATIX] Trace ID:     {ctx.trace_id}")
    print(f"[STRATIX] SDK present:  {HAS_AGUI_SDK}")
    for i, ev in enumerate(events):
        print(f"  [{i}] {ev.event_type:30s}  ts={ev.timestamp}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
