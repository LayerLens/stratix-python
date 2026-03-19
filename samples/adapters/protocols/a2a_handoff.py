"""Google A2A (Agent-to-Agent) protocol instrumentation demo.

Demonstrates:
- Agent Card creation with capabilities and skills
- Task delegation between a coordinator and specialist agent
- Cross-agent trace correlation via STRATIX handoff events
- SSE stream handling for real-time task status updates

Requires:
    LAYERLENS_STRATIX_API_KEY  - LayerLens platform API key (optional)

If the ``a2a`` SDK is not installed, the protocol flow is simulated
using STRATIX emit functions so the trace structure is identical.
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
    emit_handoff,
    emit_tool_call,
)

# ---------------------------------------------------------------------------
# Try importing the real A2A SDK; fall back to simulation
# ---------------------------------------------------------------------------

try:
    from a2a.types import AgentCard, AgentSkill, Task, TaskState  # type: ignore[import-untyped]

    HAS_A2A_SDK = True
except ImportError:
    HAS_A2A_SDK = False
    print(
        "[a2a] SDK not installed. Run: pip install a2a-sdk\n"
        "      Continuing with simulated protocol flow.\n"
    )


# ---------------------------------------------------------------------------
# Mock helpers (used when SDK is absent)
# ---------------------------------------------------------------------------

def _mock_agent_card(name: str, url: str, skills: list[str]) -> dict:
    return {
        "name": name,
        "url": url,
        "version": "0.2.1",
        "capabilities": {"streaming": True, "pushNotifications": False},
        "skills": [{"id": s, "name": s.replace("_", " ").title()} for s in skills],
    }


def _mock_sse_stream(task_id: str) -> list[dict]:
    """Simulate SSE events for a task lifecycle."""
    return [
        {"event": "task.status", "data": {"id": task_id, "state": "working"}},
        {"event": "task.artifact", "data": {"id": task_id, "parts": [{"text": "Quarterly revenue rose 12%."}]}},
        {"event": "task.status", "data": {"id": task_id, "state": "completed"}},
    ]


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="A2A protocol instrumentation demo")
    parser.add_argument("--policy-ref", default="stratix-policy-cs-v1@1.0.0")
    parser.add_argument("--coordinator", default="coordinator_agent")
    parser.add_argument("--specialist", default="finance_analyst_agent")
    parser.add_argument("--task-prompt", default="Summarize Q4 earnings for ACME Corp.")
    args = parser.parse_args()

    stratix = STRATIX(policy_ref=args.policy_ref, agent_id=args.coordinator, framework="a2a")
    ctx = stratix.start_trial()
    print(f"[STRATIX] Trial started  trace_id={ctx.trace_id}")

    with stratix.context():
        # -- 1. Agent Card discovery --
        coordinator_card = _mock_agent_card(args.coordinator, "http://localhost:8001", ["route_tasks"])
        specialist_card = _mock_agent_card(args.specialist, "http://localhost:8002", ["financial_analysis"])
        emit_tool_call(
            name="a2a.discover_agent_card",
            input_data={"url": specialist_card["url"]},
            output_data=specialist_card,
            latency_ms=45.0,
            integration="service",
        )
        print(f"[a2a] Discovered agent card: {specialist_card['name']}  skills={[s['id'] for s in specialist_card['skills']]}")

        # -- 2. Emit user input --
        emit_input(args.task_prompt, role="human")
        print(f"[a2a] User request: {args.task_prompt}")

        # -- 3. Handoff: coordinator -> specialist --
        task_id = str(uuid.uuid4())
        emit_handoff(
            source_agent=args.coordinator,
            target_agent=args.specialist,
            context_passed={"task_id": task_id, "prompt": args.task_prompt},
        )
        print(f"[a2a] Handoff  {args.coordinator} -> {args.specialist}  task={task_id[:8]}...")

        # -- 4. Task submission (tool call event) --
        t0 = time.perf_counter()
        emit_tool_call(
            name="a2a.tasks/send",
            input_data={"task_id": task_id, "message": args.task_prompt},
            output_data={"state": "submitted"},
            latency_ms=120.0,
            integration="service",
        )
        print(f"[a2a] Task submitted  id={task_id[:8]}...")

        # -- 5. SSE stream consumption --
        for sse_event in _mock_sse_stream(task_id):
            elapsed = (time.perf_counter() - t0) * 1000
            emit_tool_call(
                name=f"a2a.sse/{sse_event['event']}",
                input_data={"task_id": task_id},
                output_data=sse_event["data"],
                latency_ms=elapsed,
                integration="service",
            )
            state = sse_event["data"].get("state", "artifact")
            print(f"[a2a] SSE  event={sse_event['event']}  state={state}")
            time.sleep(0.05)

        # -- 6. Extract artifact and emit output --
        artifact_text = "Quarterly revenue rose 12%."
        emit_output(artifact_text)
        print(f"[a2a] Final output: {artifact_text}")

    # -- 7. Summary --
    summary = stratix.end_trial()
    events = stratix.get_events()
    print(f"\n{'=' * 60}")
    print(f"[STRATIX] Trial ended   status={summary.get('status')}  events={len(events)}")
    print(f"[STRATIX] Trace ID:     {ctx.trace_id}")
    print(f"[STRATIX] SDK present:  {HAS_A2A_SDK}")
    for i, ev in enumerate(events):
        print(f"  [{i}] {ev.event_type:30s}  ts={ev.timestamp}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
