# A2A protocol adapter

> **Canonical source:** [`protocols/a2a/adapter.py`](../../src/layerlens/instrument/adapters/protocols/a2a/adapter.py),
> [`protocols/a2a/task_lifecycle.py`](../../src/layerlens/instrument/adapters/protocols/a2a/task_lifecycle.py),
> and [`_vendored/events_protocol.py`](../../src/layerlens/instrument/_vendored/events_protocol.py).
> Every event-name string and method signature in this doc matches the literal
> `event_type` default and Python signature at source.

`layerlens.instrument.adapters.protocols.a2a.A2AAdapter` instruments the
[Agent-to-Agent (A2A) protocol](https://github.com/google/A2A) via
dual-channel instrumentation: server-side wrapping intercepts incoming
JSON-RPC requests and SSE streams, client-side wrapping traces outgoing
task submissions and streamed updates.

The adapter also handles ACP-origin payloads (IBM Agent Communication
Protocol, merged into A2A in August 2025) via a built-in
`ACPNormalizer`.

## Install

```bash
pip install 'layerlens[protocols-a2a]'
```

The base SDK already includes `httpx`, so no additional packages are
strictly required. To use the official A2A SDK, install `a2a-sdk`
separately.

## Quick start

```python
from layerlens.instrument.adapters.protocols.a2a import A2AAdapter
from layerlens.instrument.transport.sink_http import HttpEventSink

sink = HttpEventSink(adapter_name="a2a")
adapter = A2AAdapter()
adapter.add_sink(sink)
adapter.connect()

# Register an Agent Card discovered from a peer agent
adapter.register_agent_card(
    {
        "name": "research-agent",
        "url": "https://research.example.com/.well-known/agent.json",
        "protocolVersion": "0.2.0",
        "skills": [{"id": "search", "name": "Web search"}],
    },
    source="discovery",
)

# Trace a task submission + completion
adapter.on_task_submitted(
    task_id="t-001",
    receiver_url="https://research.example.com/a2a",
    task_type="research",
    submitter_agent_id="orchestrator-1",
)
adapter.on_task_completed(
    task_id="t-001",
    final_status="completed",
    artifacts=[{"type": "text", "content": "..."}],
)

adapter.disconnect()
sink.close()
```

## What's wrapped

`A2AAdapter` provides a set of `on_*` methods that the host application
calls at the appropriate hook points:

- `register_agent_card(card_data, source)` — emits `protocol.agent_card`
  (`adapter.py:118`).
- `on_task_submitted(task_id, receiver_url, ...)` — emits
  `protocol.task.submitted` (`adapter.py:165`,
  `events_protocol.py:138-141`).
- `on_task_completed(task_id, final_status, ...)` — emits
  `protocol.task.completed` (`adapter.py:199`,
  `events_protocol.py:187-190`).
- `on_task_delegation(from_agent, to_agent, context)` — emits
  `agent.handoff` only (`adapter.py:291-308`). There is no separate
  `protocol.task_delegation` event.
- `on_stream_event(sequence, payload)` — emits `protocol.stream.event`
  for each SSE message (`adapter.py:312-329`,
  `events_protocol.py:290-293`).

Server- and client-side wrappers (`A2AClient`, `A2AServer` helpers in
`a2a/client.py` and `a2a/server.py`) automatically call these hooks from
the JSON-RPC layer.

## Events emitted

| Event | Layer | When |
|---|---|---|
| `protocol.agent_card` | L4a | Per `register_agent_card` call. |
| `protocol.task.submitted` | cross-cutting | Per outbound or inbound task submission (`events_protocol.py:138-141`). |
| `protocol.task.completed` | cross-cutting | Per task terminal status (`events_protocol.py:187-190`). |
| `protocol.stream.event` | L6b | Per SSE event in a streamed task (`events_protocol.py:290-293`). |
| `agent.handoff` | cross-cutting | Per `on_task_delegation` cross-agent delegation (`adapter.py:303-308`). |

## A2A specifics

- **ACP normalization**: `ACPNormalizer` detects ACP-shaped payloads and
  rewrites them to A2A shape before emission. The original protocol
  origin (`acp` vs `a2a`) is preserved on the event.
- **Task state machine**: `TaskStateMachine` (in `a2a/task_lifecycle.py`)
  validates terminal-state transitions. Invalid transitions are logged
  as warnings and the call returns `False` — no event is emitted
  (`task_lifecycle.py:85-92`).
- **Memory sharing**: pass `memory_service=...` to the constructor and
  the adapter will store completed task contexts as episodic memory and
  share to a target agent via `AgentMemoryService.share_memory()`.
- **Artifact hashing**: returned artifacts are sha256-hashed and stored
  on the event as `artifact_hashes` for later content-addressing.
- **Health**: `probe_health(endpoint)` fetches `endpoint/.well-known/agent.json`
  and returns reachability + latency.

## Capture config

```python
from layerlens.instrument.adapters._base import CaptureConfig

# All protocol.* events are L4a — the standard preset captures them.
adapter = A2AAdapter(capture_config=CaptureConfig.standard())
```

## BYOK

A2A authentication is per-agent (`authScheme` in the Agent Card). The
adapter does not own those credentials; they belong to the underlying
A2A client. For platform-managed BYOK see `docs/adapters/byok.md`.
