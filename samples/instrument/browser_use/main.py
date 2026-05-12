"""Sample: instrument a browser_use agent with the LayerLens adapter.

This sample is intentionally **offline** — it does not require the
``browser-use`` runtime, a Playwright install, an OpenAI key, or any
network access. It builds a duck-typed ``Agent`` (the same shape
``browser_use.Agent`` exposes), wraps it via
``BrowserUseAdapter.instrument_agent``, and runs ``agent.run()`` to
exercise the full lifecycle: session start → per-step navigation +
action + screenshot + LLM call → session end.

The events are captured into an in-process recording client and
printed so you can see exactly what would ship to atlas-app under
real conditions — including:

  * SHA-256 references in place of the multi-megabyte screenshot
    blob (per the truncation policy);
  * truncated DOM payloads when the page HTML exceeds 16 KiB;
  * ``agent.error`` emission when the agent raises (try the
    ``--fail`` flag).

For a real end-to-end run against the browser_use runtime, install the
extra and replace ``_FakeAgent`` with ``browser_use.Agent``::

    pip install 'layerlens[browser-use]'
    # Then swap _FakeAgent for browser_use.Agent and configure
    # HttpEventSink to ship events to atlas-app.

Required environment for the offline sample: none.

Run::

    python -m samples.instrument.browser_use.main
    python -m samples.instrument.browser_use.main --fail
"""

from __future__ import annotations

import sys
import asyncio
from typing import Any, Dict, List, Optional

from layerlens.instrument.adapters._base import CaptureConfig
from layerlens.instrument.adapters.frameworks.browser_use import (
    BrowserUseAdapter,
)


class _FakeAction:
    def __init__(self, type_: str, target: Optional[str] = None) -> None:
        self.type = type_
        self.target = target


class _FakeStep:
    """Mirrors a browser_use AgentHistoryList entry."""

    def __init__(
        self,
        url: str,
        action_type: str,
        target: Optional[str] = None,
        screenshot: Optional[bytes] = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        self.url = url
        self.action = _FakeAction(action_type, target=target)
        self.screenshot = screenshot

        usage = type("_Usage", (), {})()
        usage.input_tokens = prompt_tokens
        usage.output_tokens = completion_tokens
        usage.prompt_tokens = prompt_tokens
        usage.completion_tokens = completion_tokens
        self.model_usage = usage


class _FakeHistory:
    def __init__(self, steps: List[_FakeStep]) -> None:
        self.history = steps


class _FakeBrowser:
    headless = True
    user_data_dir = "/tmp/sample-profile"


class _FakeAgent:
    """Duck-typed browser_use Agent for the offline demo."""

    def __init__(self, fail: bool = False) -> None:
        self.name = "demo-bot"
        self.task = "find the price of a Logitech mouse on a demo store"
        self.model = "gpt-4o-mini"
        self.browser = _FakeBrowser()
        self._fail = fail
        # Multi-step history mirroring a typical browser_use run.
        screenshot_blob = b"\x89PNG\r\n\x1a\n" + b"PIXEL_DATA" * 5000  # ~50 KB
        self._history = _FakeHistory(
            steps=[
                _FakeStep(
                    url="https://store.example.com/",
                    action_type="navigate",
                    screenshot=screenshot_blob,
                    prompt_tokens=120,
                    completion_tokens=40,
                ),
                _FakeStep(
                    url="https://store.example.com/search?q=logitech+mouse",
                    action_type="type",
                    target="#search",
                    screenshot=screenshot_blob,
                    prompt_tokens=80,
                    completion_tokens=20,
                ),
                _FakeStep(
                    url="https://store.example.com/p/logitech-mx-master-3s",
                    action_type="click",
                    target=".product-card:first-child",
                    screenshot=screenshot_blob,
                    prompt_tokens=200,
                    completion_tokens=60,
                ),
            ]
        )

    async def run(self, *args: Any, **kwargs: Any) -> Any:
        if self._fail:
            raise RuntimeError("simulated rate-limit / page-load failure")
        return self._history


class _RecordingClient:
    """Stand-in for the LayerLens client. Captures events for inspection."""

    def __init__(self, org_id: str = "org_demo") -> None:
        self.events: List[Dict[str, Any]] = []
        self.org_id = org_id

    def emit(self, event_type: str, payload: Dict[str, Any]) -> None:
        self.events.append({"event_type": event_type, "payload": payload})


def _short(value: Any, length: int = 60) -> str:
    text = str(value) if value is not None else ""
    if len(text) > length:
        return text[: length - 3] + "..."
    return text


def main(argv: Optional[List[str]] = None) -> int:
    args = list(argv or sys.argv[1:])
    fail_mode = "--fail" in args

    client = _RecordingClient(org_id="org_demo")
    adapter = BrowserUseAdapter(
        stratix=client,
        capture_config=CaptureConfig.full(),
        org_id="org_demo",
    )
    adapter.connect()

    agent = _FakeAgent(fail=fail_mode)
    adapter.instrument_agent(agent)

    try:
        if fail_mode:
            try:
                asyncio.run(agent.run())
            except RuntimeError as exc:
                print(f"Agent raised (expected): {exc}")
        else:
            result = asyncio.run(agent.run())
            print(f"Agent finished. {len(result.history)} step(s) executed.")
    finally:
        adapter.disconnect()

    print(f"\nEmitted {len(client.events)} event(s):")
    for evt in client.events:
        kind = evt["event_type"]
        payload = evt["payload"]
        # Print a compact one-liner per event with the most relevant
        # fields for that type.
        if kind == "browser.session.start":
            detail = f"session={_short(payload.get('session_id'), 12)}"
        elif kind == "agent.input":
            detail = f"task={_short(payload.get('input'))}"
        elif kind == "browser.navigate":
            detail = f"url={_short(payload.get('url'))}"
        elif kind in {"browser.action", "tool.call"}:
            detail = f"action={_short(payload.get('tool_name') or payload.get('action_type'))}"
        elif kind == "browser.screenshot":
            detail = f"screenshot={_short(payload.get('screenshot'), 50)}"
        elif kind == "browser.dom.extract":
            detail = f"elements={payload.get('element_count')}"
        elif kind == "model.invoke":
            detail = (
                f"model={payload.get('model')} "
                f"tokens={payload.get('tokens_prompt')}/{payload.get('tokens_completion')}"
            )
        elif kind == "cost.record":
            detail = f"total_tokens={payload.get('tokens_total')}"
        elif kind == "agent.output":
            detail = f"duration_ns={payload.get('duration_ns')}"
        elif kind == "agent.error":
            detail = f"{payload.get('error_type')}: {_short(payload.get('error_message'))}"
        elif kind == "environment.config":
            detail = f"agent={payload.get('agent_name')} model={payload.get('model')}"
        else:
            detail = ""
        print(f"  - {kind:>26}  org={payload.get('org_id', '<unset>')}  {detail}")

    print(
        "\nReplace _FakeAgent with browser_use.Agent and add an "
        "HttpEventSink to ship telemetry to the LayerLens dashboard."
    )

    snap = adapter.resilience_snapshot()
    if snap["resilience_failures_total"]:
        print(
            f"\nResilience: {snap['resilience_failures_total']} callback failure(s) "
            f"caught and contained: {snap['resilience_failures_by_callback']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
