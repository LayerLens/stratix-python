"""Sample: instrument an AutoGen one-turn conversation with LayerLens.

Builds an ``AssistantAgent`` + ``UserProxyAgent``, connects them through the
``AutoGenAdapter``, and runs a single-turn ``initiate_chat`` exchange. Each
agent ``send`` / ``receive`` / ``generate_reply`` emits LayerLens events that
ship to atlas-app via ``HttpEventSink``.

Required environment:

* ``OPENAI_API_KEY`` — used by AutoGen's ``llm_config``.
* ``LAYERLENS_STRATIX_API_KEY`` — your LayerLens API key (optional).
* ``LAYERLENS_STRATIX_BASE_URL`` — atlas-app base URL (optional).

Run::

    pip install 'layerlens[autogen,providers-openai]'
    python -m samples.instrument.autogen.main
"""

from __future__ import annotations

import os
import sys

from layerlens.instrument.adapters._base import CaptureConfig
from layerlens.instrument.transport.sink_http import HttpEventSink
from layerlens.instrument.adapters.frameworks.autogen import AutoGenAdapter


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set; cannot run sample.", file=sys.stderr)
        return 2

    try:
        from autogen import AssistantAgent, UserProxyAgent
    except ImportError:
        print(
            "pyautogen not installed. Install with:\n"
            "    pip install 'layerlens[autogen,providers-openai]'",
            file=sys.stderr,
        )
        return 2

    sink = HttpEventSink(
        adapter_name="autogen",
        path="/telemetry/spans",
        max_batch=10,
        flush_interval_s=1.0,
    )

    adapter = AutoGenAdapter(capture_config=CaptureConfig.standard())
    adapter.add_sink(sink)
    adapter.connect()

    llm_config = {
        "config_list": [
            {"model": "gpt-4o-mini", "api_key": os.environ["OPENAI_API_KEY"]},
        ],
        "temperature": 0,
        "timeout": 30,
    }

    assistant = AssistantAgent(
        name="assistant",
        system_message="You are a concise assistant. Reply with one short sentence.",
        llm_config=llm_config,
    )
    user = UserProxyAgent(
        name="user",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False,
        is_termination_msg=lambda _msg: True,
    )

    try:
        adapter.connect_agents(assistant, user)
        user.initiate_chat(assistant, message="What is 2 + 2?")
        last = assistant.last_message(user)
        print(f"Response: {last.get('content') if last else '(none)'}")
    finally:
        sink.close()
        adapter.disconnect()

    print("Telemetry shipped. Check the LayerLens dashboard adapter health page.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
