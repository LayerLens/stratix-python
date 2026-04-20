"""Sample: AutoGen two-agent conversation."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from adapters._shared import capture_events  # type: ignore[import-not-found]

from layerlens.instrument.adapters.frameworks.autogen import AutoGenAdapter


def main() -> None:
    try:
        from autogen import AssistantAgent, UserProxyAgent  # type: ignore[import-not-found]
    except ImportError:
        print("Install: pip install 'layerlens[autogen]' pyautogen")
        return

    config = {"config_list": [{"model": "gpt-4o-mini", "api_key": os.environ.get("OPENAI_API_KEY", "")}]}
    assistant = AssistantAgent(name="assistant", llm_config=config)
    user = UserProxyAgent(
        name="user", human_input_mode="NEVER", max_consecutive_auto_reply=1, code_execution_config=False
    )

    AutoGenAdapter(None).connect([assistant, user])
    with capture_events("autogen_conversation"):
        user.initiate_chat(assistant, message="Say grass is green in one line.")


if __name__ == "__main__":
    main()
