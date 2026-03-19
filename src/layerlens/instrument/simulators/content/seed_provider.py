"""Tier 1: Seed data content provider.

Loads content from agentforce-synthetic-data/ Langfuse JSON files.
125 pre-existing traces across 5 scenarios provide rich seed data.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .base import ContentProvider


class SeedContentProvider(ContentProvider):
    """Seed data content provider (Tier 1).

    Loads real content from Langfuse trace JSON files in the
    agentforce-synthetic-data/ directory.
    """

    def __init__(
        self,
        seed_data_path: str,
        seed: int | None = None,
    ):
        self._seed_data_path = Path(seed_data_path)
        self._rng = random.Random(seed)
        self._traces: dict[str, list[dict[str, Any]]] = {}
        self._loaded = False
        # Cache selected trace per (scenario, topic) to ensure consistency
        # within a single conversation turn
        self._trace_cache: dict[tuple[str, str], dict[str, Any]] = {}

    def _ensure_loaded(self) -> None:
        """Lazy-load seed data from disk."""
        if self._loaded:
            return
        self._loaded = True

        if not self._seed_data_path.exists():
            return

        # Look for scenario directories: scenario_*/langfuse/
        for scenario_dir in self._seed_data_path.iterdir():
            if not scenario_dir.is_dir() or not scenario_dir.name.startswith("scenario_"):
                continue
            scenario_name = scenario_dir.name.replace("scenario_", "")
            langfuse_dir = scenario_dir / "langfuse"
            if not langfuse_dir.exists():
                langfuse_dir = scenario_dir  # Try direct structure

            traces = []
            for json_file in langfuse_dir.glob("*.json"):
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        traces.extend(data)
                    else:
                        traces.append(data)
                except (json.JSONDecodeError, OSError):
                    continue
            if traces:
                self._traces[scenario_name] = traces

    def _get_trace(self, scenario: str, topic: str | None = None) -> dict[str, Any] | None:
        """Get a trace for the scenario, cached by (scenario, topic) for consistency."""
        self._ensure_loaded()
        cache_key = (scenario, topic or "")
        if cache_key in self._trace_cache:
            return self._trace_cache[cache_key]
        traces = self._traces.get(scenario, [])
        if not traces:
            return None
        trace = self._rng.choice(traces)
        self._trace_cache[cache_key] = trace
        return trace

    def _extract_messages(self, trace: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract messages from a Langfuse trace."""
        messages = []
        observations = trace.get("observations", [])
        for obs in observations:
            if obs.get("type") == "GENERATION":
                if "input" in obs and isinstance(obs["input"], list):
                    messages.extend(obs["input"])
                if "output" in obs:
                    messages.append(obs["output"])
        return messages

    def get_user_message(self, scenario: str, topic: str, turn: int = 1) -> str:
        trace = self._get_trace(scenario, topic)
        if trace:
            messages = self._extract_messages(trace)
            user_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "user"]
            if user_msgs:
                idx = (turn - 1) % len(user_msgs)
                return user_msgs[idx].get("content", "How can you help me?")
        return f"I need help with a {topic.replace('_', ' ').lower()} issue."

    def get_agent_response(
        self,
        scenario: str,
        topic: str,
        turn: int = 1,
        tool_results: dict[str, Any] | None = None,
    ) -> str:
        trace = self._get_trace(scenario, topic)
        if trace:
            messages = self._extract_messages(trace)
            agent_msgs = [
                m for m in messages
                if isinstance(m, dict) and m.get("role") == "assistant"
            ]
            if agent_msgs:
                idx = (turn - 1) % len(agent_msgs)
                return agent_msgs[idx].get("content", "Let me help you with that.")
        return "I'll look into that for you right away."

    def get_system_prompt(self, scenario: str, agent_name: str) -> str:
        trace = self._get_trace(scenario, None)
        if trace:
            messages = self._extract_messages(trace)
            system_msgs = [
                m for m in messages
                if isinstance(m, dict) and m.get("role") == "system"
            ]
            if system_msgs:
                return system_msgs[0].get("content", "")
        return f"You are a {scenario.replace('_', ' ')} agent named {agent_name}."

    def get_tool_input(self, action_name: str, topic: str) -> dict[str, Any]:
        return {"action": action_name, "query": topic}

    def get_tool_output(self, action_name: str, topic: str) -> dict[str, Any]:
        return {"result": "success", "action": action_name}

    def get_topics(self, scenario: str) -> list[str]:
        """Infer topics from loaded traces."""
        self._ensure_loaded()
        traces = self._traces.get(scenario, [])
        topics = set()
        for trace in traces:
            meta = trace.get("metadata", {})
            if "topic" in meta:
                topics.add(meta["topic"])
            elif "tags" in trace:
                for tag in trace["tags"]:
                    if tag != scenario:
                        topics.add(tag)
        return sorted(topics) if topics else [f"{scenario}_topic_1"]

    @property
    def loaded_scenarios(self) -> list[str]:
        self._ensure_loaded()
        return sorted(self._traces.keys())

    @property
    def trace_count(self) -> int:
        self._ensure_loaded()
        return sum(len(traces) for traces in self._traces.values())
