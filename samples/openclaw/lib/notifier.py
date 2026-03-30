"""
Alert & Leaderboard Notifier
==============================

Publishes evaluation results to Telegram, Discord, Slack, or stdout.
Used by Demo 1 (leaderboard) and Demo 6 (drift alerts).

In production, this would integrate with real messaging APIs.  For demos,
it logs messages to stdout with channel-style formatting.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class Notifier:
    """
    Multi-channel notification publisher.

    Supports Telegram, Discord, Slack (simulated), and stdout (always active).
    Each channel is a URI like ``telegram://channel-name`` or ``stdout://``.
    """

    def __init__(self, channels: list[str] | None = None) -> None:
        self.channels = channels or ["stdout://"]

    def publish(self, message: str, *, data: dict[str, Any] | None = None) -> None:
        """Send a message to all configured channels."""
        for channel in self.channels:
            self._send(channel, message, data)

    def publish_leaderboard(
        self,
        title: str,
        entries: list[dict[str, Any]],
    ) -> None:
        """Publish a formatted leaderboard update."""
        lines = [f"--- {title} ---"]
        for i, entry in enumerate(entries, 1):
            model = entry.get("model_id", "unknown")
            score = entry.get("aggregate_score", 0.0)
            medal = {1: "1st", 2: "2nd", 3: "3rd"}.get(i, f"{i}th")
            lines.append(f"  {medal}: {model} -- {score:.1f}/10")
        lines.append("")
        self.publish("\n".join(lines), data={"entries": entries})

    def publish_alert(
        self,
        severity: str,
        title: str,
        detail: str,
    ) -> None:
        """Publish a severity-tagged alert."""
        icon = {"critical": "!!!", "warning": "!!", "info": "i"}.get(severity, "?")
        msg = f"[{icon} {severity.upper()}] {title}\n  {detail}"
        self.publish(msg, data={"severity": severity, "title": title})

    def _send(self, channel: str, message: str, data: dict[str, Any] | None) -> None:
        """Route message to the appropriate channel handler."""
        proto = channel.split("://")[0] if "://" in channel else "stdout"

        if proto == "stdout":
            print(message)
        elif proto in ("telegram", "discord", "slack"):
            target = channel.split("://", 1)[1] if "://" in channel else channel
            logger.info("[%s -> %s] %s", proto.upper(), target, message[:120])
            print(f"[{proto}:{target}] {message}")
        else:
            logger.warning("Unknown channel protocol: %s", proto)
            print(message)
