"""
A2A Agent Card parser and event builder.

Handles discovery of Agent Cards from /.well-known/agent.json and
translation to Stratix protocol.agent_card events.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def parse_agent_card(card_json: str | dict[str, Any]) -> dict[str, Any]:
    """
    Parse an A2A Agent Card from JSON string or dict.

    Args:
        card_json: Raw Agent Card JSON string or already-parsed dict.

    Returns:
        Normalized Agent Card dict with standard field names.

    Raises:
        ValueError: If the card cannot be parsed.
    """
    if isinstance(card_json, str):
        try:
            card = json.loads(card_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid Agent Card JSON: {exc}") from exc
    else:
        card = dict(card_json)

    # Normalize field names (A2A spec uses camelCase)
    normalized: dict[str, Any] = {
        "name": card.get("name", "unknown"),
        "description": card.get("description"),
        "url": card.get("url", ""),
        "protocolVersion": card.get("protocolVersion", card.get("version", "unknown")),
        "capabilities": card.get("capabilities", {}),
        "skills": card.get("skills", []),
        "authentication": card.get("authentication", {}),
    }

    # Extract auth scheme
    auth = card.get("authentication", {})
    if isinstance(auth, dict):
        normalized["authScheme"] = auth.get("scheme") or auth.get("type")
    elif isinstance(auth, str):
        normalized["authScheme"] = auth
    else:
        normalized["authScheme"] = None

    return normalized


def discover_agent_card(
    base_url: str,
    timeout_s: float = 5.0,
) -> dict[str, Any] | None:
    """
    Discover an A2A Agent Card by fetching /.well-known/agent.json.

    Args:
        base_url: Base URL of the A2A agent.
        timeout_s: Request timeout in seconds.

    Returns:
        Parsed Agent Card dict, or None if discovery fails.
    """
    import urllib.error
    import urllib.request

    card_url = base_url.rstrip("/") + "/.well-known/agent.json"
    try:
        req = urllib.request.Request(card_url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            if resp.status == 200:
                body = resp.read().decode("utf-8")
                return parse_agent_card(body)
    except Exception as exc:
        logger.debug("Agent Card discovery failed for %s: %s", card_url, exc)
    return None
