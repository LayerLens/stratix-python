"""A2A Agent Card parsing and discovery.

Fetches ``/.well-known/agent.json`` from an A2A peer and normalises the
result so the adapter can emit a ``a2a.agent.discovered`` payload with
consistent field names regardless of the server's casing choices.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

log = logging.getLogger(__name__)


def parse_agent_card(card_json: str | dict[str, Any]) -> dict[str, Any]:
    """Parse an Agent Card (JSON string or dict) into a normalised dict."""
    if isinstance(card_json, str):
        try:
            card = json.loads(card_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid Agent Card JSON: {exc}") from exc
    else:
        card = dict(card_json)

    auth = card.get("authentication", {}) or {}
    if isinstance(auth, dict):
        auth_scheme: Optional[str] = auth.get("scheme") or auth.get("type")
    elif isinstance(auth, str):
        auth_scheme = auth
    else:
        auth_scheme = None

    return {
        "name": card.get("name", "unknown"),
        "description": card.get("description"),
        "url": card.get("url", ""),
        "protocolVersion": card.get("protocolVersion", card.get("version", "unknown")),
        "capabilities": card.get("capabilities", {}),
        "skills": card.get("skills", []),
        "authentication": auth,
        "authScheme": auth_scheme,
    }


def discover_agent_card(base_url: str, timeout_s: float = 5.0) -> Optional[dict[str, Any]]:
    """Fetch and parse an Agent Card. Returns ``None`` on failure."""
    import urllib.request

    card_url = base_url.rstrip("/") + "/.well-known/agent.json"
    try:
        with urllib.request.urlopen(
            urllib.request.Request(card_url, method="GET"),
            timeout=timeout_s,
        ) as resp:
            if getattr(resp, "status", 200) == 200:
                return parse_agent_card(resp.read().decode("utf-8"))
    except Exception as exc:
        log.debug("Agent Card discovery failed for %s: %s", card_url, exc)
    return None
