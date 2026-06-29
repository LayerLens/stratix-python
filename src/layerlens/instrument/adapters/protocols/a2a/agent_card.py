"""A2A Agent Card parsing, discovery, and signature provenance.

Fetches ``/.well-known/agent.json`` from an A2A peer and normalises the
result so the adapter can emit a ``a2a.agent.discovered`` payload with
consistent field names regardless of the server's casing choices.

The AgentCard ``signatures`` (a list of ``AgentCardSignature{protected,
signature, header}`` — RFC 7515 JWS over the card; spec §8.4) are the single
most security-relevant field: they let a client verify the peer's identity
before delegating. :func:`summarize_signatures` emits their PRESENCE + a
keyed-HMAC FINGERPRINT (so card authenticity is auditable even under
``capture_content=False``) — NEVER the raw JWS (D2).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Optional

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


def _extract_signatures(card: Any) -> list[Any]:
    """Return the card's ``signatures`` list (a2a AgentCard / dict / JSON str)."""
    if isinstance(card, str):
        try:
            card = json.loads(card)
        except json.JSONDecodeError:
            return []
    sigs = card.get("signatures") if isinstance(card, dict) else getattr(card, "signatures", None)
    if not sigs:
        return []
    return list(sigs)


def _raw_jws(sig: Any) -> str:
    """The raw JWS material of one signature (``protected.signature``) — used
    ONLY to compute a keyed-HMAC fingerprint; it is never emitted."""
    protected = sig.get("protected") if isinstance(sig, dict) else getattr(sig, "protected", "")
    signature = sig.get("signature") if isinstance(sig, dict) else getattr(sig, "signature", "")
    return f"{protected}.{signature}"


def summarize_signatures(card: Any, fingerprint: Callable[[Any], str]) -> dict[str, Any]:
    """Card-signature provenance for an ``a2a.agent.discovered``/``card.served``
    payload (D2). Returns the signature PRESENCE + count + a keyed-HMAC
    FINGERPRINT of the first signature's raw JWS — never the raw JWS itself, so
    card authenticity is auditable under ``capture_content=False`` without ever
    leaking the signature material.

    ``fingerprint`` is the adapter's keyed-HMAC helper (per-instance key).
    """
    sigs = _extract_signatures(card)
    summary: dict[str, Any] = {
        "signature_present": bool(sigs),
        "signature_count": len(sigs),
    }
    if sigs:
        # Fingerprint the first signature (the primary card signer). The raw
        # protected header + signature go into the HMAC and NOWHERE else.
        summary["signature_fp"] = fingerprint(_raw_jws(sigs[0]))
    return summary


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
