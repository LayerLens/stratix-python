"""
STRATIX Protocol Health Probes

Abstractions for probing protocol endpoint health, including
SSE liveness checks and JSON-RPC ping.
"""

from __future__ import annotations

import time
import logging
from typing import Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class HealthProbeResult:
    """Result of a protocol health probe."""

    reachable: bool
    latency_ms: float
    protocol_version: str | None = None
    endpoint: str | None = None
    error: str | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "reachable": self.reachable,
            "latency_ms": self.latency_ms,
            "protocol_version": self.protocol_version,
            "endpoint": self.endpoint,
            "error": self.error,
            "metadata": self.metadata or {},
        }


def probe_http_endpoint(
    url: str,
    timeout_s: float = 5.0,
    expected_status: int = 200,
) -> HealthProbeResult:
    """
    Probe an HTTP endpoint for liveness.

    Uses urllib to avoid adding a hard dependency on httpx/requests.

    Args:
        url: Endpoint URL to probe
        timeout_s: Timeout in seconds
        expected_status: Expected HTTP status code

    Returns:
        HealthProbeResult
    """
    import urllib.error
    import urllib.request

    start = time.monotonic()
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            latency = (time.monotonic() - start) * 1000
            reachable = resp.status == expected_status
            return HealthProbeResult(
                reachable=reachable,
                latency_ms=latency,
                endpoint=url,
            )
    except urllib.error.URLError as exc:
        latency = (time.monotonic() - start) * 1000
        return HealthProbeResult(
            reachable=False,
            latency_ms=latency,
            endpoint=url,
            error=str(exc),
        )
    except Exception as exc:
        latency = (time.monotonic() - start) * 1000
        return HealthProbeResult(
            reachable=False,
            latency_ms=latency,
            endpoint=url,
            error=str(exc),
        )


def probe_a2a_agent_card(url: str, timeout_s: float = 5.0) -> HealthProbeResult:
    """
    Probe an A2A endpoint by fetching its Agent Card at /.well-known/agent.json.

    Args:
        url: Base URL of the A2A agent
        timeout_s: Timeout in seconds

    Returns:
        HealthProbeResult with protocol_version from the card if available
    """
    import json
    import urllib.error
    import urllib.request

    card_url = url.rstrip("/") + "/.well-known/agent.json"
    start = time.monotonic()
    try:
        req = urllib.request.Request(card_url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            latency = (time.monotonic() - start) * 1000
            if resp.status == 200:
                body = json.loads(resp.read())
                version = body.get("protocolVersion") or body.get("version")
                return HealthProbeResult(
                    reachable=True,
                    latency_ms=latency,
                    protocol_version=version,
                    endpoint=card_url,
                    metadata={"agent_name": body.get("name")},
                )
            return HealthProbeResult(
                reachable=False,
                latency_ms=latency,
                endpoint=card_url,
                error=f"HTTP {resp.status}",
            )
    except Exception as exc:
        latency = (time.monotonic() - start) * 1000
        return HealthProbeResult(
            reachable=False,
            latency_ms=latency,
            endpoint=card_url,
            error=str(exc),
        )


def probe_mcp_server(url: str, timeout_s: float = 5.0) -> HealthProbeResult:
    """
    Probe an MCP server for liveness.

    MCP servers typically expose a health or capabilities endpoint.

    Args:
        url: MCP server URL
        timeout_s: Timeout in seconds

    Returns:
        HealthProbeResult
    """
    return probe_http_endpoint(url, timeout_s=timeout_s)
