from __future__ import annotations

import os
import json
import asyncio
import logging
import tempfile
from typing import Any, Dict, Optional

log: logging.Logger = logging.getLogger(__name__)


def _write_trace_file(payload: Dict[str, Any]) -> str:
    """Write trace payload to a temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".json", prefix="layerlens_trace_")
    with os.fdopen(fd, "w") as f:
        json.dump([payload], f, default=str)
    return path


def upload_trace(
    client: Any,
    trace_data: Dict[str, Any],
    attestation: Optional[Dict[str, Any]] = None,
) -> None:
    payload = trace_data
    if attestation:
        payload = {**trace_data, "attestation": attestation}
    path = _write_trace_file(payload)
    try:
        client.traces.upload(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            log.debug("Failed to remove temp trace file: %s", path)


async def async_upload_trace(
    client: Any,
    trace_data: Dict[str, Any],
    attestation: Optional[Dict[str, Any]] = None,
) -> None:
    payload = trace_data
    if attestation:
        payload = {**trace_data, "attestation": attestation}
    path = await asyncio.to_thread(_write_trace_file, payload)
    try:
        await client.traces.upload(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            log.debug("Failed to remove temp trace file: %s", path)
