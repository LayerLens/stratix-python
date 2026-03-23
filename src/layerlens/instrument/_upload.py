from __future__ import annotations

import os
import json
import logging
import tempfile
from typing import Any, Dict

log: logging.Logger = logging.getLogger(__name__)


def upload_trace(client: Any, trace_data: Dict[str, Any]) -> None:
    fd, path = tempfile.mkstemp(suffix=".json", prefix="layerlens_trace_")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump([trace_data], f, default=str)
        client.traces.upload(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            log.debug("Failed to remove temp trace file: %s", path)


async def async_upload_trace(client: Any, trace_data: Dict[str, Any]) -> None:
    fd, path = tempfile.mkstemp(suffix=".json", prefix="layerlens_trace_")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump([trace_data], f, default=str)
        await client.traces.upload(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            log.debug("Failed to remove temp trace file: %s", path)
