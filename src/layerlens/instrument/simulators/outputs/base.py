"""Base output formatter ABC."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..span_model import SimulatedTrace


class BaseOutputFormatter(ABC):
    """Abstract base for output formatters.

    Serializes SimulatedTrace objects to wire format
    (OTLP JSON, Langfuse JSON, or STRATIX native dicts).
    """

    @abstractmethod
    def format_trace(self, trace: SimulatedTrace) -> dict[str, Any]:
        """Format a single trace to wire format."""

    def format_batch(self, traces: list[SimulatedTrace]) -> list[dict[str, Any]]:
        """Format a batch of traces."""
        return [self.format_trace(t) for t in traces]

    def write_to_file(self, traces: list[SimulatedTrace], path: str) -> None:
        """Format and write traces to a JSON file."""
        formatted = self.format_batch(traces)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(formatted, f, indent=2)
