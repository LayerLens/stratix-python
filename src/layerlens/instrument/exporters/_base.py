"""
STRATIX Exporter Base Class

Defines the interface for event exporters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from layerlens.instrument.schema.event import STRATIXEvent


class Exporter(ABC):
    """
    Base class for STRATIX event exporters.

    Exporters send events to external telemetry backends.
    """

    @abstractmethod
    def export(self, event: "STRATIXEvent") -> None:
        """
        Export a single event.

        Args:
            event: The event to export
        """
        pass

    @abstractmethod
    def export_batch(self, events: list["STRATIXEvent"]) -> None:
        """
        Export a batch of events.

        Args:
            events: List of events to export
        """
        pass

    @abstractmethod
    def flush(self) -> None:
        """Flush any buffered events."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the exporter and release resources."""
        pass
