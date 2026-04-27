from __future__ import annotations

import abc
from typing import Any, Dict
from dataclasses import field, dataclass


@dataclass
class AdapterInfo:
    """Metadata describing a connected adapter."""

    name: str
    adapter_type: str  # "provider" or "framework"
    version: str = "0.1.0"
    connected: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAdapter(abc.ABC):
    """Minimal interface that every adapter (provider or framework) must implement."""

    @abc.abstractmethod
    def connect(self, target: Any = None, **kwargs: Any) -> Any:
        """Activate instrumentation. Providers: target = SDK client. Frameworks: target = layerlens client."""

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Deactivate instrumentation and restore originals."""

    @abc.abstractmethod
    def adapter_info(self) -> AdapterInfo:
        """Return metadata about this adapter."""

    @property
    def is_connected(self) -> bool:
        return self.adapter_info().connected
