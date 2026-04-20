"""A2UI (Agent-to-User-Interface) protocol adapter — commerce surfaces.

Observes commerce UI lifecycle:

* ``on_surface_created`` — a new product / checkout surface is rendered.
* ``on_user_action`` — a user interacts with a surface.

PII Safety: user action payloads are hashed with SHA-256 before emission so
cleartext commerce interactions never land in telemetry.
"""

from __future__ import annotations

import uuid
import hashlib
import logging
from typing import Any, Dict

from ..._events import COMMERCE_UI_USER_ACTION, COMMERCE_UI_SURFACE_CREATED
from ._base_protocol import BaseProtocolAdapter

log = logging.getLogger(__name__)


def _sha(value: Any) -> str:
    return "sha256:" + hashlib.sha256(str(value).encode()).hexdigest()


class A2UIProtocolAdapter(BaseProtocolAdapter):
    PROTOCOL = "a2ui"
    PROTOCOL_VERSION = "0.1.0"

    def connect(self, target: Any = None, **kwargs: Any) -> Any:  # noqa: ARG002
        self._client = target
        for method, event_name, hash_payload in (
            ("on_surface_created", COMMERCE_UI_SURFACE_CREATED, False),
            ("on_user_action", COMMERCE_UI_USER_ACTION, True),
        ):
            if hasattr(target, method):
                orig = getattr(target, method)
                self._originals[method] = orig
                setattr(target, method, self._wrap(orig, event_name, hash_payload))
        return target

    def _wrap(self, original: Any, event_name: str, hash_payload: bool) -> Any:
        adapter = self

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            surface_id = kwargs.get("surface_id") or (args[0] if args else uuid.uuid4().hex[:16])
            payload: Dict[str, Any] = {"surface_id": surface_id}
            if hash_payload:
                payload["action_context_hash"] = _sha(kwargs.get("context") or args[1:] or "")
                payload["action_type"] = kwargs.get("action_type")
            else:
                payload["surface_type"] = kwargs.get("surface_type") or kwargs.get("type")
                payload["item_count"] = kwargs.get("item_count")
            adapter.emit(event_name, payload)
            return original(*args, **kwargs)

        return wrapped

    def record_surface_created(
        self,
        *,
        surface_id: str,
        surface_type: str | None = None,
        item_count: int | None = None,
    ) -> None:
        self.emit(
            COMMERCE_UI_SURFACE_CREATED,
            {"surface_id": surface_id, "surface_type": surface_type, "item_count": item_count},
        )

    def record_user_action(
        self,
        *,
        surface_id: str,
        action_type: str,
        context: Any,
    ) -> None:
        self.emit(
            COMMERCE_UI_USER_ACTION,
            {
                "surface_id": surface_id,
                "action_type": action_type,
                "action_context_hash": _sha(context),
            },
        )


def instrument_a2ui(target: Any) -> A2UIProtocolAdapter:
    from .._registry import get, register

    existing = get("a2ui")
    if existing is not None:
        existing.disconnect()
    adapter = A2UIProtocolAdapter()
    adapter.connect(target)
    register("a2ui", adapter)
    return adapter


def uninstrument_a2ui() -> None:
    from .._registry import unregister

    unregister("a2ui")
