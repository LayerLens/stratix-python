"""Typed-event regression tests for the agentforce adapter.

Bundle #6 of the typed-events migration ports the agentforce
:meth:`AgentForceAdapter.import_sessions` per-event re-emission loop
from :meth:`emit_dict_event` to typed
:meth:`BaseAdapter.emit_event` calls. Because AgentForce is an
*importer-style* adapter (events come from
:class:`AgentForceNormalizer` rather than runtime instrumentation),
the migration sets ``ALLOW_UNREGISTERED_EVENTS = True`` — events
flow through as open-ended Pydantic models rather than being
re-shaped onto the canonical 13-event taxonomy. This is the same
policy decision PR #129 made for langfuse.

The full agentforce test suite is untracked on PR #129's foundation
branch (``test_agentforce_adapter.py`` does not exist on this
branch). Additionally,
:class:`AgentForceAdapter` imports its sibling submodules
(``auth.py``, ``importer.py``, ``normalizer.py``) at module load —
all untracked. We use :func:`pytest.importorskip` to defer
collection until those submodules merge in.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List
from unittest import mock

import pytest

from layerlens._compat.pydantic import (
    BaseModel as _CompatBaseModel,
    model_dump as _compat_model_dump,
)
from layerlens.instrument.adapters._base.capture import CaptureConfig

# agentforce/{auth,importer,normalizer}.py are untracked on PR
# #129's foundation branch — adapter.py imports them at module
# load. Defer collection until those submodules land.
agentforce_module = pytest.importorskip(
    "layerlens.instrument.adapters.frameworks.agentforce.adapter"
)
AgentForceAdapter = agentforce_module.AgentForceAdapter


class _RecordingStratix:
    org_id: str = "test-org"

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self.typed_payloads: List[Any] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:
        if len(args) == 2 and isinstance(args[0], str):
            self.events.append({"event_type": args[0], "payload": args[1]})
            return
        if args and isinstance(args[0], _CompatBaseModel):
            payload_model = args[0]
            self.typed_payloads.append(payload_model)
            event_type = getattr(payload_model, "event_type", "<unknown>")
            self.events.append(
                {"event_type": event_type, "payload": _compat_model_dump(payload_model)}
            )


def _build_adapter() -> Any:
    """Build an AgentForceAdapter with a mocked importer.

    The importer is the only piece exercised by ``import_sessions``
    other than the ``emit_event`` loop — we patch it so the
    regression test stays focused on the typed-emission contract.
    """
    adapter = AgentForceAdapter(
        stratix=_RecordingStratix(),
        capture_config=CaptureConfig.full(),
        org_id="test-org",
    )
    adapter._connected = True
    return adapter


class TestAgentforceTypedEvents:
    def test_import_sessions_emits_typed_open_ended_payloads(self) -> None:
        adapter = _build_adapter()
        # Stub importer: returns a list of normalised events plus a
        # result object whose ``events_generated`` counter the loop
        # increments.
        importer = mock.MagicMock()
        result = mock.MagicMock(events_generated=0)
        events = [
            {
                "event_type": "agent.input",
                "payload": {"text": "what's my balance?"},
                "identity": {"user_id": "u1", "session_id": "s1"},
                "timestamp": "2026-04-28T10:00:00Z",
            },
            {
                "event_type": "tool.call",
                "payload": {"tool_name": "lookup_balance", "tool_input": {"acct": "123"}},
            },
        ]
        importer.import_sessions = mock.MagicMock(return_value=(events, result))
        adapter._importer = importer

        adapter.import_sessions(start_date="2026-04-28")

        stratix = adapter._stratix
        # Two events emitted, both as typed open-ended models.
        assert len(stratix.events) == 2
        # Identity / timestamp are folded onto the payload root with
        # underscore prefixes — preserves the legacy downstream
        # contract.
        assert stratix.events[0]["payload"]["_identity"] == {"user_id": "u1", "session_id": "s1"}
        assert stratix.events[0]["payload"]["_timestamp"] == "2026-04-28T10:00:00Z"
        # Event type preserved on the dict view.
        assert stratix.events[0]["event_type"] == "agent.input"
        assert stratix.events[1]["event_type"] == "tool.call"

    def test_no_deprecation_warning_after_migration(self) -> None:
        adapter = _build_adapter()
        importer = mock.MagicMock()
        result = mock.MagicMock(events_generated=0)
        importer.import_sessions = mock.MagicMock(
            return_value=([{"event_type": "agent.output", "payload": {"text": "ok"}}], result)
        )
        adapter._importer = importer

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            adapter.import_sessions()

    def test_allow_unregistered_events_is_set(self) -> None:
        """The migration explicitly opts agentforce into the
        unregistered-events policy because it is an importer-style
        adapter whose taxonomy is upstream-defined.
        """
        assert AgentForceAdapter.ALLOW_UNREGISTERED_EVENTS is True
