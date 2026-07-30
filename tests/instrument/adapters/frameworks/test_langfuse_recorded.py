"""Recorded-real-response replay for the Langfuse import adapter (LAY-3614, G5).

Drives a REAL ``LangfuseAdapter.import_traces`` against ``httpx.MockTransport``
serving captured Langfuse REST responses (the connect-validation list, the
import list, and a single-trace detail), with the adapter's own real
``httpx.Client`` deserialization in the loop. The adapter builds its client
internally (no injection kwarg), so — exactly like the offline recorder — we
swap the module's ``httpx`` for a shim that injects the MockTransport while
keeping the real ``httpx.Client``. This exercises the full path — real Langfuse
trace/observation JSON shape -> real adapter normaliser -> emitted
``model.invoke`` / ``cost.record`` — which the unit doubles (hand-built response
Mocks) never combine with a real Langfuse body.

The strong tell that the real shape flowed through: the imported
``model.invoke`` carries ``model="gpt-4o-mini"`` and the token counts
(``tokens_prompt=12`` / ``tokens_completion=6`` / ``tokens_total=18``) parsed off
the real observation's ``usage`` block (Langfuse's ``{input, output, total}``
form, which the adapter maps onto its flat token fields).
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

pytest.importorskip("httpx")

import layerlens.instrument.adapters.frameworks.langfuse as lf_mod  # noqa: E402
from layerlens.instrument.adapters.frameworks.langfuse import LangfuseAdapter  # noqa: E402

from .conftest import find_event, capture_framework_trace  # noqa: E402
from ..._recorded import load_recorded, mock_transport  # noqa: E402


class _HttpxShim:
    """Stand-in for the ``httpx`` module the adapter imports: its ``Client``
    injects the recorded MockTransport; everything else delegates to real httpx."""

    def __init__(self, transport: httpx.MockTransport) -> None:
        self._transport = transport

    def Client(self, **kwargs: Any) -> httpx.Client:
        kwargs.pop("transport", None)
        return httpx.Client(transport=self._transport, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(httpx, name)


class TestLangfuseRecorded:
    def test_import_over_recorded_langfuse(self, mock_client, monkeypatch):
        fixture = load_recorded("langfuse", "default")
        transport, _ = mock_transport(fixture)
        uploaded = capture_framework_trace(mock_client)

        # Route the adapter's internal httpx.Client through the recorded transport.
        monkeypatch.setattr(lf_mod, "httpx", _HttpxShim(transport))

        adapter = LangfuseAdapter(mock_client)
        adapter.connect(public_key="pk-test", secret_key="sk-test", host="https://langfuse.example")
        try:
            imported = adapter.import_traces(limit=1)
        finally:
            adapter.disconnect()

        assert imported == 1

        events = uploaded["events"]

        # Real Langfuse generation observation -> adapter -> model.invoke.
        mi = find_event(events, "model.invoke")
        assert mi["payload"]["framework"] == "langfuse"
        assert mi["payload"]["model"] == "gpt-4o-mini"
        assert mi["payload"]["tokens_prompt"] == 12
        assert mi["payload"]["tokens_completion"] == 6
        assert mi["payload"]["tokens_total"] == 18

        # cost.record mirrors the same real token accounting.
        cost = find_event(events, "cost.record")
        assert cost["payload"]["framework"] == "langfuse"
        assert cost["payload"]["model"] == "gpt-4o-mini"
        assert cost["payload"]["tokens_prompt"] == 12
        assert cost["payload"]["tokens_completion"] == 6
        assert cost["payload"]["tokens_total"] == 18
