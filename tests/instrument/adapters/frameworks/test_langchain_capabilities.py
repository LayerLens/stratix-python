"""Capability regression tests for the LangChain framework adapter.

PR #119 (brand leak + capability declarations) wired REPLAY into the
adapters that lived on its branch but deferred LangChain because it
lives on the orchestration source-port branch (PR #96). This test
file is the regression guard for the closure: REPLAY must be declared
because ``LayerLensCallbackHandler.serialize_for_replay`` returns a
non-stub ``ReplayableTrace``, and STREAMING must NOT be declared
because the adapter does not register an ``on_llm_new_token`` callback
(no per-chunk events flow through the adapter — see callbacks.py).

Per CLAUDE.md 'no fake claims', a capability is only declared if the
adapter actually implements it.
"""

from __future__ import annotations

from layerlens.instrument.adapters._base.adapter import AdapterCapability
from layerlens.instrument.adapters.frameworks.langchain.callbacks import (
    LayerLensCallbackHandler,
)


def test_declares_replay_capability() -> None:
    handler = LayerLensCallbackHandler()
    caps = handler.info().capabilities
    assert AdapterCapability.REPLAY in caps


def test_does_not_declare_streaming_capability() -> None:
    """LangChain adapter has no ``on_llm_new_token`` callback — no
    per-chunk events flow through it. STREAMING must stay undeclared
    until ``on_llm_new_token`` is wired (and tested) explicitly."""
    handler = LayerLensCallbackHandler()
    caps = handler.info().capabilities
    assert AdapterCapability.STREAMING not in caps


def test_get_adapter_info_matches_info_wrapper() -> None:
    """``info()`` (BaseAdapter wrapper) and ``get_adapter_info()``
    (subclass override) must report identical capability lists so the
    catalog manifest emitter sees the same answer regardless of which
    entrypoint it calls."""
    handler = LayerLensCallbackHandler()
    assert handler.info().capabilities == handler.get_adapter_info().capabilities
