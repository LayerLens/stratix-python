"""Structural anti-reintroduction guard for per-run isolation (W2-followup / D1b).

The hazardous-five concurrency bug had ONE root cause: run state held in a
SCALAR ``self._collector`` (plus scalar span-id maps), so a second concurrent run
through the same adapter instance clobbered the first run's collector. The fix
moved every framework adapter onto per-run isolation, via one of:

* ContextVar ``RunState`` through ``_begin_run``/``_end_run`` (google_adk/strands/
  smolagents/langchain/langgraph/pydantic_ai/semantic_kernel/agentforce/
  bedrock_agents);
* a locked dict keyed by run id (openai_agents ``self._trace_runs``, llamaindex
  ``self._collectors``, crewai ``self._runs`` keyed by the event-bus lineage);
* a ``@property _collector`` that resolves the active run via ``_current_run`` and
  falls back to a documented, run-stateless ``self._fallback_collector`` slot used
  only by the ``__new__``-bypass cost test (crewai AND autogen — both dispatch
  handlers through a fresh ``contextvars.copy_context()`` so the run is bound
  per-callback; proven by test_concurrency_crewai.py / test_concurrency_autogen.py).

This is the standing risk: the NEXT framework adapter quietly reintroduces a
scalar ``self._collector`` that HOLDS run state. A structural guard catches that
without needing a flaky race to trigger it — "make the right thing the easy
thing, then pin it" (same flavor as the W6 emit guard). It scans framework
adapter source for a scalar ASSIGNMENT to ``self._collector`` — the write that
stores run state. The safe patterns are deliberately NOT matched: the keyed-dict
``self._collectors`` (plural), bare reads of the ``_collector`` property
(``c = self._collector``), the ``self._fallback_collector`` slot, and ``==``
comparisons.
"""

from __future__ import annotations

import re
from pathlib import Path

# test_..._run_state.py -> frameworks -> adapters -> instrument -> tests -> repo root
_FRAMEWORKS_DIR = Path(__file__).resolve().parents[4] / "src" / "layerlens" / "instrument" / "adapters" / "frameworks"

# Shared infra / non-adapter modules in the frameworks package.
_INFRA = {"__init__", "_base_framework", "_utils", "_handoff", "_langchain_memory"}

# Documented structural exceptions. Empty: every framework adapter is now on a
# per-run pattern (crewai AND autogen were moved onto the ``@property _collector``
# + ``_current_run`` + ``_fallback_collector`` pattern — no longer a residual). A
# new adapter cannot silently regress without tripping the guard below.
_ALLOWLIST: set[str] = set()

# A scalar collector ASSIGNMENT — ``self._collector = ...`` or an annotated
# ``self._collector: T = ...`` — i.e. run state stored in a plain instance
# attribute. ``self._collectors`` (keyed dict) does NOT match (the ``\b`` after
# "collector" stops it); property reads, ``self._fallback_collector``, and ``==``
# comparisons are excluded by requiring a single ``=`` immediately after.
_SCALAR_COLLECTOR = re.compile(r"self\._collector\b\s*(?::[^=\n]+)?=(?!=)")


def _framework_modules() -> list[Path]:
    return [p for p in sorted(_FRAMEWORKS_DIR.glob("*.py")) if p.stem not in _INFRA]


def test_no_framework_adapter_holds_run_state_in_a_scalar_collector() -> None:
    offenders = {
        p.stem: len(_SCALAR_COLLECTOR.findall(p.read_text()))
        for p in _framework_modules()
        if p.stem not in _ALLOWLIST and _SCALAR_COLLECTOR.search(p.read_text())
    }
    assert not offenders, (
        "framework adapter(s) assign run state to a scalar self._collector — concurrent "
        "runs will clobber each other's traces (LAY-3576 / D1b). Route run state through "
        "_begin_run/_end_run (ContextVar RunState), a locked dict keyed by run id, or a "
        "@property _collector delegating to _current_run (see smolagents.py / "
        f"openai_agents.py / crewai.py). Offenders: {offenders}"
    )


def test_guard_is_live_not_vacuous() -> None:
    """The scan must DETECT a scalar-collector assignment and must NOT match the
    safe per-run patterns — otherwise, now that every adapter is fixed, a
    mis-targeted scan would pass vacuously against a clean tree."""
    # Genuine anti-pattern — must bite:
    assert _SCALAR_COLLECTOR.search("        self._collector = TraceCollector(client, cfg)")
    assert _SCALAR_COLLECTOR.search("        self._collector: Optional[TraceCollector] = None")
    # Safe per-run patterns — must NOT trip:
    assert not _SCALAR_COLLECTOR.search("        c = self._collector")  # property read
    assert not _SCALAR_COLLECTOR.search("        self._collectors[key] = collector")  # keyed dict
    assert not _SCALAR_COLLECTOR.search("        self._fallback_collector = value")  # documented fallback slot
    assert not _SCALAR_COLLECTOR.search("        if self._collector == other:")  # comparison
