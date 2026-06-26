"""Structural anti-reintroduction guard for per-run isolation (W2-followup / D1b).

The hazardous-five concurrency bug had ONE root cause: run state held in a
SCALAR ``self._collector`` (plus scalar span-id maps), so a second concurrent run
through the same adapter instance clobbered the first run's collector. The fix
moved every framework adapter onto per-run isolation — either ContextVar
``RunState`` via ``_begin_run``/``_end_run`` (crewai/google_adk/strands/smolagents/
langchain/langgraph/pydantic_ai/semantic_kernel/agentforce/bedrock_agents) or a
locked dict keyed by run id (openai_agents ``self._trace_runs``, llamaindex
``self._collectors``).

This is the standing risk: the NEXT framework adapter quietly reintroduces a
scalar ``self._collector``. A structural guard catches that without needing a
flaky race to trigger it — "make the right thing the easy thing, then pin it"
(same flavor as the W6 emit guard). It scans framework adapter source for the
scalar ``self._collector`` attribute (``self._collectors`` plural — the safe
keyed-dict pattern — is deliberately NOT matched).

ALLOWLIST: one documented structural exception, with a PROVEN distinct reason
(see its concurrency test's xfail), tracked as the W2 residual — a new adapter
cannot silently join this allowlist:

* ``autogen`` — instruments via a global logging handler, not a per-run callback,
  so events arrive outside any run call-stack and carry no per-conversation key
  on llm/tool events, with no per-conversation flush boundary.

(``crewai`` was previously allowlisted for the same class of bug — its typed
event bus dispatches every handler through a FRESH ``contextvars.copy_context()``
on a thread-pool worker, so the ContextVar ``RunState`` migration is impossible.
It is now FIXED (LAY-3576 / D1b) by a locked per-run map keyed by the event
LINEAGE the bus stamps before dispatch — ``self._runs`` in crewai.py, the same
keyed-dict family as openai_agents — so it no longer holds a scalar collector and
is no longer allowlisted.)
"""

from __future__ import annotations

import re
from pathlib import Path

# test_..._run_state.py -> frameworks -> adapters -> instrument -> tests -> repo root
_FRAMEWORKS_DIR = Path(__file__).resolve().parents[4] / "src" / "layerlens" / "instrument" / "adapters" / "frameworks"

# Shared infra / non-adapter modules in the frameworks package.
_INFRA = {"__init__", "_base_framework", "_utils", "_handoff", "_langchain_memory"}

# The single documented structural exception (see module docstring + the autogen
# concurrency xfail). Keep this set tiny and justified.
_ALLOWLIST = {"autogen"}

# Scalar collector attribute. ``\b`` after "collector" means ``self._collectors``
# (the safe keyed-dict pattern) does NOT match.
_SCALAR_COLLECTOR = re.compile(r"self\._collector\b")


def _framework_modules() -> list[Path]:
    return [p for p in sorted(_FRAMEWORKS_DIR.glob("*.py")) if p.stem not in _INFRA]


def test_no_framework_adapter_holds_run_state_in_a_scalar_collector() -> None:
    offenders = {
        p.stem: len(_SCALAR_COLLECTOR.findall(p.read_text()))
        for p in _framework_modules()
        if p.stem not in _ALLOWLIST and _SCALAR_COLLECTOR.search(p.read_text())
    }
    assert not offenders, (
        "framework adapter(s) hold run state in a scalar self._collector — concurrent "
        "runs will clobber each other's traces (LAY-3576 / D1b). Route run state through "
        "_begin_run/_end_run (ContextVar RunState) or a locked dict keyed by run id "
        f"(see smolagents.py / openai_agents.py). Offenders: {offenders}"
    )


def test_guard_is_live_not_vacuous() -> None:
    """The scan must actually detect the pattern (the allowlisted autogen still
    holds a scalar collector), or a mis-targeted scan would pass vacuously."""
    autogen = _FRAMEWORKS_DIR / "autogen.py"
    assert _SCALAR_COLLECTOR.search(autogen.read_text()), (
        "scan found no scalar self._collector even in the allowlisted autogen — "
        "the pattern/path is mis-targeted and the guard would never bite"
    )
