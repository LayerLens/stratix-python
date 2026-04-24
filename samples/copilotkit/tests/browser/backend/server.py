"""FastAPI harness that serves the patched CopilotKit evaluator graph.

This server is used by the Playwright browser test under
``samples/copilotkit/tests/browser/``. It does two things:

1. Patches ``layerlens.Stratix`` BEFORE the evaluator agent module is
   imported, so that ``_get_client()`` returns a fake client with one
   judge, one trace, and a "success" trace-evaluation. This keeps the
   run deterministic and fast (no network, no real API key).
2. Wires the patched ``evaluator_graph`` into a FastAPI app via
   ``ag_ui_langgraph.add_langgraph_fastapi_endpoint`` at path
   ``/evaluator``, matching the ``tests/test_samples_e2e.py::
   test_copilotkit_evaluator_agui_wire`` wire test.

Run with::

    python server.py

or equivalently::

    uvicorn server:app --host 127.0.0.1 --port 8123

The Playwright ``globalSetup`` starts this via ``uvicorn`` so that the
process can be cleanly terminated on teardown.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


# ---------------------------------------------------------------------------
# Build a fake Stratix client BEFORE importing the evaluator agent.
#
# This mirrors the pattern used by
# ``tests/test_samples_e2e.py::test_copilotkit_evaluator_agui_wire`` so the
# browser test sees the exact same deterministic fixture shape.
# ---------------------------------------------------------------------------

_fake_judge = SimpleNamespace(
    id="jdg_1",
    name="Helpfulness",
    evaluation_goal="measures helpfulness",
    created_at="2026-04-23T00:00:00Z",
)
_fake_trace = SimpleNamespace(
    id="trc_1",
    filename="sample.jsonl",
    created_at="2026-04-23T00:00:00Z",
)
_fake_eval = SimpleNamespace(
    id="ev_1",
    trace_id="trc_1",
    judge_id="jdg_1",
    # status.value is read by EvaluationInfo; "success" short-circuits the
    # poll loop so the run finishes quickly.
    status=SimpleNamespace(value="success"),
)
_fake_results = SimpleNamespace(score=0.9, passed=True, reasoning="ok")

_FAKE_CLIENT = MagicMock()
_FAKE_CLIENT.judges.get_many.return_value = SimpleNamespace(judges=[_fake_judge])
_FAKE_CLIENT.traces.get_many.return_value = SimpleNamespace(traces=[_fake_trace])
_FAKE_CLIENT.trace_evaluations.create.return_value = _fake_eval
_FAKE_CLIENT.trace_evaluations.get.return_value = _fake_eval
_FAKE_CLIENT.trace_evaluations.get_results.return_value = _fake_results


# ---------------------------------------------------------------------------
# Patch layerlens.Stratix BEFORE importing the evaluator module.
# ---------------------------------------------------------------------------

os.environ.setdefault("LAYERLENS_STRATIX_API_KEY", "test-key")

import layerlens  # noqa: E402  (import after setting the env var)

layerlens.Stratix = MagicMock(return_value=_FAKE_CLIENT)  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Import the evaluator agent from ``samples/copilotkit/agents``.
#
# We load it by absolute path so the server works regardless of where the
# test is invoked from, and we stash it into ``sys.modules`` under a unique
# name so it does not clash with other e2e tests that also load the module.
# ---------------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
# backend/server.py -> browser/ -> tests/ -> copilotkit/ -> agents/
AGENT_FILE = (
    THIS_FILE.parent.parent.parent.parent / "agents" / "evaluator_agent.py"
)
if not AGENT_FILE.exists():  # pragma: no cover - dev guardrail
    raise RuntimeError(f"Could not locate evaluator_agent.py at {AGENT_FILE}")

_MOD_NAME = "copilotkit_evaluator_browser_harness"
_spec = importlib.util.spec_from_file_location(_MOD_NAME, str(AGENT_FILE))
if _spec is None or _spec.loader is None:  # pragma: no cover
    raise RuntimeError(f"Failed to build import spec for {AGENT_FILE}")
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_MOD_NAME] = _mod
_spec.loader.exec_module(_mod)

# Belt-and-suspenders: also overwrite the module-level client ref so that
# any code path that cached ``_client`` before our patch still sees the fake.
_mod._client = _FAKE_CLIENT

evaluator_graph = _mod.evaluator_graph


# ---------------------------------------------------------------------------
# Build the FastAPI app and wire the AG-UI endpoint.
# ---------------------------------------------------------------------------

from ag_ui_langgraph import add_langgraph_fastapi_endpoint  # noqa: E402
from copilotkit import LangGraphAGUIAgent  # noqa: E402

app = FastAPI(title="copilotkit-evaluator-browser-harness")

# The Next.js runtime proxies to this server from its server-side route
# handler (not from the browser), so CORS is not strictly required. We add
# a permissive policy anyway so direct browser probes from Playwright work.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Simple readiness probe used by the Playwright globalSetup."""
    return {"status": "ok"}


# The evaluator graph uses ``langchain.agents.create_agent`` + a
# frontend HITL tool (``confirm_judge``). No ``interrupt()`` path means
# we don't need the private-API workaround subclass that previous
# revisions of this harness used -- plain ``LangGraphAGUIAgent`` works.
add_langgraph_fastapi_endpoint(
    app,
    agent=LangGraphAGUIAgent(name="evaluator", graph=evaluator_graph),
    path="/evaluator",
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host=os.environ.get("HARNESS_HOST", "127.0.0.1"),
        port=int(os.environ.get("HARNESS_PORT", "8123")),
        log_level=os.environ.get("HARNESS_LOG_LEVEL", "info"),
        reload=False,
    )
