"""FastAPI server for the CopilotKit + LayerLens evaluator sample.

Wires the sample's ``evaluator_graph`` (from
``samples/copilotkit/agents/evaluator_agent.py``) into a FastAPI app
via ``ag_ui_langgraph.add_langgraph_fastapi_endpoint`` at path
``/evaluator``.

Real LayerLens only. There is no fake/fixture path: a missing or empty
``LAYERLENS_STRATIX_API_KEY`` is a hard error at startup, and the agent
talks to the real ``layerlens.Stratix`` client.

Run with::

    python server.py

or::

    uvicorn server:app --host 127.0.0.1 --port 8123

The Next.js frontend at ``samples/copilotkit/app/frontend/`` proxies
``/api/copilotkit`` to ``EVALUATOR_BACKEND_URL`` (defaults to
``http://127.0.0.1:8123/evaluator``).
"""

from __future__ import annotations

import os
import sys
import importlib.util
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# Auto-load credentials from gitignored .env files so a developer running
# the harness locally doesn't have to manually export env vars. Looks at
# ``samples/copilotkit/.env`` first (sample-local), then ``tests/.env``
# (repo-level). Values already in the environment win.
# ---------------------------------------------------------------------------

_THIS = Path(__file__).resolve()
_SAMPLE_ROOT = _THIS.parent.parent.parent  # backend -> app -> copilotkit
_REPO_ROOT = _SAMPLE_ROOT.parent.parent  # copilotkit -> samples -> repo
for env_path in [
    _SAMPLE_ROOT / ".env",
    _REPO_ROOT / "tests" / ".env",
]:
    if not env_path.is_file():
        continue
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# ---------------------------------------------------------------------------
# Real-API guardrail. No fake fixtures, no MagicMock, no silent fallback.
# ---------------------------------------------------------------------------

_real_key = os.environ.get("LAYERLENS_STRATIX_API_KEY", "").strip()
if not _real_key:
    raise RuntimeError(
        "LAYERLENS_STRATIX_API_KEY is required. Set it in your environment "
        "or in samples/copilotkit/.env / tests/.env. This sample does not "
        "ship a fake/fixture mode — it exercises the real LayerLens API."
    )

masked = _real_key[:4] + "…" + _real_key[-4:] if len(_real_key) > 10 else "<short>"
print(f"[harness] Stratix client: REAL (key={masked}).", flush=True)


# ---------------------------------------------------------------------------
# Import the evaluator agent from ``samples/copilotkit/agents``.
# Loaded by absolute path so the server works regardless of CWD, and
# stashed in ``sys.modules`` under a unique name so it does not clash
# with other tests that load the same module.
# ---------------------------------------------------------------------------

# backend/server.py -> app/ -> copilotkit/ -> agents/
AGENT_FILE = _SAMPLE_ROOT / "agents" / "evaluator_agent.py"
if not AGENT_FILE.exists():
    raise RuntimeError(f"Could not locate evaluator_agent.py at {AGENT_FILE}")

_MOD_NAME = "copilotkit_evaluator_app"
_spec = importlib.util.spec_from_file_location(_MOD_NAME, str(AGENT_FILE))
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Failed to build import spec for {AGENT_FILE}")
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_MOD_NAME] = _mod
_spec.loader.exec_module(_mod)

evaluator_graph = _mod.evaluator_graph


# ---------------------------------------------------------------------------
# Build the FastAPI app and wire the AG-UI endpoint.
# ---------------------------------------------------------------------------

from ag_ui_langgraph import add_langgraph_fastapi_endpoint  # noqa: E402

app = FastAPI(title="copilotkit-evaluator-harness")

# The Next.js runtime proxies to this server from its server-side route
# handler (not from the browser), so CORS is not strictly required. The
# permissive policy here lets direct browser probes work for diagnostics.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Readiness probe."""
    return {"status": "ok"}


@app.get("/evaluations/{evaluation_id}")
def get_evaluation(evaluation_id: str) -> dict:
    """Out-of-band polling endpoint for the frontend.

    The agent kicks off evaluations and ends without waiting for them
    to complete (real LayerLens evaluations take 30+ seconds, far
    longer than a chat turn should block). The frontend then polls
    this endpoint every few seconds and folds completed verdicts into
    the canvas — so the user sees "pending" become "passed/failed" as
    each evaluation finishes, without the agent burning recursion or
    hallucinating about state it can't observe.
    """
    from layerlens import Stratix

    client = Stratix()
    ev = client.trace_evaluations.get(evaluation_id)
    if ev is None:
        return {"evaluation_id": evaluation_id, "status": "not_found"}
    status = ev.status.value if hasattr(ev.status, "value") else str(ev.status)
    base = {
        "evaluation_id": evaluation_id,
        "status": status,
        "trace_id": getattr(ev, "trace_id", "") or "",
        "judge_id": getattr(ev, "judge_id", "") or "",
    }
    if status != "success":
        return base
    details = client.trace_evaluations.get_results(id=evaluation_id)
    if details is None or details.score is None:
        return base
    return {
        **base,
        "passed": bool(details.passed),
        "score": float(details.score),
        "reasoning": details.reasoning,
    }


# Uses the sample's ``build_agui_agent`` factory so the small runId
# workaround for ag-ui-protocol/ag-ui#1582 is applied.
add_langgraph_fastapi_endpoint(
    app,
    agent=_mod.build_agui_agent(name="evaluator", graph=evaluator_graph),
    path="/evaluator",
)


def main() -> None:
    """Entry point: boot uvicorn against the FastAPI app.

    Honours ``HARNESS_HOST`` / ``HARNESS_PORT`` / ``HARNESS_LOG_LEVEL``
    env vars; defaults are 127.0.0.1:8123 and ``info``-level logging.
    """
    import uvicorn

    uvicorn.run(
        "server:app",
        host=os.environ.get("HARNESS_HOST", "127.0.0.1"),
        port=int(os.environ.get("HARNESS_PORT", "8123")),
        log_level=os.environ.get("HARNESS_LOG_LEVEL", "info"),
        reload=False,
    )


if __name__ == "__main__":
    main()
