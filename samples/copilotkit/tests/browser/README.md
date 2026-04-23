# CopilotKit Evaluator -- Browser E2E Harness

End-to-end browser regression for the checkpointer fix in
`samples/copilotkit/agents/evaluator_agent.py`
(PR: `fix/copilotkit-interrupt-checkpointer`).

Drives the full stack -- React + `@copilotkit/react-ui` -> Next.js route
handler + `@copilotkit/runtime` -> FastAPI + `ag_ui_langgraph` -> patched
`evaluator_graph` -- through a real browser via Playwright and asserts
that the `interrupt()` pause resumes cleanly without the
`"Cannot send 'RUN_STARTED' while a run is still active"` lockup.

The Python-level wire test lives at
`tests/test_samples_e2e.py::TestCopilotKitSamples::test_copilotkit_evaluator_agui_wire`;
this harness is its browser-level counterpart.

## Layout

```
tests/browser/
├── backend/
│   ├── server.py             # FastAPI app; patches layerlens.Stratix to a fake
│   └── requirements.txt
├── frontend/
│   ├── package.json          # pinned: @copilotkit/* 1.56.3, next 16.2.4
│   ├── next.config.js
│   ├── tsconfig.json
│   ├── playwright.config.ts  # spawns `next dev`; runs globalSetup/Teardown
│   └── app/
│       ├── layout.tsx        # wraps children in <CopilotKit ... agent="evaluator">
│       ├── page.tsx          # renders <CopilotChat />
│       ├── globals.css
│       └── api/copilotkit/route.ts  # LangGraphHttpAgent -> http://127.0.0.1:8123/evaluator
│   └── tests/
│       ├── globalSetup.ts    # spawns `uvicorn server:app` on :8123, polls /healthz
│       ├── globalTeardown.ts # kills the uvicorn child
│       ├── interrupt-resume.spec.ts
│       └── tsconfig.json
└── .gitignore
```

## Version pinning

All three `@copilotkit/*` packages are pinned to **`1.56.3`** (exact, no
caret/tilde) to match the version reported in the original bug report.
`next` is pinned to **`16.2.4`**, which was the latest stable at the
time the harness was built. If any of these ever fail to resolve on
npm, bump to the nearest working patch release and note it here.

`layerlens` and `ag-ui-langgraph` on the Python side follow the ranges
declared in `samples/copilotkit/README.md`; the harness runs against a
`MagicMock` Stratix client so no real API key is required.

## Running

```bash
# 1. Install Python deps for the backend (in a venv you also use for the SDK).
pip install -r backend/requirements.txt

# 2. Install frontend deps
cd frontend
npm install

# 3. Install the Chromium Playwright browser bundle
npx playwright install chromium

# 4. Run the test suite (launches FastAPI on :8123, next dev on :3000,
#    drives Chromium, asserts no RUN_STARTED lockup).
#    HARNESS_PYTHON must point at a venv that has the backend deps.
HARNESS_PYTHON=/path/to/venv/bin/python npx playwright test

# Useful for debugging the chat flow end-to-end:
npx playwright test --headed --debug
```

## Known limitation: headless CopilotChat textarea driver

CopilotChat 1.56's textarea reports as ``aria-hidden`` / non-"visible" in
Playwright's strict actionability checks in **headless** Chromium, and
none of the standard input-driving patterns we tried
(``fill``, ``keyboard.type`` + Enter, ``pressSequentially``, DOM
``value``-setter + bubbled ``input`` event) reliably enable the Send
button in that mode. The harness **does** drive the chat correctly when
run with ``--headed``, which is sufficient for a human reviewing the
fix. Automated headless execution currently needs one of:

1. CopilotKit-internal test hooks (e.g. a ``data-testid`` on the hidden
   submit or a programmatic ``sendMessage(...)`` handle). File an issue
   upstream if you need automated headless coverage.
2. Patching ``CopilotChat`` to render without the ``aria-hidden``
   wrapper, or swapping to a custom chat surface that uses
   ``useCopilotChat`` directly.
3. Driving the ``/api/copilotkit`` GraphQL endpoint by HTTP rather than
   through the UI (this is effectively what the Python-level AG-UI wire
   test at ``tests/test_samples_e2e.py::test_copilotkit_evaluator_agui_wire``
   already does).

**The authoritative regression coverage for the checkpointer fix lives
in the Python test suite** (the unit-level ``interrupt_resume`` test and
the AG-UI wire test). This browser harness is corroborating evidence
that customers who follow the sample can reach the same end-to-end
success under a real CopilotKit runtime; its value is demonstrative,
not gate-keeping.

### Running against an already-running backend

If you are iterating on `backend/server.py` in a separate shell, the
Playwright `globalSetup` will detect an existing healthy backend on
`http://127.0.0.1:8123/healthz` and skip spawning a new one. In that
case `globalTeardown` will leave your process alone.

### Environment overrides

| Variable | Purpose | Default |
| --- | --- | --- |
| `HARNESS_HOST` | Bind address for uvicorn when run directly | `127.0.0.1` |
| `HARNESS_PORT` | Port for uvicorn when run directly | `8123` |
| `HARNESS_PYTHON` | Python executable used by `globalSetup.ts` | `python3` on POSIX, `python` on Windows |
| `EVALUATOR_BACKEND_URL` | URL the Next route handler proxies to | `http://127.0.0.1:8123/evaluator` |

## What the test proves

The Playwright spec drives three turns of the CopilotChat UI:

1. **"evaluate my traces"** -- kicks off the graph. Expects visible
   assistant text mentioning `Helpfulness`/`judge`/`which judge should i
   use`.
2. **"ok"** -- resumes the interrupted run. Expects `using judge
   Helpfulness`.
3. **"thanks"** -- starts a brand-new run after the resume finished.

After every turn the test asserts the exact string
`Cannot send 'RUN_STARTED' while a run is still active` does NOT appear
in either the visible DOM or the browser console (captured via
`page.on('console')` + `page.on('pageerror')`). If it does, the test
fails loudly with the matching message text.
