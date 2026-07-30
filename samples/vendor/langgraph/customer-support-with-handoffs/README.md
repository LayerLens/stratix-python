# Vendor fork — Customer Support with Handoffs (LangGraph × LayerLens)

A **genuine vendor fork** of the upstream LangChain / LangGraph tutorial
*"Build customer support with handoffs"*, instrumented **non-invasively** with the
LayerLens `LangGraphCallbackHandler`. This is the SDK's reference **instrumented
vendor-fork** kind — a real fork of an upstream example app, wired to our adapter,
proving a real multi-agent DAG end-to-end.

See [`UPSTREAM`](./UPSTREAM) for full provenance (tutorial URL, fork repo + commit,
license, and the exact list of changes vs upstream).

## Topology

```
START → triage → (billing | technical | returns)_specialist → closer → END
```

`triage` classifies the request and routes to one specialist; the specialist uses
its department tools (lookup_invoice/issue_credit, run_diagnostic/create_ticket,
check_return_eligibility/start_return) and resolves; `closer` confirms. Each named
node transition is a real `agent.handoff` the LayerLens adapter emits — so the
server renders a **multi-agent DAG** (`triage → <dept>_specialist → closer`).

## Layout

```
customer-support-with-handoffs/
  UPSTREAM               provenance (upstream tutorial + fork commit + changes)
  app.py                 the forked LangGraph graph, agents, tools, model wiring
  run_instrumented.py    non-invasive LayerLens capture → upload → server-DAG read-back
  requirements.txt       langgraph + langchain-openai (+ layerlens)
```

> The upstream LayerLens fork also ships a full-stack CopilotKit UX (Next.js +
> FastAPI AG-UI). This SDK reference ports only the **console** surface — the
> self-contained instrumented sample. See the fork repo in `UPSTREAM` for the UX.

## Run it (real graph run + upload + server DAG read-back)

```bash
# from the SDK repo root, in a venv that has langgraph + langchain-openai + layerlens
set -a; . tests/e2e/live/.env; set +a
export VENDOR_MODEL_BACKEND=openai          # this env has no OpenRouter credential
python samples/vendor/langgraph/customer-support-with-handoffs/run_instrumented.py
```

- With a model key **and** a LayerLens key + base URL, it runs the real graph,
  uploads each conversation's trace (LayerLens key only), and reads the
  **server-computed agent DAG** back — asserting `≥2` nodes and `≥1` edge.
- With only a model key it runs **capture-only** and prints a BLOCKED upload
  notice — never a fabricated success.
- With no model key it prints BLOCKED and runs nothing — no fake success.

Model backend: `VENDOR_MODEL_BACKEND=openrouter` (upstream default, needs
`OPENROUTER_API_KEY`) or `openai` (needs `OPENAI_API_KEY`).
