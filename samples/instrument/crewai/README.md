# CrewAI instrumentation sample

End-to-end demo of `CrewAIAdapter` instrumenting a single-agent,
single-task crew via the CrewAI callback protocol (v0.41+,
`LayerLensCrewCallback`). The sample builds one `Agent` + one `Task`,
calls `adapter.instrument_crew(crew)`, and runs `kickoff()`. Each crew
kickoff emits LayerLens telemetry that ships to atlas-app via
`HttpEventSink`.

## Install

```bash
pip install 'layerlens[crewai,providers-openai]'
```

The `crewai` extra pins `crewai>=0.30,<0.90`. The `providers-openai` extra
adds `openai>=1.30,<2` (CrewAI's default LLM is OpenAI and honours the
standard `OPENAI_API_KEY` env var). The default `pip install layerlens`
does NOT pull either — that's the lazy-import guarantee tested by
`tests/instrument/test_lazy_imports.py`.

## Run

```bash
export OPENAI_API_KEY=sk-...
export LAYERLENS_STRATIX_API_KEY=ll-...   # optional
export LAYERLENS_STRATIX_BASE_URL=...     # optional
python -m samples.instrument.crewai.main
```

If `OPENAI_API_KEY` is unset the sample exits with code 2 (see
`main.py:30-32`). If `crewai` is not installed the sample exits with code
2 and points at the right install command (see `main.py:37-43`).

## Expected output

```text
Result: <result of the kickoff — for the prompt "What is 2 + 2?" the model returns "4">
Telemetry shipped. Check the LayerLens dashboard adapter health page.
```

`Result` is whatever `instrumented.kickoff()` returns (see `main.py:73`).
The task is configured with `expected_output="A single integer."` so the
model is steered to emit a one-token reply (see `main.py:64-68`).

## What this sample demonstrates

| Component | Where in `main.py` | What it proves |
|---|---|---|
| `HttpEventSink(adapter_name="crewai", path="/telemetry/spans")` | L46-51 | Telemetry transport is configured against the atlas-app spans endpoint with a 10-event batch and 1 s flush interval. |
| `CrewAIAdapter(capture_config=CaptureConfig.standard())` | L53 | Adapter is constructed using the standard capture profile. |
| `adapter.add_sink(sink)` + `adapter.connect()` | L54-55 | The sink is wired and the adapter lifecycle reaches `HEALTHY`. |
| `Agent(role="Math Tutor", ...)` + `Task(...)` + `Crew(agents=[...], tasks=[...])` | L57-69 | Demonstrates a minimal single-agent, single-task crew with delegation disabled (`allow_delegation=False`). |
| `adapter.instrument_crew(crew)` | L72 | The crew is wrapped via the CrewAI v0.41+ callback protocol so kickoff events are intercepted. |
| `instrumented.kickoff()` | L73 | A single crew run emits `agent.input`, `model.invoke`, and `agent.output` events (per the adapter's lifecycle docstring) — `tool.call` would also fire if the agent used a tool, but this sample defines none. |
| `sink.close()` + `adapter.disconnect()` | L76-77 | Adapter is torn down cleanly inside a `try/finally`. |

## Production note (multi-tenancy)

This sample does **not** pass `org_id` to `CrewAIAdapter`. In a
multi-tenant deployment every adapter must be scoped to a tenant — pass
`org_id=...` to the adapter (or via the `STRATIX` instance) so emitted
telemetry is correctly attributed. The omission here is acceptable for a
local single-tenant smoke test only.
