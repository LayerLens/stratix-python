# CrewAI sample

Instruments a one-task CrewAI crew with `CrewAIAdapter` and prints every
event the adapter emits via a tiny in-process sink.

## Install

```bash
pip install 'layerlens[crewai]' openai
```

CrewAI 0.30+ requires Python 3.10 or newer.

## Run

```bash
export OPENAI_API_KEY=sk-...
python -m samples.instrument.crewai.main
```

## What you'll see

```
[event] environment.config
[event] agent.input
[event] agent.code
[event] model.invoke
[event] agent.state.change
[event] agent.output

Crew result: 4

Captured 6 event(s):
  - environment.config
  - agent.input
  - agent.code
  - model.invoke
  - agent.state.change
  - agent.output
```

The exact event count depends on whether the LLM ends up using tools,
delegating to other agents, or completing the task in one step.

## Shipping events to atlas-app

Replace `_PrintSink` in `main.py` with a real transport sink
(`HttpEventSink` / `OTLPHttpSink`) once they land in a sibling M2/M3 PR.
The adapter's sink dispatch is already wired — only the transport
wrapper differs.

## Files

- `main.py` — sample entrypoint
- `__init__.py` — package marker
