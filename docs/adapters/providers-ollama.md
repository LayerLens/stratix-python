# Ollama provider adapter

`layerlens.instrument.adapters.providers.ollama_adapter.OllamaAdapter`
instruments the Ollama Python SDK for local LLM inference.

## Install

```bash
pip install 'layerlens[providers-ollama]'
```

Pulls `ollama>=0.2`.

## Quick start

```python
import ollama
from layerlens.instrument.adapters.providers.ollama_adapter import OllamaAdapter
from layerlens.instrument.transport.sink_http import HttpEventSink

sink = HttpEventSink(adapter_name="ollama")
adapter = OllamaAdapter(cost_per_second=0.005)  # optional infra cost
adapter.add_sink(sink)
adapter.connect()

# Wrap the ollama module — it acts as a global client.
adapter.connect_client(ollama)

ollama.chat(model="llama3.1", messages=[{"role": "user", "content": "Hi"}])
```

## Ollama-specific behavior

- **`api_cost_usd: 0.0`** is always emitted because Ollama runs locally — there's
  no API to bill for.
- **Optional `infra_cost_usd`**: if you pass `cost_per_second` to the adapter
  constructor, the adapter sums `prompt_eval_duration` + `eval_duration` (both
  in nanoseconds) and computes `total_seconds * cost_per_second`. Useful for
  attributing GPU rental cost to specific LLM calls.
- **Endpoint capture**: `OLLAMA_HOST` env var (or `http://localhost:11434`) is
  recorded in every event so you can identify which Ollama instance handled a
  request.
- **Three methods wrapped**: `chat`, `generate`, and `embeddings`. The
  `method` field in `model.invoke.metadata` distinguishes them.

## Token extraction

Ollama responses (dict or object form) expose `prompt_eval_count` and
`eval_count` — these map to `prompt_tokens` and `completion_tokens` in
`NormalizedTokenUsage`.
