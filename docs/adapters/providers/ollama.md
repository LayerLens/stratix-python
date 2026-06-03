# Ollama provider adapter

Instruments local [Ollama](https://ollama.com) inference via
monkey-patching the Ollama Python SDK. Captures token usage,
`done_reason` → `finish_reason`, response model id, total/eval durations,
and (optionally) attributes compute time as infra cost.

## Install

```bash
pip install layerlens[providers-ollama]
```

Pulls `ollama>=0.1`. The `ollama` extra is kept as an alias for prior
installs.

## `ollama serve` setup

Ollama is local-only by default. The adapter talks to whatever endpoint
the SDK is pointed at — no auth, just an HTTP daemon.

1. **Install the Ollama runtime** for your platform:
   `https://ollama.com/download`
2. **Pull a model**:

   ```bash
   ollama pull llama3
   ```

3. **Start the daemon** (most installers do this automatically):

   ```bash
   ollama serve
   ```

   By default this listens on `http://localhost:11434`.

4. **(Optional) Point at a remote box** by setting `OLLAMA_HOST`:

   ```bash
   export OLLAMA_HOST="http://my-ollama-box:11434"
   ```

   The adapter reads this on `connect()` and emits it as `endpoint` on
   every `model.invoke` event so you can split traces by daemon.

## Usage

```python
import ollama
from layerlens.instrument.adapters.providers import OllamaProvider

client = ollama.Client()                        # honours OLLAMA_HOST
provider = OllamaProvider(cost_per_second=0.0001)  # optional, see below
provider.connect(client)                        # patches chat / generate / embeddings / embed

response = client.chat(
    model="llama3",
    messages=[{"role": "user", "content": "Hi"}],
)
print(response["message"]["content"])
```

## Event surface

- `model.invoke` for `chat`, `generate`, `embeddings`, and `embed` calls.
  Payload includes `model`, `usage` (from `prompt_eval_count` +
  `eval_count`), `finish_reason` (from `done_reason`), `endpoint`, and
  `duration_ms` (from `total_duration`).
- `cost.record` with `cost_usd: 0.0` since Ollama is self-hosted; token
  counts are still emitted so downstream cost analytics can attribute infra
  separately.
- When `cost_per_second` is configured (see below), each `model.invoke`
  payload includes `infra_cost_usd` derived from `eval_duration +
  prompt_eval_duration` × the configured rate.

## Pricing

Ollama models have no public API price — inference is local. The adapter
treats Ollama API cost as `$0.00` and surfaces zero-cost entries through
the standard cost-record path. Callers who want to attribute hardware
cost can pass:

```python
OllamaProvider(cost_per_second=0.0001)   # $0.0001/sec of GPU/CPU time
```

This computes `infra_cost_usd = (eval_ns + prompt_eval_ns) / 1e9 *
cost_per_second` for each invoke and includes it in the `model.invoke`
payload. Rough rule of thumb for a hosted GPU at ~$0.50/hr: ~$0.000139/sec.

## Sample

[`samples/instrument/ollama/example.py`](../../../samples/instrument/ollama/example.py)

## Compat

- `ollama>=0.1`
- Python 3.9+
