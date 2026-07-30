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
- `cost.record` with `cost_usd: None` — Ollama is self-hosted and has no
  per-token API price, so the model is intentionally unpriced. Token counts
  are still emitted so downstream cost analytics can attribute infra
  separately.
- When `cost_per_second` is configured (see below), each `model.invoke`
  payload includes `infra_cost_usd` derived from `eval_duration +
  prompt_eval_duration` × the configured rate.

## Pricing

Ollama models have no public API price — inference is local and there is no
per-token pricing table for it. The adapter therefore leaves the model
**unpriced**: `cost.record.cost_usd` is `None` (not a fabricated `0.0`),
which honestly signals "no per-token API cost was charged" rather than
implying a real zero-dollar billed call.

Callers who want to attribute hardware/compute cost can pass an optional
`cost_per_second` to the constructor:

```python
OllamaProvider(cost_per_second=0.0001)   # $0.0001/sec of GPU/CPU time
```

This attributes compute time as `infra_cost_usd` on each `model.invoke`
event, computed as `(eval_duration + prompt_eval_duration) / _NS_PER_SECOND
* cost_per_second` (the two durations are the response's nanosecond
`eval_duration` and `prompt_eval_duration`; `_NS_PER_SECOND` is `1e9`).
`infra_cost_usd` is distinct from the (unpriced) `cost.record.cost_usd`.
Rough rule of thumb for a hosted GPU at ~$0.50/hr: ~$0.000139/sec.

## Sample

[`samples/instrument/ollama/example.py`](../../../samples/instrument/ollama/example.py)

## Compat

- `ollama>=0.1`
- Python 3.9+
