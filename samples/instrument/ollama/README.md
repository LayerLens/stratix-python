# Ollama adapter sample

This sample demonstrates the LayerLens Ollama provider adapter wrapping a real
`ollama.Client`. Every `chat`, `generate`, and `embeddings` call is intercepted
and turned into telemetry events. Ollama is a local-only inference engine, so:

- `api_cost_usd` is **always 0.0** — there's no API to bill for.
- `infra_cost_usd` is computed from `prompt_eval_duration + eval_duration`
  if you pass `cost_per_second` to the adapter constructor (handy for
  attributing GPU rental cost to specific calls).

## What you'll see

Running `python -m samples.instrument.ollama.main` (mocked mode, default)
produces two events for a single chat completion:

- `model.invoke` (L3) — request and response, with parameters, tokens,
  latency, the `method` (chat / generate / embeddings) and the
  `endpoint` Ollama was reached at.
- `cost.record` (cross-cutting) — `api_cost_usd: 0.0` always, plus
  `infra_cost_usd` if `cost_per_second` was set.

## Install

```bash
pip install 'layerlens[providers-ollama]'
```

The `providers-ollama` extra installs `ollama>=0.2`. The default
`pip install layerlens` does NOT pull `ollama` — that's the lazy-import
guarantee tested by `tests/instrument/test_lazy_imports.py`.

## Run mocked (no daemon required)

```bash
pip install 'layerlens[providers-ollama]' respx
python -m samples.instrument.ollama.main
```

The sample uses `respx` to mock the local Ollama HTTP endpoints
(`POST /api/pull`, `POST /api/chat`) so it runs cleanly in CI or on a
fresh checkout without an Ollama daemon. This is exactly the same
fixture pattern used by
`tests/instrument/adapters/providers/test_ollama_adapter.py`.

## Run live against a real `ollama serve` daemon

### macOS / Linux

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Start the daemon (binds 127.0.0.1:11434 by default)
ollama serve &

# 3. Pull the model the sample calls (~2 GB on disk)
ollama pull llama3.2:3b

# 4. Run the instrumented sample
LAYERLENS_OLLAMA_LIVE=1 python -m samples.instrument.ollama.main
```

### Windows

Download the installer from <https://ollama.com/download/windows> and
launch the `Ollama` app from the Start menu — it runs the daemon in the
background. Then in PowerShell:

```powershell
ollama pull llama3.2:3b
$env:LAYERLENS_OLLAMA_LIVE = "1"
python -m samples.instrument.ollama.main
```

### Choose a different model

Set `LAYERLENS_OLLAMA_MODEL` to any tag from the
[Ollama library](https://ollama.com/library):

```bash
LAYERLENS_OLLAMA_MODEL=mistral:7b LAYERLENS_OLLAMA_LIVE=1 \
  python -m samples.instrument.ollama.main
```

### Attribute GPU rental cost

If you're running Ollama on a paid GPU instance, pass your effective
per-second rate (e.g. `$0.005`/s for a small inference instance) and
the adapter computes `infra_cost_usd = total_compute_seconds * rate`
on every event:

```bash
LAYERLENS_OLLAMA_COST_PER_SECOND=0.005 LAYERLENS_OLLAMA_LIVE=1 \
  python -m samples.instrument.ollama.main
```

## GPU notes

Ollama auto-detects available accelerators at daemon start time:

| Platform                       | Backend                       |
|--------------------------------|-------------------------------|
| NVIDIA GPU (CUDA 11.8+, 12.x)  | CUDA — preferred when present |
| AMD GPU (ROCm 5.7+)            | ROCm                          |
| Apple Silicon (M1/M2/M3/M4)    | Metal                         |
| Intel / Apple Intel / no GPU   | CPU                           |

Use `ollama ps` to see which backend a loaded model is using and how
much GPU VRAM it's consuming. The adapter records `eval_duration` and
`prompt_eval_duration` from the daemon — those values are the source
of truth for "how long did this token cost", regardless of which
backend ran them.

To force CPU-only inference (e.g. to free GPU memory for other work):

```bash
OLLAMA_NUM_GPU=0 ollama serve
```

## Verify telemetry shape

The mocked-mode output looks like:

```text
[event  1] model.invoke: provider='ollama' model='llama3.2:3b' method='chat'
           endpoint='http://localhost:11434' prompt_tokens=18 completion_tokens=7
           total_tokens=25 latency_ms=0.0 finish_reason='stop'
[event  2] cost.record:  provider='ollama' model='llama3.2:3b' prompt_tokens=18
           completion_tokens=7 total_tokens=25 api_cost_usd=0.0
           infra_cost_usd=0.00175
```

Notice: `api_cost_usd=0.0` (Ollama is local) and `infra_cost_usd=0.00175`
(0.35 s of compute @ \$0.005/s).
