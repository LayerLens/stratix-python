# Ollama provider adapter

`layerlens.instrument.adapters.providers.ollama_adapter.OllamaAdapter`
instruments the [Ollama Python SDK](https://github.com/ollama/ollama-python)
for local LLM inference. Ollama is a self-hosted runtime (no cloud API) so
the adapter records all token usage but reports `api_cost_usd: 0.0` —
the only billable resource is the operator's own compute.

## Status

| Field         | Value                                      |
|---------------|--------------------------------------------|
| Adapter type  | LLM provider                               |
| Framework     | `ollama`                                   |
| SDK pin       | `ollama>=0.2`                              |
| Adapter ver.  | `0.1.0`                                    |
| Local-only    | Yes — default endpoint `http://localhost:11434` |
| Pricing       | All models recorded as `0.0` USD/token (self-hosted) |
| GA milestone  | M3 (LLM provider fan-out)                  |

## Install

```bash
pip install 'layerlens[providers-ollama]'
```

The `providers-ollama` extra pulls `ollama>=0.2`. The default
`pip install layerlens` does NOT pull `ollama` — adapter modules and
their vendor SDKs are loaded lazily on first use.

## `ollama serve` setup

The adapter wraps an in-process `ollama.Client`, but the client itself
talks HTTP to a daemon (`ollama serve`) running locally or remotely.
You need to install + start the daemon BEFORE running any
instrumented code.

### macOS

```bash
brew install ollama
brew services start ollama          # background service
# or, foreground for debugging:
ollama serve
```

### Linux

```bash
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl enable --now ollama   # systemd service on most distros
# or, foreground:
ollama serve
```

### Windows

Download the installer from <https://ollama.com/download/windows> and
launch the **Ollama** app from the Start menu — the installer registers
a Windows service that runs the daemon in the background. Verify with:

```powershell
Get-Service Ollama
curl.exe http://localhost:11434/api/version
```

### Docker / Compose

```yaml
services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama-models:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
volumes:
  ollama-models:
```

### Verify the daemon is up

```bash
curl http://localhost:11434/api/version
# {"version":"0.6.0"}
```

### Pull at least one model

The first call to `client.chat(model=X)` blocks while Ollama downloads
`X` (multiple GB). Pre-pull models with:

```bash
ollama pull llama3.2:3b      # ~2 GB, fast on CPU, fits on 4 GB GPU
ollama pull llama3.1:8b      # ~5 GB, recommended baseline
ollama pull qwen2.5:7b       # ~4 GB, strong open model
ollama pull nomic-embed-text # ~270 MB, embeddings only
```

## Quick start

```python
from ollama import Client
from layerlens.instrument.adapters._base import CaptureConfig
from layerlens.instrument.adapters.providers.ollama_adapter import OllamaAdapter

adapter = OllamaAdapter(
    capture_config=CaptureConfig.standard(),
    cost_per_second=0.005,  # optional infra-cost rate
)
adapter.connect()

client = Client()  # reads OLLAMA_HOST env var, defaults to http://localhost:11434
adapter.connect_client(client)

response = client.chat(
    model="llama3.1",
    messages=[{"role": "user", "content": "Hi"}],
)
print(response.message.content)
```

Each instrumented call emits two events:

1. `model.invoke` — request and response, with `method` (chat / generate /
   embeddings), `endpoint`, `latency_ms`, token counts, and the assistant
   output (when `capture_content=True`).
2. `cost.record` — `api_cost_usd: 0.0` always, plus `infra_cost_usd` if
   `cost_per_second` was set.

A failed call additionally emits a third event:

3. `policy.violation` — `provider: ollama`, `error: <message>`,
   `violation_type: safety`. The original exception is re-raised after
   the events fire.

## Ollama-specific behaviour

- **`api_cost_usd: 0.0`** is always emitted because Ollama runs locally —
  there is no API to bill for. The pricing table includes explicit
  zero-cost entries for `llama3.x`, `mistral`, `mixtral`, `phi3`,
  `qwen2.5`, `gemma2`, `deepseek-r1`, `codellama`, `nomic-embed-text`,
  `mxbai-embed-large`, and `all-minilm` so `calculate_cost` returns
  `0.0` (a real number) rather than `None` (pricing-unavailable).
- **Optional `infra_cost_usd`**: pass `cost_per_second` to the
  constructor to attribute compute cost. The adapter sums
  `prompt_eval_duration + eval_duration` (both in nanoseconds) and
  computes `total_seconds * cost_per_second`. Useful for charging back
  GPU rental cost to specific calls.
- **Endpoint capture**: the `OLLAMA_HOST` env var (or
  `http://localhost:11434`) is recorded in every event so you can
  identify which Ollama instance handled a request when running
  multi-host fleets.
- **Three methods wrapped**: `chat`, `generate`, and `embeddings`. The
  `method` field in `model.invoke` distinguishes them. Other SDK
  methods (`pull`, `push`, `list`, `show`, etc.) are NOT instrumented
  because they don't represent inference workload.

## Token extraction

Ollama responses (dict or `ChatResponse`-object form) expose
`prompt_eval_count` and `eval_count` — these map to `prompt_tokens` and
`completion_tokens` in `NormalizedTokenUsage`. `total_tokens` is the
sum.

Embeddings responses don't carry token counts; the adapter falls back
to zeros for the `cost.record` payload.

## GPU notes

Ollama auto-detects available accelerators at daemon start time:

| Platform                       | Backend                       |
|--------------------------------|-------------------------------|
| NVIDIA GPU (CUDA 11.8 / 12.x)  | CUDA — preferred when present |
| AMD GPU (ROCm 5.7+)            | ROCm                          |
| Apple Silicon (M1/M2/M3/M4)    | Metal                         |
| Intel / no GPU                 | CPU                           |

The adapter is GPU-agnostic — it only sees the JSON the daemon returns.
However, the `eval_duration` value the adapter uses for
`infra_cost_usd` is wall-clock time on the daemon, so swapping backends
will change reported infra cost without any adapter change.

### Force CPU-only

```bash
OLLAMA_NUM_GPU=0 ollama serve
```

### Inspect what's loaded

```bash
ollama ps
# NAME              ID        SIZE   PROCESSOR  UNTIL
# llama3.1:latest   abc...    5.0GB  100% GPU   4 minutes from now
```

### NVIDIA driver requirements

| Ollama version | CUDA runtime | Min driver |
|----------------|-------------|-------------|
| 0.5+           | CUDA 12.x   | 525.60.13   |
| 0.4 and older  | CUDA 11.8   | 450.80.02   |

If `ollama ps` reports `100% CPU` despite a CUDA card being present,
check `nvidia-smi` for driver presence and re-run `ollama serve` with
`OLLAMA_DEBUG=1` to see why CUDA was rejected.

## Configuration

| Env var               | Default                       | Effect                                        |
|-----------------------|-------------------------------|-----------------------------------------------|
| `OLLAMA_HOST`         | `http://localhost:11434`      | Daemon endpoint the SDK + adapter point at    |
| `OLLAMA_NUM_GPU`      | auto                          | Layers to offload to GPU (0 = CPU-only)       |
| `OLLAMA_KEEP_ALIVE`   | `5m`                          | How long the daemon keeps a model resident    |
| `OLLAMA_DEBUG`        | unset                         | Verbose daemon logging                        |

## Sample

A runnable end-to-end sample lives at
[`samples/instrument/ollama/`](../../../samples/instrument/ollama/) — runs
mocked-by-default (no daemon required), or live against a real
`ollama serve` with `LAYERLENS_OLLAMA_LIVE=1`.

## Test fixtures

The provider's pytest suite at
`tests/instrument/adapters/providers/test_ollama_adapter.py` uses
`respx` to mock the daemon's HTTP endpoints. This is the recommended
pattern when writing your own integration tests against the adapter —
it gives you coverage of the real httpx → adapter event-emission path
without requiring an Ollama daemon in CI.
