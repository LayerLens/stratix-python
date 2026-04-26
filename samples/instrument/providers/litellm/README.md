# LiteLLM adapter sample

This sample demonstrates the LayerLens LiteLLM adapter intercepting
calls routed by [LiteLLM](https://docs.litellm.ai/) — a multi-provider
gateway that dispatches a single `litellm.completion(...)` call to one
of ~100 underlying providers.

The sample is **mocked by default** — it does not require any provider
API key and never reaches the network. A live mode is opt-in via an
environment variable.

## Install

```bash
pip install 'layerlens[providers-litellm]'
```

The `providers-litellm` extra installs `litellm>=1.40,<2`. The default
`pip install layerlens` does **not** pull `litellm` — that's the
lazy-import guarantee tested by `tests/instrument/test_lazy_imports.py`.

## Run (offline / mocked)

```bash
python -m samples.instrument.providers.litellm.main
```

You'll see the adapter emit `model.invoke` and `cost.record` events for
six routing scenarios:

| Model string | Resolves to |
|---|---|
| `openai/gpt-4o-mini` | OpenAI (explicit prefix) |
| `anthropic/claude-3-5-sonnet` | Anthropic (explicit prefix) |
| `bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0` | AWS Bedrock |
| `vertex_ai/gemini-1.5-pro` | Google Vertex AI |
| `gpt-4` | OpenAI (bare-name heuristic) |
| `claude-3-5-sonnet` | Anthropic (bare-name heuristic) |

The "stratix" stub in the sample just prints each event to stdout —
production usage attaches an `HttpEventSink` to ship the events to
atlas-app (see `samples/instrument/openai/main.py` for that pattern).

## Run (live LiteLLM round-trip)

```bash
export LAYERLENS_LITELLM_LIVE=1
export OPENAI_API_KEY=sk-...
python -m samples.instrument.providers.litellm.main
```

This calls the real `litellm.completion(...)` and dispatches the
adapter callback for an actual `gpt-4o-mini` chat completion.

## Provider routing reference

The full prefix → provider mapping is documented in
[`docs/adapters/providers-litellm.md`](../../../../docs/adapters/providers-litellm.md).
