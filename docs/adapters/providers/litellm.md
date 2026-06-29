# LiteLLM provider adapter

Instruments [LiteLLM](https://github.com/BerriAI/litellm) by monkey-patching
the `litellm` module's `completion` and `acompletion` functions. LiteLLM
normalizes many providers onto the OpenAI request/response shape, so the
adapter reuses `OpenAIProvider`'s output and metadata extractors. Because the
patch is on the module functions (not a client instance), `connect()` takes no
argument.

## Install

```bash
pip install layerlens[litellm]
```

Pulls the `litellm` package.

## Usage

```python
import litellm
from layerlens.instrument.adapters.providers import LiteLLMProvider

provider = LiteLLMProvider()
provider.connect()   # patches litellm.completion / litellm.acompletion

resp = litellm.completion(
    model="gpt-4o-mini",   # or "claude-3-5-sonnet", "bedrock/anthropic.claude-3-..."
    messages=[{"role": "user", "content": "Hi"}],
)
print(resp.choices[0].message.content)
```

`connect()` takes no target — it patches the imported `litellm` module
directly. The underlying provider key (e.g. `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`) must be configured for LiteLLM to make a live call; the
adapter does not manage provider credentials.

## Routing

LiteLLM dispatches each call to the underlying provider based on the `model`
string — `gpt-4o-mini` routes to OpenAI, `claude-3-5-sonnet` to Anthropic,
`bedrock/anthropic.claude-3-...` to Bedrock, and so on. The adapter observes
the call at the `litellm.completion` boundary, so the `model` captured on the
event is whatever model string was passed, and pricing is resolved from the
bundled pricing manifest by that routed model slug.

Note: the adapter currently attributes every event with `provider="litellm"`.
It does **not** yet classify the routed target provider (OpenAI / Anthropic /
Bedrock / etc.) — events from a LiteLLM-routed Anthropic call are tagged
`litellm`, not `anthropic`. Cost still resolves correctly from the routed model
slug when that slug exists in the manifest.

## Event surface

- `model.invoke` for every `litellm.completion` (sync) and
  `litellm.acompletion` (async) call, named `litellm.completion` /
  `litellm.acompletion`. Payload includes `model`, `usage`, and `finish_reason`
  via the shared OpenAI extractors.
- `tool.call` per function/tool call surfaced in the response.
- `cost.record` for each invoke whose response carries usage data, priced from
  the bundled pricing manifest by the routed model slug.
- Streaming (`stream=True`) is handled by the shared streaming path — chunks
  are aggregated and emitted as a single `model.invoke` when the stream ends.
- `agent.error` if the underlying call raises.

## Pricing

There is no LiteLLM-specific pricing table; the adapter uses the default
bundled manifest and resolves cost by the routed model slug (the same id you
passed to `litellm.completion`). Model slugs absent from the manifest resolve
to no cost.

## Sample

[`samples/instrument/litellm/example.py`](../../../samples/instrument/litellm/example.py)
and [`samples/adapters/providers/litellm_chat.py`](../../../samples/adapters/providers/litellm_chat.py)

## Compat

- `litellm`
- Python 3.9+
