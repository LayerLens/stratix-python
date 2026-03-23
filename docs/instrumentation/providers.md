# LLM Provider Instrumentation

Provider adapters automatically capture LLM spans when SDK methods are called inside a `@trace` context. No changes to your LLM calling code are needed.

## Supported Providers

| Provider | Adapter | Wraps |
| -------- | ------- | ----- |
| OpenAI | `instrument_openai(client)` | `client.chat.completions.create` |
| Anthropic | `instrument_anthropic(client)` | `client.messages.create` |
| LiteLLM | `instrument_litellm()` | `litellm.completion`, `litellm.acompletion` |

LiteLLM provides a unified interface to 100+ providers (Azure, Google, Cohere, Mistral, Bedrock, etc.), so `instrument_litellm()` covers all of them.

## OpenAI

### Installation

```bash
pip install layerlens[openai]
```

### Usage

```python
import openai
from layerlens import Stratix
from layerlens.instrument import trace
from layerlens.instrument.adapters.providers.openai import instrument_openai

client = Stratix()
openai_client = openai.OpenAI()

# Instrument the client instance
instrument_openai(openai_client)

@trace(client)
def my_agent(question: str):
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": question}],
    )
    return response.choices[0].message.content

my_agent("What is Python?")
```

The adapter captures:

- **Span name**: `openai.chat.completions.create`
- **Kind**: `llm`
- **Input**: Messages array
- **Output**: Assistant message content
- **Metadata**: `model`, `temperature`, `max_tokens`, `usage` (prompt/completion/total tokens)

### Disconnect

```python
from layerlens.instrument.adapters.providers.openai import OpenAIProvider

provider = OpenAIProvider()
provider.connect_client(openai_client)

# Later, restore original methods:
provider.disconnect()
```

## Anthropic

### Installation

```bash
pip install layerlens[anthropic]
```

### Usage

```python
import anthropic
from layerlens import Stratix
from layerlens.instrument import trace
from layerlens.instrument.adapters.providers.anthropic import instrument_anthropic

client = Stratix()
anthropic_client = anthropic.Anthropic()

instrument_anthropic(anthropic_client)

@trace(client)
def my_agent(question: str):
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": question}],
    )
    return response.content[0].text

my_agent("What is Python?")
```

The adapter captures:

- **Span name**: `anthropic.messages.create`
- **Kind**: `llm`
- **Input**: Messages array
- **Output**: Response content blocks
- **Metadata**: `model`, `usage` (input/output tokens), `stop_reason`

### Disconnect

```python
from layerlens.instrument.adapters.providers.anthropic import AnthropicProvider

provider = AnthropicProvider()
provider.connect_client(anthropic_client)
provider.disconnect()
```

## LiteLLM

LiteLLM works differently from OpenAI/Anthropic — it patches module-level functions rather than client instances.

### Installation

```bash
pip install layerlens[litellm]
```

### Usage

```python
import litellm
from layerlens import Stratix
from layerlens.instrument import trace
from layerlens.instrument.adapters.providers.litellm import instrument_litellm

client = Stratix()

# Patch litellm module (call once at startup)
instrument_litellm()

@trace(client)
def my_agent(question: str):
    response = litellm.completion(
        model="gpt-4o",
        messages=[{"role": "user", "content": question}],
    )
    return response.choices[0].message.content

my_agent("What is Python?")
```

Since LiteLLM supports 100+ providers, this single call instruments all of them:

```python
instrument_litellm()

@trace(client)
def multi_provider():
    # All of these generate LLM spans
    litellm.completion(model="gpt-4o", messages=[...])
    litellm.completion(model="claude-sonnet-4-20250514", messages=[...])
    litellm.completion(model="gemini/gemini-pro", messages=[...])
```

### Uninstrument

```python
from layerlens.instrument.adapters.providers.litellm import uninstrument_litellm

uninstrument_litellm()
```

## Captured Metadata

All provider adapters capture these request parameters when present:

| Parameter | Description |
| --------- | ----------- |
| `model` | Model name/ID |
| `temperature` | Sampling temperature |
| `max_tokens` | Maximum response tokens |
| `top_p` | Nucleus sampling parameter |
| `frequency_penalty` | Frequency penalty |
| `presence_penalty` | Presence penalty |
| `response_format` | Structured output format |

Response metadata varies by provider but always includes token usage when available.

## Passthrough Behavior

When called **outside** a `@trace` context, all adapters pass through to the original SDK method with zero overhead. This means you can instrument at startup and leave it on — it only activates when a trace is running.

```python
instrument_openai(openai_client)

# No active trace — passes through directly to OpenAI
openai_client.chat.completions.create(model="gpt-4o", messages=[...])

@trace(client)
def traced_call():
    # Active trace — generates an LLM span
    openai_client.chat.completions.create(model="gpt-4o", messages=[...])
```

## Error Handling

If an LLM call raises an exception inside a `@trace`, the adapter records the error on the span and re-raises the exception:

```python
@trace(client)
def my_agent():
    try:
        openai_client.chat.completions.create(model="gpt-4o", messages=[...])
    except openai.APIError:
        pass  # Span is recorded with status="error"
```

## Next Steps

- [Agent Frameworks](frameworks.md) — LangChain and LangGraph callback handlers
- [Quick Start](quickstart.md) — Manual instrumentation with `@trace` and `span()`
