# Semantic Kernel adapter

Instruments [Semantic Kernel](https://github.com/microsoft/semantic-kernel)
kernels via the SK filter API (semantic-kernel ≥ 1.0).

## Install

```bash
pip install layerlens[semantic-kernel]
```

Pulls `semantic-kernel>=1.0.0`. Semantic Kernel requires Python 3.10+.

## Usage

```python
import asyncio
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from layerlens.instrument.adapters.frameworks import SemanticKernelAdapter

kernel = Kernel()
kernel.add_service(OpenAIChatCompletion(service_id="gpt4", ai_model_id="gpt-4o"))

adapter = SemanticKernelAdapter(client=layerlens_client)
adapter.connect(target=kernel)            # registers filters on this Kernel

async def run():
    return await kernel.invoke_prompt("Hello!")

asyncio.run(run())
adapter.disconnect()                      # removes filters and flushes the trace
```

`connect(target=kernel)` is required — the adapter installs filters on a
specific `Kernel` instance rather than monkey-patching a module.

## Event surface

The adapter registers three SK filters and emits flat events:

- `tool.call` / `tool.result` from the function invocation filter — one pair
  per plugin function call with arguments and result. The auto function
  invocation filter emits the same pair for LLM-initiated calls discovered
  during a chat completion.
- `agent.code` from the prompt rendering filter — the rendered prompt
  template (with substituted variables) under `event_subtype="prompt_render"`.
- `model.invoke` from wrapped chat services on the kernel, including model
  name and token usage when reported by the service.
- `cost.record` alongside `model.invoke` when the service reports token usage.
- `environment.config` when a plugin is discovered/registered on the kernel.
- `agent.error` when a function invocation raises.

Run boundaries are detected by a nesting depth counter: `_begin_run` fires
on the outermost function invocation and `_end_run` on its completion.
Concurrent invocations on different asyncio tasks are isolated via a
ContextVar-based `RunState`.

## Sample

[`samples/instrument/semantic_kernel/example.py`](../../../samples/instrument/semantic_kernel/example.py)

## Compat

- Semantic Kernel 1.0+
- Python 3.10+
