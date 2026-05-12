# Semantic Kernel framework adapter

`layerlens.instrument.adapters.frameworks.semantic_kernel.SemanticKernelAdapter`
instruments [Microsoft Semantic Kernel](https://github.com/microsoft/semantic-kernel)
using the kernel's native filter API — non-invasive, no monkey-patching.

## Install

```bash
pip install 'layerlens[semantic-kernel]'
```

Pulls `semantic-kernel>=1.0,<2.0`. Requires Python 3.10+.

## Quick start

```python
import asyncio
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

from layerlens.instrument.adapters.frameworks.semantic_kernel import SemanticKernelAdapter
from layerlens.instrument.transport.sink_http import HttpEventSink

sink = HttpEventSink(adapter_name="semantic_kernel")
adapter = SemanticKernelAdapter()
adapter.add_sink(sink)
adapter.connect()

kernel = Kernel()
kernel.add_service(OpenAIChatCompletion(ai_model_id="gpt-4o-mini"))
adapter.instrument_kernel(kernel)

async def main() -> None:
    result = await kernel.invoke_prompt("What is 2 + 2?")
    print(result)

asyncio.run(main())

adapter.disconnect()
sink.close()
```

## What's wrapped

`adapter.instrument_kernel(kernel)` registers three Semantic Kernel filters
on the supplied kernel:

- `function_invocation_filter` — fires before/after every `KernelFunction`
  call (plugin function, prompt function, etc.).
- `prompt_rendering_filter` — fires before/after the prompt template is
  rendered for prompt functions.
- `auto_function_invocation_filter` — fires when the model auto-selects a
  plugin function via tool-calling.

No methods are monkey-patched; on `disconnect()` the filter list is cleared
and the kernel returns to its original behaviour.

## Events emitted

| Event | Layer | When |
|---|---|---|
| `environment.config` | L4a | First plugin invocation per kernel. |
| `agent.input` | L1 | Function invocation start. |
| `agent.output` | L1 | Function invocation end (success or error). |
| `agent.code` | L2 | Per plugin function when `l2_agent_code` is true. |
| `agent.action` | L4a | Per planner step. |
| `agent.state.change` | cross-cutting | Memory store reads/writes. |
| `tool.call` | L5a | Per `auto_function_invocation` (model-selected plugin). |
| `model.invoke` | L3 | Per LLM call inside the kernel. |

## Semantic Kernel specifics

- **Plugin attribution**: every event includes `plugin_name`,
  `function_name`, and (for prompt functions) the rendered prompt token
  count when available.
- **Filter API is preferred**: filters are first-class Semantic Kernel
  citizens — they survive kernel cloning and don't break the type system.
  This is why this adapter uses filters instead of method-wrapping.
- **Async-first**: Semantic Kernel is async-first; all filters are async
  and propagate the `next` continuation correctly.

## Capture config

```python
from layerlens.instrument.adapters._base import CaptureConfig

# Recommended.
adapter = SemanticKernelAdapter(capture_config=CaptureConfig.standard())

# Capture rendered prompt template body.
adapter = SemanticKernelAdapter(
    capture_config=CaptureConfig(
        l1_agent_io=True,
        l3_model_metadata=True,
        l5a_tool_calls=True,
        capture_content=True,
    ),
)
```

## BYOK

Semantic Kernel uses `OpenAIChatCompletion`, `AzureChatCompletion`,
`HuggingFacePromptExecutionSettings`, etc. for model access. The adapter
does not own those credentials. For platform-managed BYOK see
`docs/adapters/byok.md` (atlas-app M1.B).
