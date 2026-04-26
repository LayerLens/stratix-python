# Semantic Kernel sample

Runnable end-to-end sample for the Microsoft Semantic Kernel framework
adapter. The script wires the LayerLens filters via
`SemanticKernelAdapter.instrument_kernel(kernel)` and runs a single
`invoke_prompt` call against an `OpenAIChatCompletion` service.

## Install

```bash
pip install 'layerlens[semantic-kernel,providers-openai]'
```

This pulls `semantic-kernel>=1.0,<2.0` (which itself depends on
`openai>=1.30`). Requires Python 3.10+.

## Run

```bash
export OPENAI_API_KEY=sk-...
export LAYERLENS_STRATIX_API_KEY=...    # optional — needed only to ship spans
export LAYERLENS_STRATIX_BASE_URL=...    # optional — defaults to LayerLens cloud

python -m samples.instrument.semantic_kernel.main
```

The sample prints the model's response and ships an
`agent.input` / `agent.output` / `model.invoke` event triple to
atlas-app via `HttpEventSink`. If the LayerLens credentials are not set,
the sink buffers the events in memory and drops them on shutdown — the
SK call still runs.

## What this exercises

- `SemanticKernelAdapter` lifecycle (`connect` → `instrument_kernel` →
  `disconnect`).
- All three SK filters: `function_invocation`, `prompt_rendering`,
  `auto_function_invocation`.
- The HTTP transport sink batched flush path.

For the full adapter reference see
[`docs/adapters/frameworks-semantic_kernel.md`](../../../docs/adapters/frameworks-semantic_kernel.md).
