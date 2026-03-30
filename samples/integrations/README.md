# Integration Samples

Most AI applications interact with LLMs through provider-specific SDKs such as OpenAI or
Anthropic. These samples demonstrate how to instrument those API calls with LayerLens tracing
and run post-completion evaluations -- without modifying your existing provider integration
code. This enables teams to add observability and quality evaluation to production LLM
calls with minimal effort.

## Prerequisites

```bash
pip install layerlens --index-url https://sdk.layerlens.ai/package
export LAYERLENS_STRATIX_API_KEY=your-api-key
```

Each sample requires the corresponding provider SDK and API key:

| Sample | Additional Dependencies |
|--------|------------------------|
| `openai_traced.py` | `pip install openai` and `OPENAI_API_KEY` |
| `anthropic_traced.py` | `pip install anthropic` and `ANTHROPIC_API_KEY` |

## Quick Start

Start with `openai_traced.py` if you use OpenAI, or `anthropic_traced.py` for Anthropic:

```bash
export OPENAI_API_KEY=your-openai-key
python openai_traced.py
```

Expected output: the script makes an LLM API call, automatically captures the interaction
as a LayerLens trace (including prompt, completion, token usage, and latency), runs a
post-completion evaluation, and prints the trace ID and evaluation scores.

## Samples

| File | Scenario | Description |
|------|----------|-------------|
| `openai_traced.py` | Teams using OpenAI GPT models in production | Wraps an OpenAI chat completion call with LayerLens tracing, captures the full request/response cycle, and runs a post-completion evaluation with safety and relevance judges. |
| `anthropic_traced.py` | Teams using Anthropic Claude models in production | Wraps an Anthropic message API call with LayerLens tracing, captures the full request/response cycle, and runs a post-completion evaluation with safety and relevance judges. |

## Expected Behavior

Each sample makes a real API call to the respective provider, so valid provider credentials
are required. The trace is uploaded to your LayerLens workspace and the evaluation results
are printed to stdout. Both samples follow the same pattern, making it straightforward to
adapt the approach to additional providers.
