<p align="center">
  <a href="https://layerlens.ai">
    <img src="https://layerlens-public-assets.s3.us-east-1.amazonaws.com/logo-full.png" alt="LayerLens" width="280" />
  </a>
</p>

<h1 align="center">Stratix Python SDK</h1>

<p align="center">
  <strong>Ship AI that actually works. 
    
  Evaluate 200+ models across 100+ benchmarks, trace agent behavior, build custom judges, and gate CI/CD on eval results.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/layerlens/"><img src="https://img.shields.io/pypi/v/layerlens?color=blue" alt="PyPI" /></a>
  <a href="https://pypi.org/project/layerlens/"><img src="https://img.shields.io/pypi/pyversions/layerlens" alt="Python" /></a>
  <a href="https://github.com/LayerLens/stratix-python/stargazers"><img src="https://img.shields.io/github/stars/LayerLens/stratix-python?style=social" alt="GitHub Stars" /></a>
  <a href="https://codecov.io/gh/LayerLens/stratix-python"><img src="https://codecov.io/gh/LayerLens/stratix-python/branch/main/graph/badge.svg" alt="Coverage" /></a>
  <a href="https://github.com/LayerLens/stratix-python/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License" /></a>
  <!-- Replace with actual Discord server ID once created -->
  <a href="https://discord.gg/layerlens"><img src="https://img.shields.io/badge/Discord-Join%20us-5865F2?logo=discord&logoColor=white" alt="Discord" /></a>
</p>

<p align="center">
  <a href="#installation">Install</a> &middot;
  <a href="#quick-start">Quick Start</a> &middot;
  <a href="#how-stratix-compares">Compare</a> &middot;
  <a href="https://layerlens.gitbook.io/stratix-python-sdk">Docs</a> &middot;
  <a href="#examples">Examples</a> &middot;
  <a href="https://discord.gg/layerlens">Discord</a>
</p>

---
<p align="center">
  <img src="./demo-stratix.gif" alt="Stratix Python SDK demo: list 217 frontier models in 5 lines of Python" width="720">
</p>

## Why Stratix?

Stratix is built differently. It gives you production-grade evaluation infrastructure out of the box: rich public benchmarks, powerful custom judges, full agent trace analysis, playback, bulk evaluation, and CI/CD gates.

**What makes it click:**

- **200+ models and 100+ benchmarks, ready to query.** No scraping leaderboards, no CSV wrangling. `pc.models.get()` and you're looking at real evaluation data.
- **Prompt-level comparisons.** Not just "Model A scores 82%." You get the exact prompts where Model A passes and Model B fails, with outcome filters to find the interesting divergences.
- **A 4-generation eval ladder.** Start with heuristic checks, graduate to model-graded scoring, add deliberation panels, then build auto-optimized GEPA judges. One SDK covers the full spectrum.
- **Agent trace evaluation.** Upload a multi-step agent trace, replay it, and judge every step. Built for the world where agents do real work.
- **CI/CD eval gates.** `layerlens ci run --threshold 0.8` in your pipeline. Non-zero exit on regression. No custom scripts needed.

## How Stratix Compares

| Capability              | **Stratix**                                    | LangSmith                  | Langfuse                | DeepEval            | Phoenix (Arize)        |
| ----------------------- | ---------------------------------------------- | -------------------------- | ----------------------- | ------------------- | ---------------------- |
| Pre-built benchmarks    | 100+ benchmarks, 200+ models                   | No public benchmarks       | No public benchmarks    | 50+ metrics         | Bring your own         |
| Prompt-level comparison | Native head-to-head with outcome filters       | Side-by-side runs (manual) | Side-by-side runs + Playground/Experiments (UI Supported)            | Manual setup        | Not built-in           |
| Custom judge builder    | Auto-optimized GEPA judges with budget control | LLM-as-judge (manual)      | LLM-as-judge (manual)   | Basic LLM judges    | LLM-as-judge templates |
| Agent trace evaluation  | Upload, replay, judge every step               | Trace logging + annotation | Trace logging + scoring | Trace logging only  | Trace visualization    |
| Eval generation ladder  | Heuristic > model-graded > deliberation > GEPA | Single generation          | Single generation       | Single generation   | Single generation      |
| CI/CD eval gate         | `layerlens ci run` with threshold              | Custom integration         | Custom integration      | `deepeval test`     | Manual integration     |
| Evaluation Spaces       | Collaborative eval environments                | Hub (paid)                 | Not available           | Not available       | Not available          |
| Dataset versioning      | Pin evals to versions, diff between runs       | Dataset management         | Not built-in            | Basic support       | Dataset management     |
| OpenTelemetry export    | Native OTLP exporter                           | Not built-in               | Native OTLP             | Not built-in        | Native (OpenInference) |
| Pricing model           | Free public data; premium for org features     | Per-trace pricing          | Per-event pricing       | Open source + cloud | Open source + cloud    |

## Pricing

**Free to start.** `PublicClient` is free with an API key–query 200+ models, 50+ benchmarks, and run head-to-head comparisons. Advanced features (traces, custom judges, scorers, CI gates) require **Stratix Premium**. Sign up and purchase credits at [app.layerlens.ai](https://app.layerlens.ai).

## Installation

> [!NOTE]
> `layerlens` is hosted on a private index during early access. Use the command below — the plain `pip install layerlens[cli]` will not work yet.

```bash
pip install --extra-index-url https://sdk.layerlens.ai/package layerlens[cli]
```

## Quick Start

> [!NOTE]
> **Two clients, one SDK.** Use `PublicClient` for models, benchmarks, and comparisons. Use `Stratix` for traces, custom judges, scorers, and CI gates. Both take the same API key.

### 1. Install

```bash
pip install --extra-index-url https://sdk.layerlens.ai/package layerlens[cli]
```

### 2. Set your API key

Get a key from [app.layerlens.ai](https://app.layerlens.ai) → Settings → API Keys.

```bash
export LAYERLENS_STRATIX_API_KEY="your-api-key"
```

### 3. Run your first comparison

```python
from layerlens import PublicClient

pc = PublicClient()

# List available models
models = pc.models.get(page_size=10)
print(f"{models.total_count} models available")

# Compare two models head-to-head on a benchmark
comparison = pc.comparisons.compare_models(
    benchmark_id="aime2024",
    model_id_1="openai/gpt-4o",
    model_id_2="anthropic/claude-opus-4",
    outcome_filter="comparison_fails",  # prompts where model 2 fails
)

print(comparison)
```

That's it! You're comparing frontier models on real benchmark data. **[See full results in the dashboard →](https://stratix.layerlens.ai)**

### Next steps

- **[Run a custom evaluation](./examples/)** ➡️ score your own model on any benchmark
- **[Gate CI/CD on eval results](./examples/ci-gate)** ➡️ `layerlens ci run --threshold 0.8` in your pipeline
- **[Upload and evaluate agent traces](./examples/agent-traces)** ➡️ multi-step trace analysis

## CLI

The SDK ships with a full CLI for managing evaluations from your terminal or CI pipeline:

```bash
# Set your API key
export LAYERLENS_STRATIX_API_KEY="your-api-key"

# List traces
layerlens trace list

# Run a judge evaluation
layerlens judge run --judge-id <id> --trace-id <id>

# Evaluate in CI mode (exits non-zero on failure)
layerlens ci run --judge-id <id> --trace-id <id> --threshold 0.8
```

## Architecture

```
layerlens/
  _client.py          # Stratix (premium) client
  _public_client.py   # PublicClient (open data)
  cli/                # Click-based CLI with rich output
    commands/         # trace, judge, evaluate, scorer, space, bulk, ci
  models/             # Pydantic response models
  resources/          # API resource implementations
  contrib/
    rich_output.py    # Rich terminal tables & progress bars
    otel.py           # OpenTelemetry integration
    tracing.py        # @stratix.trace decorator
    datasets.py       # Dataset versioning & diffs
    error_suggestions.py  # Context-aware error messages
```

## Samples

The [`samples/`](./samples) directory contains 70+ production-ready samples organized by use case. See [`samples/README.md`](./samples/README.md) for the full index.

| Category | Description |
|---|---|
| [Core samples](./samples) | Quickstart, traces, evaluations, judges, async workflows |
| [Industry solutions](./samples/industry) | Healthcare, financial, legal, government, retail, insurance |
| [CI/CD integration](./samples/cicd) | Quality gates, pre-commit hooks, GitHub Actions workflow |
| [Multi-agent (Cowork)](./samples/cowork) | Generator-Evaluator, Code Review, RAG, Incident Response patterns |
| [Content-type evaluations](./samples/modalities) | Text, brand, and document quality scoring |
| [LLM provider integrations](./samples/integrations) | OpenAI, Anthropic, LangChain tracing and instrumentation |
| [MCP server](./samples/mcp) | Expose LayerLens as tools for Claude, Cursor, and any MCP-compatible assistant |
| [CopilotKit CoAgents](./samples/copilotkit) | Full-stack LangGraph + generative UI components |
| [Claude Code skills](./samples/claude-code) | Slash commands for managing LayerLens from the Claude Code CLI |
| [OpenClaw agent evaluation](./samples/openclaw) | Trace, evaluate, and monitor OpenClaw autonomous agents |
| [Sample data](./samples/data) | Pre-built traces, test datasets, and industry evaluation data |

## Used By

<!-- Update this section as adoption grows -->

Stratix powers evaluation workflows at LayerLens and across teams building production AI systems. The public benchmark data is queried thousands of times per week via the SDK and [stratix.layerlens.ai](https://stratix.layerlens.ai).

If your team uses Stratix, [open a PR](https://github.com/LayerLens/stratix-python/pulls) to add your logo here.

## Join the Community

The LayerLens Discord is the best place to:
- Get help with the SDK and trace evaluations
- Share your custom judges and agent workflows
- Access free Stratix Premium Credits for active contributors
- Join weekly Eval Office Hours & model comparison discussions
- Influence the roadmap

[Join the LayerLens Discord!](https://discord.gg/layerlens)

## Documentation

Full documentation is available at [layerlens.gitbook.io/stratix-python-sdk](https://layerlens.gitbook.io/stratix-python-sdk).

To build docs locally:

```bash
pip install layerlens[docs]
mkdocs serve
```

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

## Security

To report a vulnerability, see [SECURITY.md](./SECURITY.md).

## License

Apache 2.0. See [LICENSE](./LICENSE).

## Next Steps

**Get started in under 2 minutes:**

```bash
pip install --extra-index-url https://sdk.layerlens.ai/package "layerlens[cli]"
export LAYERLENS_STRATIX_API_KEY="your-api-key"
python3 -c "from layerlens import PublicClient; pc = PublicClient(); print(pc.models.get(page_size=5))"
```

Then explore the [Quick Start guide](https://layerlens.gitbook.io/stratix-python-sdk), try a [cookbook recipe](https://github.com/LayerLens/stratix-python/tree/main/samples), or [join the Discord](https://discord.gg/layerlens) to ask questions and share what you're building.

---

<p align="center">
  ⭐ <strong>Star us if you found this useful!</strong> ⭐<br />
  It helps more developers discover Stratix.
</p>

<p align="center">
  Built by <a href="https://layerlens.ai">LayerLens</a> &middot; <a href="https://discord.gg/layerlens">Discord</a> &middot; <a href="https://twitter.com/LayerLens_AI">Twitter</a>
</p>
