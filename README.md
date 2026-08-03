<p align="center">
  <a href="https://layerlens.ai">
    <img src="https://layerlens-public-assets.s3.us-east-1.amazonaws.com/logo-full.png" alt="LayerLens" width="280" />
  </a>
</p>

<h1 align="center">Stratix Python SDK</h1>

<p align="center">
  <strong>Ship AI that actually works.<br />
  Reference results on 172 models and 78 benchmarks, and evaluation that survives a framework migration.</strong>
</p>

<p align="center">
  <a href="./CHANGELOG.md"><img src="https://img.shields.io/badge/version-1.9.0-blue" alt="Version" /></a>
  <a href="#requirements"><img src="https://img.shields.io/badge/python-3.8%2B-blue" alt="Python 3.8+" /></a>
  <a href="https://github.com/LayerLens/stratix-python/stargazers"><img src="https://img.shields.io/github/stars/LayerLens/stratix-python?style=social" alt="GitHub Stars" /></a>
  <a href="https://codecov.io/gh/LayerLens/stratix-python"><img src="https://codecov.io/gh/LayerLens/stratix-python/branch/main/graph/badge.svg" alt="Coverage" /></a>
  <a href="https://github.com/LayerLens/stratix-python/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License" /></a>
  <a href="https://discord.gg/layerlens"><img src="https://img.shields.io/badge/Discord-Join%20us-5865F2?logo=discord&logoColor=white" alt="Discord" /></a>
</p>

<p align="center">
  <a href="#installation">Install</a> &middot;
  <a href="#quick-start">Quick Start</a> &middot;
  <a href="#adapters">Adapters</a> &middot;
  <a href="#how-stratix-compares">Compare</a> &middot;
  <a href="https://layerlens.gitbook.io/stratix-python-sdk">Docs</a> &middot;
  <a href="#samples">Samples</a> &middot;
  <a href="https://discord.gg/layerlens">Discord</a>
</p>

---
<p align="center">
  <img src="./demo-stratix.gif" alt="Stratix Python SDK demo: list frontier models in 5 lines of Python" width="720">
</p>

## Why Stratix?

Production-grade evaluation infrastructure out of the box: public benchmark data, custom judges, full agent trace analysis, playback, bulk evaluation, and CI/CD gates.

**What makes it click:**

- **172 models and 78 benchmarks, ready to query.** No scraping leaderboards, no CSV wrangling. `pc.models.get()` and you're looking at real evaluation data.
- **Prompt-level comparisons.** Not just "Model A scores 82%." You get the exact prompts where Model A passes and Model B fails, with outcome filters to find the interesting divergences.
- **A 4-generation eval ladder.** Start with heuristic checks, graduate to model-graded scoring, add deliberation panels, then build auto-optimized GEPA judges. One SDK covers the full spectrum.
- **Agent trace evaluation.** Upload a multi-step agent trace, replay it, and judge every step. Built for the world where agents do real work.
- **[Adapters](#adapters) for the stack you already run.** Traces from your agent frameworks, protocols and model providers land in one schema, so a judge you tuned last quarter runs unchanged on next quarter's stack.
- **CI/CD eval gates.** `layerlens ci report` emits a markdown summary for your job output, and [`samples/cicd/`](./samples/cicd/) has a ready-to-copy threshold gate that exits non-zero on regression.

## How Stratix Compares

| Capability              | **Stratix**                                    | LangSmith                  | Langfuse                | DeepEval            | Phoenix (Arize)        |
| ----------------------- | ---------------------------------------------- | -------------------------- | ----------------------- | ------------------- | ---------------------- |
| Pre-built benchmarks    | 78 benchmarks, 172 models                      | No public benchmarks       | No public benchmarks    | 50+ metrics         | Bring your own         |
| Prompt-level comparison | Native head-to-head with outcome filters       | Side-by-side runs (manual) | Side-by-side runs + Playground/Experiments (UI Supported)            | Manual setup        | Not built-in           |
| Custom judge builder    | Auto-optimized GEPA judges with budget control | LLM-as-judge (manual)      | LLM-as-judge (manual)   | Basic LLM judges    | LLM-as-judge templates |
| Agent trace evaluation  | Upload, replay, judge every step               | Trace logging + annotation | Trace logging + scoring | Trace logging only  | Trace visualization    |
| Eval generation ladder  | Heuristic > model-graded > deliberation > GEPA | Single generation          | Single generation       | Single generation   | Single generation      |
| CI/CD eval gate         | `layerlens ci report` + sample threshold gate  | Custom integration         | Custom integration      | `deepeval test`     | Manual integration     |
| Evaluation Spaces       | Collaborative eval environments                | Hub (paid)                 | Not available           | Not available       | Not available          |
| Dataset versioning      | Pin evals to versions, diff between runs       | Dataset management         | Not built-in            | Basic support       | Dataset management     |
| OpenTelemetry export    | Native OTLP exporter                           | Not built-in               | Native OTLP             | Not built-in        | Native (OpenInference) |
| Pricing model           | Free public data; premium for org features     | Per-trace pricing          | Per-event pricing       | Open source + cloud | Open source + cloud    |

## Pricing

**Free to start.** `PublicClient` is free with an API key–query 172 models, 78 benchmarks, and run head-to-head comparisons. Advanced features (traces, custom judges, scorers, CI gates) require **Stratix Premium**. Sign up and purchase credits at [stratix.layerlens.ai](https://stratix.layerlens.ai).

## Requirements

- **Python 3.8+.** CI runs the test suite on 3.9–3.12. A few framework extras (`crewai`, `autogen`, `semantic-kernel`, `mcp`, `a2a`, `dspy`, `marvin`, `mirascope`, `pydantic-ai`) require Python 3.10+ and are skipped automatically on older interpreters. On the other end, `layerlens[crewai]` has been reported to fail to install on Python 3.14 (a `tiktoken` build failure upstream, not in this SDK) — use 3.11 or 3.12 for CrewAI work until that resolves.
- **OS independent.** Linux, macOS and Windows.
- **Async supported.** Every client has an async twin — `AsyncStratix`, `AsyncPublicClient` — with the same methods and `await` semantics.
- **Runtime dependencies:** `httpx` and `pydantic` — the core SDK works on Pydantic v1.9+ or v2; some framework adapters (LangGraph, CrewAI, AutoGen) are Pydantic v2-only, following their upstream. The CLI adds `click`.

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

Get a key from [stratix.layerlens.ai](https://stratix.layerlens.ai) → Settings → API Keys.

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
       benchmark_key="aime2024",
       model_key_1="openai/gpt-4o",
       model_key_2="anthropic/claude-3.5-haiku",
       outcome_filter="comparison_fails",  # prompts where model 2 fails
   )

print(comparison)
```

That's it! You're comparing frontier models on real benchmark data. **[See full results in the dashboard →](https://stratix.layerlens.ai)**

### Next steps

- **[Run a custom evaluation](./samples/core/)** ➡️ score your own model on any benchmark
- **[Gate CI/CD on eval results](./samples/cicd/)** ➡️ `python quality_gate.py --threshold 0.85` in your pipeline
- **[Upload and evaluate agent traces](./samples/instrument/)** ➡️ multi-step trace analysis

## Adapters

Adapters normalize traces from the agent frameworks, protocols and model providers you already use into a single canonical schema, so evaluation built against one stack keeps working when the stack changes.

**The problem they solve.** Judges and baselines are tuned against the traces a particular framework emits. Migrate frameworks, or let a second team pick a different one, and the traces come out a different shape: the pipeline stops parsing them, and every baseline you built stops being comparable to anything new. The evaluation work is locked to the framework that made it necessary.

**What an adapter does.** It maps a source-specific run into the Stratix event schema: one representation of steps, tool calls, inputs, outputs, errors and timings, regardless of origin. A judge tuned last quarter runs unchanged on the agent you ship next quarter, and switching frameworks is a wiring change rather than a rewrite.

**Integrity.** Every captured trace carries a SHA-256 hash chain over its events, sealed with a root hash. Altering or removing an event breaks verification rather than passing quietly, so a trace you evaluated is a trace you can attest to later.

### Usage

Install the extra for your stack — one per framework or provider, e.g. `layerlens[langchain]`, `layerlens[crewai]`, `layerlens[mcp]`:

```bash
pip install --extra-index-url https://sdk.layerlens.ai/package "layerlens[langchain]"
```

Then let `auto()` wire an adapter for everything it finds installed:

```python
from layerlens import Stratix
from layerlens.instrument import auto, discover_installed

client = Stratix()

print(discover_installed())   # what auto() would wire, without connecting anything
auto(client)                  # instrument every installed framework and provider

# Run your agent as usual. Captured events upload as a normalized trace.
```

Wire a single framework explicitly when you want control over what gets captured:

```python
from layerlens.instrument import CaptureConfig
from layerlens.instrument.adapters.frameworks.langchain import LangChainCallbackHandler

handler = LangChainCallbackHandler(
    client,
    capture_config=CaptureConfig(capture_content=True),
)
result = chain.invoke({"question": "..."}, config={"callbacks": [handler]})
```

> **`capture_content` is off by default, and content-quality judges need it on.**
> The default captures structure and metadata — that a model was invoked, with what
> latency and token count — but not the prompt or response text. That is deliberate:
> content capture is opt-in so a trace cannot leak prompts you did not choose to send.
> The consequence is that a judge grading answer quality has nothing to read and will
> correctly score 0.0 with a "not determinable" verdict. Turn it on when you intend to
> grade content, and treat it as a per-environment decision rather than a global default.

Switching the source does not touch your evaluation — same judge, same baselines, same schema. Swap the adapter, or just let `auto()` pick up the framework you moved to:

```python
from layerlens.instrument.adapters.frameworks.openai_agents import OpenAIAgentsAdapter

adapter = OpenAIAgentsAdapter(client)
adapter.connect()
```

From here every trace is the same shape, whatever it came from:

```python
traces = client.traces.get_many(page_size=1, sort_by="created_at", sort_order="desc")

evaluation = client.trace_evaluations.create(
    trace_id=traces.traces[0].id,
    judge_id="<your-judge-id>",
)
```

Trace evaluations run asynchronously — poll `client.trace_evaluations.get(evaluation.id)` until it reports a terminal status.

Verify the hash chain offline at any point — `payload` is the trace dict the SDK uploaded, or a saved replay snapshot:

```python
from layerlens.attestation import AttestationEnvelope, HashScope, verify_chain

envelopes = [
    AttestationEnvelope(
        hash=e["hash"],
        scope=HashScope(e["scope"]),
        previous_hash=e.get("previous_hash"),
    )
    for e in payload["attestation"]["chain"]["events"]
]

assert verify_chain(envelopes).valid
```

`detect_tampering(envelopes, payload["events"])` goes further and reports which events were modified.

### Coverage

Adapters cover agent frameworks, agent protocols and model providers. `discover_installed()` reports what is wired in your environment; per-adapter guides live in [docs/adapters/](./docs/adapters) and the event schema every adapter emits is documented in [docs/event-schema.md](./docs/event-schema.md).

### Status

Adapters are in public preview. The event schema is stable for the documented fields; additional fields may be added in minor releases. Breaking changes are called out in [CHANGELOG.md](./CHANGELOG.md).

## CLI

The SDK ships with a full CLI for managing evaluations from your terminal or CI pipeline:

```bash
# Set your API key
export LAYERLENS_STRATIX_API_KEY="your-api-key"

# List traces
layerlens trace list

# Run a judge evaluation
layerlens judge test --judge-id <id> --trace-id <id>

# Emit a markdown eval summary for a CI job summary
layerlens ci report -o "$GITHUB_STEP_SUMMARY"
```

`ci report` summarizes; it has no pass/fail semantics. For a gate that exits
non-zero below a threshold, copy [`samples/cicd/quality_gate.py`](./samples/cicd/quality_gate.py).

## Architecture

```
layerlens/
  _client.py          # Stratix / Client (premium) + AsyncStratix
  _public_client.py   # PublicClient (open data) + AsyncPublicClient
  cli/                # Click-based CLI
    commands/         # trace, judge, evaluate, evaluations, scorer, space,
                      # bulk, ci, replay, synthetic, integration, auth
  resources/          # API resource implementations
  models/             # Pydantic response models
  instrument/         # @trace decorator, spans, context propagation, collector
    adapters/
      frameworks/     # LangChain, LangGraph, CrewAI, AutoGen, LlamaIndex, ...
      protocols/      # MCP, A2A, AG-UI, A2UI, AP2, UCP
      providers/      # OpenAI, Anthropic, Azure, Vertex, Bedrock, Ollama, ...
  attestation/        # SHA-256 hash chain, envelopes, signing, verification
  replay/             # Trace snapshots, replay controller, diff engine
  evaluation_runs/    # Run orchestration, scheduling, comparison
  datasets/           # Dataset versioning & diffs
  synthetic/          # Synthetic trace & dataset generation
  benchmarks/         # Benchmark import helpers
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

## Versioning and Compatibility

The SDK follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html): breaking changes only in a major release, new capability in minors, fixes in patches. Anything not documented here or in the docs site — modules and attributes prefixed with `_`, and internal event fields — is not part of the public API and can change in any release. Every release is recorded in [CHANGELOG.md](./CHANGELOG.md), with deprecations announced there at least one minor version before removal.

## Data Handling

Traces you capture or upload are sent to LayerLens and stored in your organization's project scope; they are not shared across organizations, and the SDK writes nothing to disk. API keys and authorization headers are redacted from SDK logs, and secrets found in error text are scrubbed before upload. Capture is configurable — use `CaptureConfig` to drop payload content or whole event layers before anything leaves your process. See [docs/security/data-privacy.md](./docs/security/data-privacy.md) for what is transmitted and how to sanitize it.

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
