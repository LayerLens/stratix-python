# Claude Code Skills

These Markdown files define Claude Code slash-command skills that expose LayerLens SDK
operations directly within the Claude Code CLI. Instead of writing Python scripts, developers
can invoke LayerLens capabilities through natural language commands like `/trace`, `/evaluate`,
or `/investigate` -- enabling AI-assisted quality assurance without leaving the terminal.

## Prerequisites

- Claude Code CLI installed and configured
- The LayerLens Python SDK installed: `pip install layerlens --index-url https://sdk.layerlens.ai/package`
- `LAYERLENS_STRATIX_API_KEY` set as an environment variable

To register the skills, add the `samples/claude-code/skills/` directory to your Claude Code
skill search path, or copy individual `.md` files into your project's `.claude/skills/`
directory.

## Quick Start

After registering the skills, invoke the trace skill from Claude Code:

```
/trace
```

Claude Code will guide you through creating and uploading a trace interactively.

## Skills

| File | Command | Scenario | Description |
|------|---------|----------|-------------|
| `skills/trace.md` | `/trace` | Developers instrumenting LLM calls | Create, upload, and manage trace records. Guides the user through trace creation with prompts for model, input, and output data. |
| `skills/evaluate.md` | `/evaluate` | QA teams running evaluations | Run evaluations against traces using specified judges. Supports selecting judges, setting thresholds, and reviewing results interactively. |
| `skills/judge.md` | `/judge` | Platform teams managing evaluation criteria | Create, list, update, and delete judges. Provides an interactive workflow for defining judge criteria and testing them against sample traces. |
| `skills/investigate.md` | `/investigate` | On-call engineers debugging production issues | Analyze traces for errors, latency anomalies, and cost outliers. Produces a structured investigation report with suggested remediation steps. |
| `skills/benchmark.md` | `/benchmark` | ML teams comparing model performance | Run benchmarks across models and review comparative results. Supports custom task batteries and historical trend analysis. |
| `skills/optimize.md` | `/optimize` | Teams refining judge accuracy | Optimize judge configurations by testing against labeled datasets and adjusting scoring parameters to improve precision and recall. |

## Expected Behavior

Each skill operates interactively within the Claude Code session. The skill prompts the user
for required inputs, executes the corresponding LayerLens SDK operations, and presents
formatted results directly in the terminal. All operations use the authenticated Stratix
client and persist data to your LayerLens workspace.
