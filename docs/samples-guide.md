# Samples Guide

The LayerLens Python SDK ships with 70+ runnable samples covering every API resource, from a single trace evaluation to enterprise compliance pipelines and multi-agent orchestration. All samples live in the [`samples/`](../samples/) directory and can be run directly after installing the SDK and setting your API key.

## Quick Start

```bash
pip install layerlens --index-url https://sdk.layerlens.ai/package
export LAYERLENS_STRATIX_API_KEY=your-api-key
python samples/core/quickstart.py
```

[`quickstart.py`](../samples/core/quickstart.py) walks through the complete workflow end-to-end: upload a trace, create a judge, run an evaluation, and retrieve results.

## Samples by Category

### Core SDK Operations (18 samples)

Located in [`samples/core/`](../samples/core/). Start here to learn how every LayerLens resource -- traces, judges, evaluations, results, models, and benchmarks -- works individually and together, including async patterns and pagination.

Key samples:
- [`quickstart.py`](../samples/core/quickstart.py) -- Your first evaluation in under 30 lines
- [`trace_evaluation.py`](../samples/core/trace_evaluation.py) -- Full trace evaluation lifecycle
- [`judge_optimization.py`](../samples/core/judge_optimization.py) -- Optimize judge accuracy via automated prompt engineering
- [`evaluation_pipeline.py`](../samples/core/evaluation_pipeline.py) -- Chain judges, traces, and results into an automated pipeline
- [`async_workflow.py`](../samples/core/async_workflow.py) -- Concurrent operations with AsyncStratix

See the [Core SDK README](../samples/core/README.md) for the full list.

### Industry Solutions (10 samples)

Located in [`samples/industry/`](../samples/industry/). Domain-specific evaluation scenarios with judges tuned for regulated and high-stakes verticals including healthcare, financial services, legal, government, insurance, and retail.

Key samples:
- [`healthcare_clinical.py`](../samples/industry/healthcare_clinical.py) -- Clinical decision support evaluation
- [`financial_trading.py`](../samples/industry/financial_trading.py) -- SOX-aligned trading compliance
- [`legal_contracts.py`](../samples/industry/legal_contracts.py) -- Contract review quality assessment

See the [Industry Solutions README](../samples/industry/README.md) for the full list.

### Multi-Agent Evaluation (5 samples)

Located in [`samples/cowork/`](../samples/cowork/). Patterns for [Claude Cowork](https://claude.com/product/cowork), [Agent Teams](https://code.claude.com/docs/en/agent-teams), or any multi-agent framework where multiple agents collaborate and each agent's output needs independent quality assessment.

Key samples:
- [`multi_agent_eval.py`](../samples/cowork/multi_agent_eval.py) -- Generator-Evaluator pattern
- [`code_review.py`](../samples/cowork/code_review.py) -- Instrumentor-Reviewer pattern
- [`rag_assessment.py`](../samples/cowork/rag_assessment.py) -- RAG quality evaluation

See the [Multi-Agent README](../samples/cowork/README.md) for the full list.

### CI/CD Integration (2 samples + workflow)

Located in [`samples/cicd/`](../samples/cicd/). Embed evaluation quality gates into your build and deployment pipelines so regressions never reach production.

- [`quality_gate.py`](../samples/cicd/quality_gate.py) -- Gate deployments on evaluation pass rates
- [`pre_commit_hook.py`](../samples/cicd/pre_commit_hook.py) -- Catch regressions at commit time
- [`github_actions_gate.yml`](../samples/cicd/github_actions_gate.yml) -- Drop-in GitHub Actions workflow

See the [CI/CD README](../samples/cicd/README.md) for details.

### LLM Provider Integrations (2 samples)

Located in [`samples/integrations/`](../samples/integrations/). Trace and evaluate outputs from OpenAI and Anthropic with minimal instrumentation.

- [`openai_traced.py`](../samples/integrations/openai_traced.py) -- Trace an OpenAI completion and evaluate it
- [`anthropic_traced.py`](../samples/integrations/anthropic_traced.py) -- Capture multi-turn Claude conversations

### Content-Type Evaluations (3 samples)

Located in [`samples/modalities/`](../samples/modalities/). Apply specialized judges to different content types -- text responses, brand assets, and structured documents.

- [`text_evaluation.py`](../samples/modalities/text_evaluation.py) -- Score text across safety, relevance, and compliance
- [`brand_evaluation.py`](../samples/modalities/brand_evaluation.py) -- Enforce brand voice consistency
- [`document_evaluation.py`](../samples/modalities/document_evaluation.py) -- Validate document extraction accuracy

### OpenClaw Agent Evaluation (10 demos + skill)

Located in [`samples/openclaw/`](../samples/openclaw/). Trace, evaluate, and monitor [OpenClaw](https://openclaw.ai/) autonomous AI agents using LayerLens -- including cage match model tournaments, code gating, drift detection, content auditing, honeypot skill auditing, and adversarial red-teaming.

See the [OpenClaw README](../samples/openclaw/README.md) for the full list of integration samples and advanced evaluation patterns.

### MCP Server (1 sample)

Located in [`samples/mcp/`](../samples/mcp/). Expose LayerLens capabilities as tools for Claude, Cursor, and any MCP-compatible AI assistant.

- [`layerlens_server.py`](../samples/mcp/layerlens_server.py) -- MCP server with trace management, judge creation, and evaluation execution

See the [MCP README](../samples/mcp/README.md) for setup instructions.

### CopilotKit Integration (2 agents + UI components)

Located in [`samples/copilotkit/`](../samples/copilotkit/). Full-stack integration with CopilotKit using LangGraph CoAgents and generative UI card components.

- [`agents/evaluator_agent.py`](../samples/copilotkit/agents/evaluator_agent.py) -- LangGraph CoAgent for evaluation workflows
- [`agents/investigator_agent.py`](../samples/copilotkit/agents/investigator_agent.py) -- LangGraph CoAgent for trace investigation
- [`components/*.tsx`](../samples/copilotkit/components/) -- React card components for rendering results
- [`hooks/*.ts`](../samples/copilotkit/hooks/) -- CopilotKit hooks for wiring LayerLens actions

See the [CopilotKit README](../samples/copilotkit/README.md) for the full list.

### Claude Code Skills (6 skills)

Located in [`samples/claude-code/`](../samples/claude-code/). Slash commands that bring LayerLens workflows directly into the Claude Code CLI -- manage traces, judges, evaluations, optimizations, benchmarks, and investigations without leaving your terminal.

See the [Claude Code Skills README](../samples/claude-code/README.md) for the full list.

### Sample Data

Located in [`samples/data/`](../samples/data/). Pre-built trace files, test datasets, and 16 industry-specific evaluation datasets so you can run every sample without generating your own data first.

See the [Sample Data README](../samples/data/README.md) for contents.

## Full Sample Reference

For the complete table of every sample with descriptions, see the [samples README](../samples/README.md).
