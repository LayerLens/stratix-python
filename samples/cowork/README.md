# Multi-Agent Evaluation Samples (Cowork Patterns)

When multiple AI agents collaborate on a task -- whether through
[Claude Cowork](https://claude.com/product/cowork) sub-agent orchestration,
[Claude Code Agent Teams](https://code.claude.com/docs/en/agent-teams),
or any multi-agent framework -- each agent's contribution needs independent
quality assessment. A generator agent might produce fluent but factually wrong text.
A code reviewer might miss a security vulnerability. A RAG retriever might surface
irrelevant documents.

These samples demonstrate multi-agent evaluation patterns where LayerLens judges
score each agent's output independently, and evaluation results drive the coordination
between agents. The patterns apply to any multi-agent architecture:

- **Claude Cowork**: sub-agents working on parallel workstreams with shared files
- **Claude Code Agent Teams**: coordinated Claude Code sessions with a team lead
- **LangGraph / CrewAI / AutoGen**: any framework-based multi-agent system
- **Custom pipelines**: hand-rolled agent orchestration in your own code

## Prerequisites

```bash
pip install layerlens --index-url https://sdk.layerlens.ai/package
export LAYERLENS_STRATIX_API_KEY=your-api-key
```

## Quick Start

Start with `multi_agent_eval.py` for the simplest two-agent pattern:

```bash
python multi_agent_eval.py
```

Expected output: a Generator agent produces a response, an Evaluator agent scores it
using safety and factual accuracy judges via the LayerLens SDK, and the combined results
are printed with per-judge verdicts.

## Samples

| File | Pattern | Description |
|------|---------|-------------|
| `multi_agent_eval.py` | **Generator-Evaluator** | One agent produces responses, a second agent scores them with safety and factual accuracy judges. The foundational pattern for any multi-agent evaluation workflow. |
| `code_review.py` | **Instrumentor-Reviewer** | One agent uploads code traces, a reviewer agent evaluates them with code execution, security, and metrics judges. Applicable to AI-assisted code generation in Cowork or Agent Teams. |
| `pair_programming.py` | **Rubric Writer-Tester** | One agent creates a judge, the other tests it against traces, and they iteratively refine the judge via `judges.update()`. Demonstrates collaborative judge development. |
| `rag_assessment.py` | **RAG Runner-Quality Judge** | One agent executes retrieval queries, the other evaluates groundedness and retrieval quality. Useful for monitoring RAG accuracy in any pipeline. |
| `incident_response.py` | **Detector-Responder** | A Detector agent evaluates recent traces for safety violations and a Responder agent triages flagged traces with additional targeted judges. Models automated incident triage. |

## Expected Behavior

Each sample simulates a multi-agent interaction and prints evaluation results from each
stage. Agent roles, trace IDs, and judge verdicts are clearly labeled in the output. No
external agent framework is required -- all coordination is implemented directly in each
script using the LayerLens SDK. In production, the shared state dict would be replaced
by your agent framework's coordination mechanism (Cowork shared files, Agent Teams
messaging, LangGraph state, etc.).
