# CopilotKit Samples

Building AI-powered user interfaces requires more than backend evaluation logic -- it requires
real-time feedback loops between the AI backend and the frontend. These samples provide
CopilotKit CoAgents (LangGraph-based) and React components that connect LayerLens evaluation
capabilities to interactive UIs, enabling human-in-the-loop evaluation workflows where users
can review, confirm, and act on AI quality assessments in real time.

## Prerequisites

```bash
pip install layerlens --index-url https://sdk.layerlens.ai/package copilotkit langgraph pydantic mcp
npm install @copilotkit/react-core  # for frontend components
export LAYERLENS_STRATIX_API_KEY=your-api-key
```

## Quick Start

Start with the Evaluator Agent to see the core human-in-the-loop pattern:

```bash
python agents/evaluator_agent.py
```

Expected output: the agent parses an evaluation intent, selects appropriate judges, pauses
for human confirmation, executes the evaluation, and emits AG-UI protocol events at each step.

## Agents (LangGraph CoAgents)

| File | Scenario | Description |
|------|----------|-------------|
| `agents/evaluator_agent.py` | Product teams building evaluation dashboards | A multi-step evaluation workflow: parses user intent, selects judges, requests human confirmation before execution, runs evaluations, and summarizes results. Emits AG-UI events for real-time frontend rendering. |
| `agents/investigator_agent.py` | Operations teams building trace debugging UIs | Fetches a trace by ID, analyzes spans for errors, latency anomalies, and cost outliers, then generates actionable fix suggestions. Designed for integration into incident investigation dashboards. |

## React Components

| File | Description |
|------|-------------|
| `components/EvaluationCard.tsx` | Renders evaluation results with score breakdowns and judge verdicts. |
| `components/TraceCard.tsx` | Displays trace metadata, span hierarchy, and timing information. |
| `components/JudgeVerdictCard.tsx` | Shows individual judge verdicts with pass/fail indicators. |
| `components/MetricCard.tsx` | Renders a single metric with trend visualization. |
| `components/ComplianceCard.tsx` | Displays compliance status with regulation-specific details. |
| `components/index.ts` | Barrel export for all components. |

## React Hooks

| File | Description |
|------|-------------|
| `hooks/useLayerLensActions.ts` | CopilotKit action hooks for triggering evaluations and investigations. |
| `hooks/useLayerLensContext.ts` | Context hook for sharing LayerLens state across components. |
| `hooks/index.ts` | Barrel export for all hooks. |

## Architecture

```
CopilotKit Frontend (React)
    |
    v
CoAgent (LangGraph StateGraph)
    |
    v
LayerLens Python SDK (Stratix client)
    |
    v
LayerLens API
```

The agents emit AG-UI protocol events that CopilotKit renders as progress cards, confirmation
dialogs, and result summaries in the frontend. The React components and hooks are provided as
reference implementations for building your own LayerLens-powered UI.
