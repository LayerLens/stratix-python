/**
 * useLayerLensActions — CopilotKit action hook for LayerLens operations.
 *
 * Registers 8 actions that map to the LayerLens API, allowing the CopilotKit
 * assistant to interact with traces, evaluations, judges, replays, and exports.
 *
 * Usage:
 *   import { useLayerLensActions } from "./useLayerLensActions";
 *   function App() {
 *     useLayerLensActions();
 *     return <CopilotSidebar />;
 *   }
 */

import { useCopilotAction } from "@copilotkit/react-core";

const API_BASE = process.env.NEXT_PUBLIC_LAYERLENS_API_URL || "https://api.layerlens.ai";

async function llFetch(path: string, options?: RequestInit) {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${process.env.NEXT_PUBLIC_LAYERLENS_API_KEY}`,
      ...options?.headers,
    },
  });
  if (!res.ok) throw new Error(`LayerLens API error: ${res.status} ${res.statusText}`);
  return res.json();
}

export function useLayerLensActions() {
  useCopilotAction({
    name: "listTraces",
    description: "List recent traces, optionally filtered by agent ID or search term",
    parameters: [
      { name: "limit", type: "number", description: "Max traces to return", required: false },
      { name: "agentId", type: "string", description: "Filter by agent ID", required: false },
      { name: "search", type: "string", description: "Search keyword", required: false },
    ],
    handler: async ({ limit = 10, agentId, search }) => {
      const params = new URLSearchParams({ limit: String(limit) });
      if (agentId) params.set("agent_id", agentId);
      if (search) params.set("search", search);
      return llFetch(`/v1/traces?${params}`);
    },
  });

  useCopilotAction({
    name: "getTrace",
    description: "Retrieve a single trace by its ID",
    parameters: [{ name: "traceId", type: "string", description: "Trace ID", required: true }],
    handler: async ({ traceId }) => llFetch(`/v1/traces/${traceId}`),
  });

  useCopilotAction({
    name: "runEvaluation",
    description: "Start an evaluation run with a judge and dataset",
    parameters: [
      { name: "judgeId", type: "string", description: "Judge ID", required: true },
      { name: "datasetId", type: "string", description: "Dataset ID", required: true },
    ],
    handler: async ({ judgeId, datasetId }) =>
      llFetch("/v1/evaluations", { method: "POST", body: JSON.stringify({ judge_id: judgeId, dataset_id: datasetId }) }),
  });

  useCopilotAction({
    name: "listJudges",
    description: "List all available evaluation judges",
    parameters: [],
    handler: async () => llFetch("/v1/judges"),
  });

  useCopilotAction({
    name: "createJudge",
    description: "Create a new evaluation judge with scoring criteria",
    parameters: [
      { name: "name", type: "string", description: "Judge name", required: true },
      { name: "description", type: "string", description: "Judge description", required: false },
      { name: "model", type: "string", description: "LLM model for judging", required: false },
    ],
    handler: async ({ name, description = "", model = "gpt-4o" }) =>
      llFetch("/v1/judges", { method: "POST", body: JSON.stringify({ name, description, model, pass_threshold: 0.7 }) }),
  });

  useCopilotAction({
    name: "getEvaluationResults",
    description: "Get the results of a completed evaluation",
    parameters: [{ name: "evaluationId", type: "string", description: "Evaluation ID", required: true }],
    handler: async ({ evaluationId }) => llFetch(`/v1/evaluations/${evaluationId}/results`),
  });

  useCopilotAction({
    name: "replayTrace",
    description: "Replay a trace, optionally with a different model",
    parameters: [
      { name: "traceId", type: "string", description: "Trace ID to replay", required: true },
      { name: "modelOverride", type: "string", description: "Override model for replay", required: false },
    ],
    handler: async ({ traceId, modelOverride }) =>
      llFetch("/v1/replays", { method: "POST", body: JSON.stringify({ trace_id: traceId, model_override: modelOverride }) }),
  });

  useCopilotAction({
    name: "exportData",
    description: "Export traces or evaluations as CSV, JSON, or Parquet",
    parameters: [
      { name: "exportType", type: "string", description: "What to export: traces or evaluations", required: true },
      { name: "format", type: "string", description: "Output format: csv, json, or parquet", required: false },
    ],
    handler: async ({ exportType, format = "json" }) =>
      llFetch(`/v1/exports/${exportType}?format=${format}`),
  });
}

export default useLayerLensActions;
