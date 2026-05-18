/**
 * useLayerLensActions — Registers CopilotKit actions for LayerLens operations.
 *
 * Each action maps to a LayerLens API endpoint and is exposed to the
 * CopilotKit assistant so it can drive the platform on behalf of the user.
 */

import { useCopilotAction } from "@copilotkit/react-core";

// ---------------------------------------------------------------------------
// Shared types & helpers
// ---------------------------------------------------------------------------

export interface UseLayerLensActionsOptions {
  /** LayerLens API base URL. Defaults to "/api/v1" */
  apiBaseUrl?: string;
  /** Bearer token for authentication. */
  apiKey?: string;
  /** Callback invoked when the dashboard should navigate somewhere. */
  onNavigate?: (path: string) => void;
}

interface FetchOptions {
  method?: string;
  body?: unknown;
}

function buildHeaders(apiKey?: string): Record<string, string> {
  const h: Record<string, string> = {
    "Content-Type": "application/json",
    Accept: "application/json",
  };
  if (apiKey) {
    h["Authorization"] = `Bearer ${apiKey}`;
  }
  return h;
}

async function apiFetch<T = unknown>(
  base: string,
  path: string,
  apiKey?: string,
  opts: FetchOptions = {},
): Promise<T> {
  const url = `${base.replace(/\/$/, "")}${path}`;
  const res = await fetch(url, {
    method: opts.method ?? "GET",
    headers: buildHeaders(apiKey),
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(
      `LayerLens API error ${res.status}: ${text.slice(0, 200)}`,
    );
  }
  const text = await res.text();
  if (!text) {
    return {} as T;
  }
  try {
    return JSON.parse(text) as T;
  } catch {
    throw new Error(
      `LayerLens API returned non-JSON response: ${text.slice(0, 200)}`,
    );
  }
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useLayerLensActions(options: UseLayerLensActionsOptions = {}) {
  const { apiBaseUrl = "/api/v1", apiKey, onNavigate } = options;

  // ---- navigate_to_trace ----
  useCopilotAction({
    name: "navigate_to_trace",
    description:
      "Navigate the LayerLens dashboard to a specific trace by its ID. " +
      "Use this when the user asks to view, inspect, or open a trace.",
    parameters: [
      {
        name: "traceId",
        type: "string",
        description: "The trace ID to navigate to.",
        required: true,
      },
      {
        name: "view",
        type: "string",
        description:
          'Dashboard view to open. One of "explorer", "agentgraph".',
        required: false,
      },
    ],
    handler: async ({ traceId, view }) => {
      const segment = view === "agentgraph" ? "agentgraph" : "traces";
      const path = `/${segment}/${traceId}`;
      onNavigate?.(path);
      return { navigated: true, path };
    },
  });

  // ---- run_evaluation ----
  useCopilotAction({
    name: "run_evaluation",
    description:
      "Start a new evaluation run against one or more traces. " +
      "Returns the evaluation ID so results can be polled later.",
    parameters: [
      {
        name: "name",
        type: "string",
        description: "Human-readable name for the evaluation run.",
        required: true,
      },
      {
        name: "traceIds",
        type: "string[]",
        description: "Array of trace IDs to evaluate.",
        required: true,
      },
      {
        name: "judgeIds",
        type: "string[]",
        description: "Array of judge IDs to apply. If empty, uses defaults.",
        required: false,
      },
    ],
    handler: async ({ name, traceIds, judgeIds }) => {
      const result = await apiFetch(apiBaseUrl, "/evaluate", apiKey, {
        method: "POST",
        body: {
          name,
          trace_ids: traceIds,
          judge_ids: judgeIds ?? [],
        },
      });
      return result;
    },
  });

  // ---- create_judge ----
  useCopilotAction({
    name: "create_judge",
    description:
      "Create a new AI judge with a given name, criteria, and severity. " +
      "Returns the created judge object.",
    parameters: [
      {
        name: "name",
        type: "string",
        description: "Name for the new judge.",
        required: true,
      },
      {
        name: "criteria",
        type: "string",
        description: "Evaluation criteria the judge should apply.",
        required: true,
      },
      {
        name: "severity",
        type: "string",
        description:
          'Default severity for findings: "critical", "high", "medium", or "low".',
        required: false,
      },
      {
        name: "rubric",
        type: "string",
        description: "Optional detailed rubric for scoring.",
        required: false,
      },
    ],
    handler: async ({ name, criteria, severity, rubric }) => {
      const result = await apiFetch(apiBaseUrl, "/judges", apiKey, {
        method: "POST",
        body: {
          name,
          criteria,
          severity: severity ?? "medium",
          rubric: rubric ?? "",
        },
      });
      return result;
    },
  });

  // ---- list_traces ----
  useCopilotAction({
    name: "list_traces",
    description:
      "List recent traces with optional filters. Returns an array of trace summaries.",
    parameters: [
      {
        name: "limit",
        type: "number",
        description: "Maximum number of traces to return (default 10).",
        required: false,
      },
      {
        name: "framework",
        type: "string",
        description:
          "Filter by framework (e.g. langchain, crewai, autogen).",
        required: false,
      },
      {
        name: "status",
        type: "string",
        description: 'Filter by status: "ok", "error", "timeout".',
        required: false,
      },
    ],
    handler: async ({ limit, framework, status }) => {
      const params = new URLSearchParams();
      if (limit) params.set("limit", String(limit));
      if (framework) params.set("framework", framework);
      if (status) params.set("status", status);
      const qs = params.toString();
      const path = `/traces${qs ? `?${qs}` : ""}`;
      const result = await apiFetch(apiBaseUrl, path, apiKey);
      return result;
    },
  });

  // ---- search_traces ----
  useCopilotAction({
    name: "search_traces",
    description:
      "Search traces using a natural-language query or structured filters.",
    parameters: [
      {
        name: "query",
        type: "string",
        description: "Natural-language search query.",
        required: true,
      },
      {
        name: "startDate",
        type: "string",
        description: "ISO-8601 start date filter.",
        required: false,
      },
      {
        name: "endDate",
        type: "string",
        description: "ISO-8601 end date filter.",
        required: false,
      },
      {
        name: "limit",
        type: "number",
        description: "Maximum results to return.",
        required: false,
      },
    ],
    handler: async ({ query, startDate, endDate, limit }) => {
      const result = await apiFetch(apiBaseUrl, "/traces/search", apiKey, {
        method: "POST",
        body: {
          query,
          start_date: startDate,
          end_date: endDate,
          limit: limit ?? 20,
        },
      });
      return result;
    },
  });

  // ---- export_data ----
  useCopilotAction({
    name: "export_data",
    description:
      "Export traces, evaluations, or other data as CSV, JSON, or Parquet. " +
      "Returns an export job ID that can be polled for completion.",
    parameters: [
      {
        name: "source",
        type: "string",
        description:
          'Data source to export: "traces", "evaluations", "feedback".',
        required: true,
      },
      {
        name: "format",
        type: "string",
        description: 'Output format: "csv", "json", or "parquet".',
        required: true,
      },
      {
        name: "filters",
        type: "object",
        description:
          "Optional filter object (e.g. {framework: 'langchain', status: 'error'}).",
        required: false,
      },
    ],
    handler: async ({ source, format, filters }) => {
      const result = await apiFetch(apiBaseUrl, "/exports", apiKey, {
        method: "POST",
        body: { source, format, filters: filters ?? {} },
      });
      return result;
    },
  });

  // ---- view_agent_graph ----
  useCopilotAction({
    name: "view_agent_graph",
    description:
      "Navigate to the agent graph visualization for a specific trace.",
    parameters: [
      {
        name: "traceId",
        type: "string",
        description: "The trace ID whose agent graph to display.",
        required: true,
      },
    ],
    handler: async ({ traceId }) => {
      const path = `/agentgraph/${traceId}`;
      onNavigate?.(path);
      return { navigated: true, path };
    },
  });

  // ---- replay_trace ----
  useCopilotAction({
    name: "replay_trace",
    description:
      "Replay a trace with optional modifications to the prompt or model. " +
      "Creates a new trace as a replay of the original.",
    parameters: [
      {
        name: "traceId",
        type: "string",
        description: "The original trace ID to replay.",
        required: true,
      },
      {
        name: "modifications",
        type: "object",
        description:
          "Optional modifications: {prompt?: string, model?: string, temperature?: number}.",
        required: false,
      },
    ],
    handler: async ({ traceId, modifications }) => {
      const result = await apiFetch(apiBaseUrl, "/replay", apiKey, {
        method: "POST",
        body: {
          trace_id: traceId,
          modifications: modifications ?? {},
        },
      });
      return result;
    },
  });
}

export default useLayerLensActions;
