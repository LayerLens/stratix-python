/**
 * TraceCard — Renders a trace summary inline in CopilotKit chat.
 *
 * Displays framework badge, status, duration/cost/tokens metrics, tag chips,
 * and links to the trace explorer and agent graph views.
 */

import React from "react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type TraceStatus = "ok" | "error" | "timeout" | "running";

export interface TraceCardProps {
  traceId: string;
  framework: string;
  agentName: string;
  status: TraceStatus;
  duration_ms: number;
  tokenCount: number;
  costUsd: number;
  eventCount: number;
  agentCount: number;
  timestamp: string; // ISO-8601
  tags?: string[];
  /** Base URL of the LayerLens Stratix dashboard. Defaults to "/" */
  dashboardBaseUrl?: string;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const STATUS_STYLES: Record<TraceStatus, { dot: string; label: string; cls: string }> = {
  ok: {
    dot: "bg-emerald-500",
    label: "OK",
    cls: "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300",
  },
  error: {
    dot: "bg-red-500",
    label: "Error",
    cls: "bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300",
  },
  timeout: {
    dot: "bg-amber-500",
    label: "Timeout",
    cls: "bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300",
  },
  running: {
    dot: "bg-blue-500 animate-pulse",
    label: "Running",
    cls: "bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300",
  },
};

const FRAMEWORK_COLORS: Record<string, string> = {
  langchain: "bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300",
  langgraph: "bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300",
  crewai: "bg-purple-100 text-purple-800 dark:bg-purple-900/40 dark:text-purple-300",
  autogen: "bg-sky-100 text-sky-800 dark:bg-sky-900/40 dark:text-sky-300",
  openai: "bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300",
  anthropic: "bg-orange-100 text-orange-800 dark:bg-orange-900/40 dark:text-orange-300",
  haystack: "bg-teal-100 text-teal-800 dark:bg-teal-900/40 dark:text-teal-300",
  semantic_kernel: "bg-indigo-100 text-indigo-800 dark:bg-indigo-900/40 dark:text-indigo-300",
};

function formatDuration(ms: number): string {
  if (ms < 1_000) return `${ms}ms`;
  if (ms < 60_000) return `${(ms / 1_000).toFixed(1)}s`;
  return `${(ms / 60_000).toFixed(1)}m`;
}

function formatCost(usd: number): string {
  if (usd < 0.01) return `$${(usd * 100).toFixed(2)}c`;
  return `$${usd.toFixed(4)}`;
}

function formatTokens(count: number): string {
  if (count >= 1_000_000) return `${(count / 1_000_000).toFixed(1)}M`;
  if (count >= 1_000) return `${(count / 1_000).toFixed(1)}k`;
  return String(count);
}

function formatTimestamp(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col items-center">
      <span className="text-sm font-semibold text-gray-900 dark:text-gray-100">
        {value}
      </span>
      <span className="text-[10px] uppercase tracking-wide text-gray-400">
        {label}
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export const TraceCard: React.FC<TraceCardProps> = ({
  traceId,
  framework,
  agentName,
  status,
  duration_ms,
  tokenCount,
  costUsd,
  eventCount,
  agentCount,
  timestamp,
  tags = [],
  dashboardBaseUrl = "/",
}) => {
  const base = dashboardBaseUrl.replace(/\/$/, "");
  const traceUrl = `${base}/traces/${traceId}`;
  const graphUrl = `${base}/agentgraph/${traceId}`;
  const st = STATUS_STYLES[status];
  const fwCls =
    FRAMEWORK_COLORS[framework.toLowerCase()] ??
    "bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300";

  return (
    <div className="w-full max-w-md overflow-hidden rounded-xl border border-gray-200 bg-white shadow-sm dark:border-gray-700 dark:bg-gray-800">
      {/* Header */}
      <div className="flex items-start justify-between border-b border-gray-100 px-4 py-3 dark:border-gray-700">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span
              className={`inline-flex items-center rounded-md px-2 py-0.5 text-[11px] font-medium ${fwCls}`}
            >
              {framework}
            </span>
            <span className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium ${st.cls}`}>
              <span className={`h-1.5 w-1.5 rounded-full ${st.dot}`} />
              {st.label}
            </span>
          </div>
          <h3 className="mt-1 truncate text-sm font-semibold text-gray-900 dark:text-gray-100">
            {agentName}
          </h3>
          <p className="mt-0.5 truncate text-xs font-mono text-gray-400">
            {traceId}
          </p>
        </div>
        <span className="shrink-0 text-xs text-gray-400">
          {formatTimestamp(timestamp)}
        </span>
      </div>

      {/* Metric row */}
      <div className="grid grid-cols-5 gap-2 border-b border-gray-100 px-4 py-3 dark:border-gray-700">
        <Metric label="Duration" value={formatDuration(duration_ms)} />
        <Metric label="Tokens" value={formatTokens(tokenCount)} />
        <Metric label="Cost" value={formatCost(costUsd)} />
        <Metric label="Events" value={String(eventCount)} />
        <Metric label="Agents" value={String(agentCount)} />
      </div>

      {/* Tags */}
      {tags.length > 0 && (
        <div className="flex flex-wrap gap-1.5 border-b border-gray-100 px-4 py-2 dark:border-gray-700">
          {tags.map((tag) => (
            <span
              key={tag}
              className="inline-flex rounded-full bg-gray-100 px-2 py-0.5 text-[11px] font-medium text-gray-600 dark:bg-gray-700 dark:text-gray-300"
            >
              {tag}
            </span>
          ))}
        </div>
      )}

      {/* Footer links */}
      <div className="flex items-center gap-4 px-4 py-2">
        <a
          href={traceUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="text-xs font-medium text-indigo-600 hover:text-indigo-500 dark:text-indigo-400"
        >
          Trace Explorer &rarr;
        </a>
        <a
          href={graphUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="text-xs font-medium text-indigo-600 hover:text-indigo-500 dark:text-indigo-400"
        >
          Agent Graph &rarr;
        </a>
      </div>
    </div>
  );
};

export default TraceCard;
