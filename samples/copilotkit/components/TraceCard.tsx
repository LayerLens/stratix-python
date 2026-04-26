/**
 * TraceCard — Inline summary of a single agent trace.
 *
 * Built on shadcn/ui ``Card`` + ``Badge``; status and framework pills
 * use the ``bg-{color}-50 text-{color}-600 dark:bg-{color}-900/20``
 * pattern that CopilotKit's banking and travel samples use for
 * tinted status surfaces.
 */

import * as React from "react";

import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type TraceStatus = "ok" | "error" | "timeout" | "running";

export interface TraceCardProps {
  traceId: string;
  framework: string;
  agentName: string;
  /** Trace lifecycle status. Optional — when the data source doesn't
   *  expose this (e.g. ``traces.get_many`` snapshots without a
   *  ``status`` column), omit it rather than ship a misleading
   *  default. The status pill is hidden when undefined. */
  status?: TraceStatus;
  duration_ms: number;
  tokenCount: number;
  costUsd: number;
  eventCount: number;
  agentCount: number;
  timestamp: string; // ISO-8601
  tags?: string[];
  /** Base URL of the Stratix dashboard for "Trace Explorer" / "Agent
   *  Graph" deep-links. When omitted, the footer is hidden. */
  dashboardBaseUrl?: string;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const STATUS_STYLES: Record<TraceStatus, { dot: string; label: string }> = {
  ok: { dot: "bg-green-500", label: "OK" },
  error: { dot: "bg-red-500", label: "Error" },
  timeout: { dot: "bg-amber-500", label: "Timeout" },
  running: { dot: "bg-blue-500 animate-pulse", label: "Running" },
};

function frameworkBadgeVariant(framework: string): "secondary" | "outline" {
  // Stratix-known frameworks get the slightly more emphatic
  // ``secondary`` variant; everything else falls back to ``outline``
  // to keep the visual noise low.
  const known = new Set([
    "langchain",
    "langgraph",
    "crewai",
    "autogen",
    "openai",
    "anthropic",
    "stratix",
  ]);
  return known.has(framework.toLowerCase()) ? "secondary" : "outline";
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  return `${Math.floor(ms / 60_000)}m ${Math.floor((ms % 60_000) / 1000)}s`;
}

function formatTokens(t: number): string {
  if (t < 1000) return String(t);
  if (t < 1_000_000) return `${(t / 1000).toFixed(1)}k`;
  return `${(t / 1_000_000).toFixed(2)}m`;
}

function formatCost(usd: number): string {
  if (usd === 0) return "$0";
  if (usd < 0.01) return `${(usd * 100).toFixed(2)}¢`;
  return `$${usd.toFixed(2)}`;
}

function formatTimestamp(iso: string): string {
  try {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
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
    <div className="flex flex-col items-center gap-0.5">
      <span className="text-sm font-semibold tabular-nums">{value}</span>
      <span className="text-[10px] uppercase tracking-wide text-muted-foreground">
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
  dashboardBaseUrl,
}) => {
  const st = status ? STATUS_STYLES[status] : null;
  const base = dashboardBaseUrl ? dashboardBaseUrl.replace(/\/$/, "") : null;
  const traceUrl = base ? `${base}/traces/${traceId}` : null;
  const graphUrl = base ? `${base}/agentgraph/${traceId}` : null;

  return (
    <Card className="gap-0 py-0 transition-shadow duration-200 hover:shadow-md">
      <CardHeader className="px-5 py-4">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0 flex-1 space-y-1">
            <div className="flex flex-wrap items-center gap-1.5">
              <Badge variant={frameworkBadgeVariant(framework)} className="font-medium">
                {framework}
              </Badge>
              {st ? (
                <Badge variant="outline" className="gap-1.5 font-medium">
                  <span aria-hidden className={cn("h-1.5 w-1.5 rounded-full", st.dot)} />
                  {st.label}
                </Badge>
              ) : null}
            </div>
            <CardTitle className="truncate text-base">{agentName}</CardTitle>
            <p className="truncate font-mono text-[11px] text-muted-foreground">
              {traceId}
            </p>
          </div>
          <span className="shrink-0 text-[11px] text-muted-foreground">
            {formatTimestamp(timestamp)}
          </span>
        </div>
      </CardHeader>

      <Separator />

      <CardContent className="grid grid-cols-5 gap-2 px-5 py-4">
        <Metric label="Duration" value={formatDuration(duration_ms)} />
        <Metric label="Tokens" value={formatTokens(tokenCount)} />
        <Metric label="Cost" value={formatCost(costUsd)} />
        <Metric label="Events" value={String(eventCount)} />
        <Metric label="Agents" value={String(agentCount)} />
      </CardContent>

      {tags.length > 0 ? (
        <>
          <Separator />
          <div className="flex flex-wrap gap-1.5 px-5 py-3">
            {tags.map((tag) => (
              <Badge key={tag} variant="secondary" className="font-normal">
                {tag}
              </Badge>
            ))}
          </div>
        </>
      ) : null}

      {traceUrl && graphUrl ? (
        <>
          <Separator />
          <div className="flex items-center gap-4 px-5 py-3 text-xs">
            <a
              href={traceUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="font-medium text-[#6766FC] transition hover:underline"
            >
              Trace Explorer →
            </a>
            <a
              href={graphUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="font-medium text-[#6766FC] transition hover:underline"
            >
              Agent Graph →
            </a>
          </div>
        </>
      ) : null}
    </Card>
  );
};

export default TraceCard;
