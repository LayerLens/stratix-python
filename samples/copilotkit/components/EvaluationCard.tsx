/**
 * EvaluationCard — Evaluation run summary.
 *
 * Built on shadcn/ui ``Card`` + ``Badge`` + ``Progress``. The pass-rate
 * "hero" is intentionally small (matching the Pinterest-style numeric
 * stats in CopilotKit's banking sample) — if you want a bigger hero
 * for a stand-alone evaluation page, wrap multiple ``EvaluationCard``s
 * in a parent grid rather than re-styling the card itself.
 */

import * as React from "react";

import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface Score {
  label: string;
  value: number; // 0–1
}

export interface TrendPoint {
  date: string; // ISO-8601 or human-readable label
  rate: number; // 0–100
}

export type EvaluationStatus = "running" | "completed" | "failed";

export interface EvaluationCardProps {
  evaluationId: string;
  name: string;
  passRate: number; // 0–100
  totalCases: number;
  passedCases: number;
  failedCases: number;
  errorCases: number;
  scores: Score[];
  trendData?: TrendPoint[];
  status: EvaluationStatus;
  /** Base URL of the Stratix dashboard. When omitted the footer link
   *  is hidden — use this when the dashboard route is not yet
   *  deployed. */
  dashboardBaseUrl?: string;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function passRateColor(rate: number): string {
  if (rate >= 80) return "text-green-600 dark:text-green-400";
  if (rate >= 60) return "text-amber-600 dark:text-amber-400";
  return "text-red-600 dark:text-red-400";
}

function statusBadge(status: EvaluationStatus) {
  if (status === "running") {
    return (
      <Badge
        variant="secondary"
        className="gap-1.5 bg-blue-50 text-blue-700 dark:bg-blue-900/20 dark:text-blue-400"
      >
        <span aria-hidden className="h-1.5 w-1.5 animate-pulse rounded-full bg-blue-500" />
        Running
      </Badge>
    );
  }
  if (status === "completed") {
    return (
      <Badge
        variant="secondary"
        className="bg-green-50 text-green-700 dark:bg-green-900/20 dark:text-green-400"
      >
        Completed
      </Badge>
    );
  }
  return <Badge variant="destructive">Failed</Badge>;
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function CaseBar({
  passed,
  failed,
  errors,
  total,
}: {
  passed: number;
  failed: number;
  errors: number;
  total: number;
}) {
  if (total === 0) return null;
  const pPct = (passed / total) * 100;
  const fPct = (failed / total) * 100;
  const ePct = (errors / total) * 100;
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-baseline gap-x-4 text-xs">
        <span className="inline-flex items-center gap-1.5">
          <span className="h-1.5 w-1.5 rounded-full bg-green-500" />
          <span className="font-medium tabular-nums">{passed}</span>
          <span className="text-muted-foreground">passed</span>
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="h-1.5 w-1.5 rounded-full bg-red-500" />
          <span className="font-medium tabular-nums">{failed}</span>
          <span className="text-muted-foreground">failed</span>
        </span>
        {errors > 0 ? (
          <span className="inline-flex items-center gap-1.5">
            <span className="h-1.5 w-1.5 rounded-full bg-amber-500" />
            <span className="font-medium tabular-nums">{errors}</span>
            <span className="text-muted-foreground">errors</span>
          </span>
        ) : null}
      </div>
      <div className="flex h-2 w-full overflow-hidden rounded-full bg-muted">
        <div className="bg-green-500 transition-all duration-500" style={{ width: `${pPct}%` }} />
        <div className="bg-red-500 transition-all duration-500" style={{ width: `${fPct}%` }} />
        {errors > 0 ? (
          <div className="bg-amber-500 transition-all duration-500" style={{ width: `${ePct}%` }} />
        ) : null}
      </div>
    </div>
  );
}

function ScoreDistribution({ scores }: { scores: Score[] }) {
  if (scores.length === 0) return null;
  return (
    <div className="space-y-2">
      <p className="text-xs font-medium text-muted-foreground">Scores</p>
      <div className="space-y-2">
        {scores.map((s) => (
          <div key={s.label} className="flex items-center gap-3 text-xs">
            <span className="w-28 truncate text-muted-foreground">{s.label}</span>
            <div className="flex-1">
              <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
                <div
                  className="h-full rounded-full bg-foreground transition-all duration-500"
                  style={{ width: `${s.value * 100}%` }}
                />
              </div>
            </div>
            <span className="w-10 text-right font-mono tabular-nums">
              {(s.value * 100).toFixed(0)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function Sparkline({ data }: { data: TrendPoint[] }) {
  if (data.length === 0) return null;
  const max = Math.max(...data.map((d) => d.rate), 1);
  return (
    <div className="space-y-2">
      <p className="text-xs font-medium text-muted-foreground">Trend</p>
      <div className="flex items-end gap-px" style={{ height: 32 }}>
        {data.map((p, i) => {
          const pct = (p.rate / max) * 100;
          const fill = p.rate >= 80 ? "bg-green-500" : p.rate >= 60 ? "bg-amber-500" : "bg-red-500";
          return (
            <div
              key={i}
              title={`${p.date}: ${p.rate.toFixed(1)}%`}
              className={cn("flex-1 rounded-t transition-all", fill)}
              style={{ height: `${Math.max(pct, 4)}%`, opacity: 0.6 + 0.4 * (pct / 100) }}
            />
          );
        })}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export const EvaluationCard: React.FC<EvaluationCardProps> = ({
  evaluationId,
  name,
  passRate,
  totalCases,
  passedCases,
  failedCases,
  errorCases,
  scores,
  trendData,
  status,
  dashboardBaseUrl,
}) => {
  const dashUrl = dashboardBaseUrl
    ? `${dashboardBaseUrl.replace(/\/$/, "")}/evaluations/${evaluationId}`
    : null;

  return (
    <Card className="gap-0 py-0 transition-shadow duration-200 hover:shadow-md">
      <CardHeader className="px-6 py-5">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0 space-y-1">
            <CardTitle className="truncate text-lg">{name}</CardTitle>
            <CardDescription className="truncate font-mono text-[11px]">
              {evaluationId}
            </CardDescription>
          </div>
          {statusBadge(status)}
        </div>
      </CardHeader>

      <Separator />

      <CardContent className="space-y-5 px-6 py-5">
        <div className="flex items-baseline gap-2">
          <span className={cn("text-3xl font-semibold tabular-nums", passRateColor(passRate))}>
            {passRate.toFixed(1)}%
          </span>
          <span className="text-sm text-muted-foreground">
            pass rate · {totalCases} {totalCases === 1 ? "case" : "cases"}
          </span>
        </div>

        <CaseBar
          passed={passedCases}
          failed={failedCases}
          errors={errorCases}
          total={totalCases}
        />

        <ScoreDistribution scores={scores} />

        {trendData && trendData.length > 1 ? <Sparkline data={trendData} /> : null}
      </CardContent>

      {dashUrl ? (
        <>
          <Separator />
          <CardFooter className="px-6 py-3">
            <a
              href={dashUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs font-medium text-[#6766FC] transition hover:underline"
            >
              View in Dashboard →
            </a>
          </CardFooter>
        </>
      ) : null}
    </Card>
  );
};

export default EvaluationCard;
