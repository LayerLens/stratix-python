/**
 * EvaluationCard — Renders evaluation results inline in CopilotKit chat.
 *
 * Displays pass rate, case breakdown, score distribution, and an optional
 * CSS-only sparkline trend.  Designed for embedding inside CopilotKit's
 * message renderer so that the AI assistant can surface eval results
 * without the user leaving the chat window.
 */

import React, { useState } from "react";

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
  /** Base URL of the LayerLens Stratix dashboard. Defaults to "/" */
  dashboardBaseUrl?: string;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function passRateColor(rate: number): string {
  if (rate >= 80) return "text-emerald-500";
  if (rate >= 60) return "text-amber-500";
  return "text-red-500";
}

function passRateBg(rate: number): string {
  if (rate >= 80) return "bg-emerald-500";
  if (rate >= 60) return "bg-amber-500";
  return "bg-red-500";
}

function statusBadge(status: EvaluationStatus) {
  const map: Record<EvaluationStatus, { label: string; cls: string }> = {
    running: {
      label: "Running",
      cls: "bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300",
    },
    completed: {
      label: "Completed",
      cls: "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300",
    },
    failed: {
      label: "Failed",
      cls: "bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300",
    },
  };
  const { label, cls } = map[status];
  return (
    <span
      className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${cls}`}
    >
      {status === "running" && (
        <span className="mr-1.5 h-1.5 w-1.5 animate-pulse rounded-full bg-blue-500" />
      )}
      {label}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

/** CSS-only sparkline using inline column heights. */
function Sparkline({ data }: { data: TrendPoint[] }) {
  if (data.length === 0) return null;
  const max = Math.max(...data.map((d) => d.rate), 1);

  return (
    <div className="mt-3">
      <p className="mb-1 text-xs font-medium text-gray-500 dark:text-gray-400">
        Trend
      </p>
      <div className="flex items-end gap-px" style={{ height: 32 }}>
        {data.map((point, i) => {
          const pct = (point.rate / max) * 100;
          return (
            <div
              key={i}
              title={`${point.date}: ${point.rate.toFixed(1)}%`}
              className={`flex-1 rounded-t transition-all ${passRateBg(point.rate)}`}
              style={{
                height: `${Math.max(pct, 4)}%`,
                opacity: 0.6 + 0.4 * (pct / 100),
              }}
            />
          );
        })}
      </div>
    </div>
  );
}

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
    <div className="mt-3">
      <div className="mb-1 flex justify-between text-xs text-gray-500 dark:text-gray-400">
        <span>{passed} passed</span>
        <span>{failed} failed</span>
        {errors > 0 && <span>{errors} errors</span>}
      </div>
      <div className="flex h-2 overflow-hidden rounded-full bg-gray-200 dark:bg-gray-700">
        <div
          className="bg-emerald-500 transition-all"
          style={{ width: `${pPct}%` }}
        />
        <div
          className="bg-red-500 transition-all"
          style={{ width: `${fPct}%` }}
        />
        {errors > 0 && (
          <div
            className="bg-amber-500 transition-all"
            style={{ width: `${ePct}%` }}
          />
        )}
      </div>
    </div>
  );
}

function ScoreDistribution({ scores }: { scores: Score[] }) {
  if (scores.length === 0) return null;

  return (
    <div className="mt-3">
      <p className="mb-1.5 text-xs font-medium text-gray-500 dark:text-gray-400">
        Scores
      </p>
      <div className="space-y-1.5">
        {scores.map((s) => (
          <div key={s.label} className="flex items-center gap-2 text-xs">
            <span className="w-24 truncate text-gray-600 dark:text-gray-300">
              {s.label}
            </span>
            <div className="flex-1">
              <div className="h-1.5 overflow-hidden rounded-full bg-gray-200 dark:bg-gray-700">
                <div
                  className="h-full rounded-full bg-indigo-500 transition-all"
                  style={{ width: `${s.value * 100}%` }}
                />
              </div>
            </div>
            <span className="w-8 text-right font-mono text-gray-500 dark:text-gray-400">
              {(s.value * 100).toFixed(0)}%
            </span>
          </div>
        ))}
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
  dashboardBaseUrl = "/",
}) => {
  const dashUrl = `${dashboardBaseUrl.replace(/\/$/, "")}/evaluations/${evaluationId}`;

  return (
    <div className="w-full max-w-md overflow-hidden rounded-xl border border-gray-200 bg-white shadow-sm dark:border-gray-700 dark:bg-gray-800">
      {/* Header */}
      <div className="flex items-start justify-between border-b border-gray-100 px-4 py-3 dark:border-gray-700">
        <div className="min-w-0">
          <h3 className="truncate text-sm font-semibold text-gray-900 dark:text-gray-100">
            {name}
          </h3>
          <p className="mt-0.5 truncate text-xs text-gray-400 font-mono">
            {evaluationId}
          </p>
        </div>
        {statusBadge(status)}
      </div>

      {/* Body */}
      <div className="px-4 py-3">
        {/* Pass rate hero */}
        <div className="flex items-baseline gap-1.5">
          <span className={`text-3xl font-bold tabular-nums ${passRateColor(passRate)}`}>
            {passRate.toFixed(1)}%
          </span>
          <span className="text-sm text-gray-400">
            pass rate ({totalCases} cases)
          </span>
        </div>

        <CaseBar
          passed={passedCases}
          failed={failedCases}
          errors={errorCases}
          total={totalCases}
        />

        <ScoreDistribution scores={scores} />

        {trendData && trendData.length > 1 && <Sparkline data={trendData} />}
      </div>

      {/* Footer */}
      <div className="border-t border-gray-100 px-4 py-2 dark:border-gray-700">
        <a
          href={dashUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="text-xs font-medium text-indigo-600 hover:text-indigo-500 dark:text-indigo-400 dark:hover:text-indigo-300"
        >
          View in Dashboard &rarr;
        </a>
      </div>
    </div>
  );
};

export default EvaluationCard;
