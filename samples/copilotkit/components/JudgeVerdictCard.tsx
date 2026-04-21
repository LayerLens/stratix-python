/**
 * JudgeVerdictCard — Renders an individual judge verdict inline in CopilotKit chat.
 *
 * Displays pass/fail badge, score bar, reasoning (collapsible), evidence table,
 * and severity indicator.
 */

import React, { useState } from "react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type Verdict = "pass" | "fail" | "error";
export type Severity = "critical" | "high" | "medium" | "low";

export interface Evidence {
  field: string;
  expected: string;
  actual: string;
}

export interface JudgeVerdictCardProps {
  judgeName: string;
  verdict: Verdict;
  score: number; // 0–1
  reasoning: string;
  evidence: Evidence[];
  severity: Severity;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const VERDICT_STYLES: Record<Verdict, { label: string; cls: string }> = {
  pass: {
    label: "Pass",
    cls: "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300",
  },
  fail: {
    label: "Fail",
    cls: "bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300",
  },
  error: {
    label: "Error",
    cls: "bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300",
  },
};

const SEVERITY_STYLES: Record<Severity, { icon: string; cls: string }> = {
  critical: {
    icon: "\u25C6", // diamond
    cls: "text-red-600 dark:text-red-400",
  },
  high: {
    icon: "\u25B2", // triangle up
    cls: "text-orange-600 dark:text-orange-400",
  },
  medium: {
    icon: "\u25CF", // circle
    cls: "text-amber-500 dark:text-amber-400",
  },
  low: {
    icon: "\u25CB", // circle outline
    cls: "text-gray-400 dark:text-gray-500",
  },
};

function scoreBarColor(score: number): string {
  if (score >= 0.8) return "bg-emerald-500";
  if (score >= 0.6) return "bg-amber-500";
  return "bg-red-500";
}

const REASONING_COLLAPSE_THRESHOLD = 180; // characters

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export const JudgeVerdictCard: React.FC<JudgeVerdictCardProps> = ({
  judgeName,
  verdict,
  score,
  reasoning,
  evidence,
  severity,
}) => {
  const [expanded, setExpanded] = useState(false);
  const vStyle = VERDICT_STYLES[verdict];
  const sStyle = SEVERITY_STYLES[severity];
  const needsCollapse = reasoning.length > REASONING_COLLAPSE_THRESHOLD;
  const displayedReasoning =
    needsCollapse && !expanded
      ? reasoning.slice(0, REASONING_COLLAPSE_THRESHOLD) + "\u2026"
      : reasoning;

  return (
    <div className="w-full max-w-md overflow-hidden rounded-xl border border-gray-200 bg-white shadow-sm dark:border-gray-700 dark:bg-gray-800">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-gray-100 px-4 py-3 dark:border-gray-700">
        <div className="flex items-center gap-2 min-w-0">
          <h3 className="truncate text-sm font-semibold text-gray-900 dark:text-gray-100">
            {judgeName}
          </h3>
          <span className={`inline-flex items-center gap-1 ${sStyle.cls}`} title={`Severity: ${severity}`}>
            <span className="text-xs">{sStyle.icon}</span>
            <span className="text-[10px] font-medium uppercase">{severity}</span>
          </span>
        </div>
        <span
          className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-bold ${vStyle.cls}`}
        >
          {vStyle.label}
        </span>
      </div>

      {/* Body */}
      <div className="px-4 py-3 space-y-3">
        {/* Score bar */}
        <div>
          <div className="mb-1 flex items-baseline justify-between">
            <span className="text-xs font-medium text-gray-500 dark:text-gray-400">
              Score
            </span>
            <span className="text-sm font-bold tabular-nums text-gray-900 dark:text-gray-100">
              {(score * 100).toFixed(0)}%
            </span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-gray-200 dark:bg-gray-700">
            <div
              className={`h-full rounded-full transition-all ${scoreBarColor(score)}`}
              style={{ width: `${score * 100}%` }}
            />
          </div>
        </div>

        {/* Reasoning */}
        <div>
          <p className="mb-0.5 text-xs font-medium text-gray-500 dark:text-gray-400">
            Reasoning
          </p>
          <p className="text-xs leading-relaxed text-gray-700 dark:text-gray-300">
            {displayedReasoning}
          </p>
          {needsCollapse && (
            <button
              type="button"
              onClick={() => setExpanded((prev) => !prev)}
              className="mt-1 text-xs font-medium text-indigo-600 hover:text-indigo-500 dark:text-indigo-400"
            >
              {expanded ? "Show less" : "Show more"}
            </button>
          )}
        </div>

        {/* Evidence table */}
        {evidence.length > 0 && (
          <div>
            <p className="mb-1 text-xs font-medium text-gray-500 dark:text-gray-400">
              Evidence
            </p>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-gray-100 dark:border-gray-700">
                    <th className="pb-1 pr-3 text-left font-medium text-gray-500 dark:text-gray-400">
                      Field
                    </th>
                    <th className="pb-1 pr-3 text-left font-medium text-gray-500 dark:text-gray-400">
                      Expected
                    </th>
                    <th className="pb-1 text-left font-medium text-gray-500 dark:text-gray-400">
                      Actual
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {evidence.map((e, i) => (
                    <tr
                      key={i}
                      className="border-b border-gray-50 last:border-0 dark:border-gray-700/50"
                    >
                      <td className="py-1.5 pr-3 font-mono text-gray-600 dark:text-gray-300">
                        {e.field}
                      </td>
                      <td className="py-1.5 pr-3 text-emerald-600 dark:text-emerald-400">
                        {e.expected}
                      </td>
                      <td className="py-1.5 text-red-600 dark:text-red-400">
                        {e.actual}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default JudgeVerdictCard;
