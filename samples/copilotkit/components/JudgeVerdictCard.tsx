/**
 * JudgeVerdictCard — Single judge's verdict on a single case.
 *
 * Built on shadcn/ui ``Card`` + ``Badge``. Severity is rendered as a
 * colored dot + label combo (the same pattern CopilotKit's banking
 * sample uses for transaction direction indicators).
 */

import * as React from "react";

import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { MarkdownLite } from "./markdown-lite";

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

const SEVERITY_STYLES: Record<
  Severity,
  { cls: string; label: string }
> = {
  critical: {
    cls: "bg-red-600 text-white dark:bg-red-500/90",
    label: "Critical",
  },
  high: {
    cls: "bg-orange-500 text-white dark:bg-orange-500/90",
    label: "High",
  },
  medium: {
    cls: "bg-amber-500 text-white dark:bg-amber-500/90",
    label: "Medium",
  },
  low: {
    cls: "bg-muted text-muted-foreground",
    label: "Low",
  },
};

/** Triangle "alert" glyph — communicates severity-as-warning, not
 *  severity-as-direction (chevrons would imply trend). */
const AlertIcon = (props: React.SVGProps<SVGSVGElement>) => (
  <svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth={2}
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden
    {...props}
  >
    <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
    <line x1="12" y1="9" x2="12" y2="13" />
    <line x1="12" y1="17" x2="12.01" y2="17" />
  </svg>
);

function verdictBadge(verdict: Verdict) {
  if (verdict === "pass") {
    return (
      <Badge className="bg-green-600 px-2.5 py-0.5 text-white shadow-sm hover:bg-green-600 dark:bg-green-500/90">
        Pass
      </Badge>
    );
  }
  if (verdict === "fail") {
    return (
      <Badge className="bg-red-600 px-2.5 py-0.5 text-white shadow-sm hover:bg-red-600 dark:bg-red-500/90">
        Fail
      </Badge>
    );
  }
  return (
    <Badge className="bg-amber-600 px-2.5 py-0.5 text-white shadow-sm hover:bg-amber-600 dark:bg-amber-500/90">
      Error
    </Badge>
  );
}

function scoreBarColor(score: number): string {
  if (score >= 0.8) return "bg-green-500";
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
  const [expanded, setExpanded] = React.useState(false);
  const needsCollapse = reasoning.length > REASONING_COLLAPSE_THRESHOLD;
  const displayedReasoning =
    needsCollapse && !expanded
      ? reasoning.slice(0, REASONING_COLLAPSE_THRESHOLD) + "…"
      : reasoning;

  // Hide the severity chip when there's nothing meaningful to flag —
  // a passed verdict with "low" severity has no concerns to surface.
  const showSeverity = !(verdict === "pass" && severity === "low");
  const sev = SEVERITY_STYLES[severity];

  return (
    <Card className="gap-0 py-0 transition-shadow duration-200 hover:shadow-md">
      <CardHeader className="px-6 py-5">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0 space-y-1.5">
            <CardTitle className="truncate text-base">{judgeName}</CardTitle>
            {showSeverity ? (
              <span
                className={cn(
                  "inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[11px] font-medium",
                  sev.cls,
                )}
                title={`Severity: ${sev.label}`}
              >
                <AlertIcon className="h-3 w-3" />
                {sev.label}
              </span>
            ) : null}
          </div>
          {verdictBadge(verdict)}
        </div>
      </CardHeader>

      <Separator />

      <CardContent className="space-y-4 px-6 py-5">
        <div className="space-y-1.5">
          <div className="flex items-baseline justify-between">
            <span className="text-xs font-medium text-muted-foreground">Score</span>
            <span className="text-sm font-semibold tabular-nums">
              {(score * 100).toFixed(0)}%
            </span>
          </div>
          <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
            <div
              className={cn("h-full rounded-full transition-all duration-500", scoreBarColor(score))}
              style={{ width: `${score * 100}%` }}
            />
          </div>
        </div>

        <div className="space-y-2">
          <p className="text-xs font-medium text-muted-foreground">Reasoning</p>
          <MarkdownLite text={displayedReasoning} />
          {needsCollapse ? (
            <Button
              variant="link"
              size="sm"
              onClick={() => setExpanded((p) => !p)}
              className="h-auto px-0 text-xs"
            >
              {expanded ? "Show less" : "Show more"}
            </Button>
          ) : null}
        </div>

        {evidence.length > 0 ? (
          <div className="space-y-2">
            <p className="text-xs font-medium text-muted-foreground">Evidence</p>
            <div className="overflow-x-auto rounded-md border">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b bg-muted/40">
                    <th className="px-2 py-1.5 text-left font-medium text-muted-foreground">Field</th>
                    <th className="px-2 py-1.5 text-left font-medium text-muted-foreground">Expected</th>
                    <th className="px-2 py-1.5 text-left font-medium text-muted-foreground">Actual</th>
                  </tr>
                </thead>
                <tbody>
                  {evidence.map((e, i) => (
                    <tr key={i} className="border-b last:border-0">
                      <td className="px-2 py-1.5 font-mono text-foreground">{e.field}</td>
                      <td className="px-2 py-1.5 text-green-700 dark:text-green-400">{e.expected}</td>
                      <td className="px-2 py-1.5 text-red-700 dark:text-red-400">{e.actual}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
};

export default JudgeVerdictCard;
