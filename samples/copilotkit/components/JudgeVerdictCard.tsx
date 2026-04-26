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

const SEVERITY_DOT: Record<Severity, string> = {
  critical: "bg-red-500",
  high: "bg-orange-500",
  medium: "bg-amber-500",
  low: "bg-muted-foreground",
};

function verdictBadge(verdict: Verdict) {
  if (verdict === "pass") {
    return (
      <Badge
        variant="secondary"
        className="bg-green-50 text-green-700 dark:bg-green-900/20 dark:text-green-400"
      >
        Pass
      </Badge>
    );
  }
  if (verdict === "fail") {
    return <Badge variant="destructive">Fail</Badge>;
  }
  return (
    <Badge
      variant="secondary"
      className="bg-amber-50 text-amber-700 dark:bg-amber-900/20 dark:text-amber-400"
    >
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

  return (
    <Card className="gap-0 py-0 transition-shadow duration-200 hover:shadow-md">
      <CardHeader className="px-6 py-5">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0 space-y-1">
            <CardTitle className="truncate text-base">{judgeName}</CardTitle>
            <span
              className="inline-flex items-center gap-1.5 text-xs uppercase tracking-wide text-muted-foreground"
              title={`Severity: ${severity}`}
            >
              <span aria-hidden className={cn("h-1.5 w-1.5 rounded-full", SEVERITY_DOT[severity])} />
              {severity}
            </span>
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

        <div className="space-y-1">
          <p className="text-xs font-medium text-muted-foreground">Reasoning</p>
          <p className="text-sm leading-relaxed text-foreground">{displayedReasoning}</p>
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
