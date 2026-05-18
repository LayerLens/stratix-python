/**
 * ComplianceCard — Compliance / attestation status.
 *
 * Built on shadcn/ui ``Card`` + ``Badge``. Attestations and violations
 * use the same colored-dot + label pattern as ``JudgeVerdictCard``
 * for consistency across the SDK.
 */

import * as React from "react";

import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type ComplianceStatus = "compliant" | "partial" | "non-compliant";

export interface Attestation {
  name: string;
  status: "passed" | "pending" | "failed";
  date: string; // ISO-8601 or human-readable
}

export interface Violation {
  rule: string;
  severity: "critical" | "high" | "medium" | "low";
  description: string;
}

export interface ComplianceCardProps {
  framework: string; // e.g. "HIPAA", "SOC2", "GDPR"
  status: ComplianceStatus;
  attestations: Attestation[];
  violations: Violation[];
  lastAuditDate: string;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function statusBadge(status: ComplianceStatus) {
  if (status === "compliant") {
    return (
      <Badge
        variant="secondary"
        className="bg-green-50 text-green-700 dark:bg-green-900/20 dark:text-green-400"
      >
        Compliant
      </Badge>
    );
  }
  if (status === "partial") {
    return (
      <Badge
        variant="secondary"
        className="bg-amber-50 text-amber-700 dark:bg-amber-900/20 dark:text-amber-400"
      >
        Partial
      </Badge>
    );
  }
  return <Badge variant="destructive">Non-compliant</Badge>;
}

const ATTESTATION_DOT: Record<Attestation["status"], string> = {
  passed: "bg-green-500",
  pending: "bg-blue-500",
  failed: "bg-red-500",
};

const SEVERITY_DOT: Record<Violation["severity"], string> = {
  critical: "bg-red-500",
  high: "bg-orange-500",
  medium: "bg-amber-500",
  low: "bg-muted-foreground",
};

function formatTimestamp(iso: string): string {
  try {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    return d.toLocaleDateString(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
  } catch {
    return iso;
  }
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export const ComplianceCard: React.FC<ComplianceCardProps> = ({
  framework,
  status,
  attestations,
  violations,
  lastAuditDate,
}) => {
  return (
    <Card className="gap-0 py-0 transition-shadow duration-200 hover:shadow-md">
      <CardHeader className="px-6 py-5">
        <div className="flex items-start justify-between gap-3">
          <div className="space-y-1">
            <CardTitle className="text-base">{framework}</CardTitle>
            <CardDescription className="text-xs">
              Last audit · {formatTimestamp(lastAuditDate)}
            </CardDescription>
          </div>
          {statusBadge(status)}
        </div>
      </CardHeader>

      <Separator />

      <CardContent className="space-y-5 px-6 py-5">
        {attestations.length > 0 ? (
          <div className="space-y-2">
            <p className="text-xs font-medium text-muted-foreground">Attestations</p>
            <ul className="space-y-1.5">
              {attestations.map((a) => (
                <li
                  key={`${a.name}-${a.date}`}
                  className="flex items-center justify-between gap-3 text-sm"
                >
                  <span className="inline-flex items-center gap-2 text-foreground">
                    <span aria-hidden className={cn("h-1.5 w-1.5 rounded-full", ATTESTATION_DOT[a.status])} />
                    {a.name}
                  </span>
                  <span className="text-xs text-muted-foreground">{formatTimestamp(a.date)}</span>
                </li>
              ))}
            </ul>
          </div>
        ) : null}

        {violations.length > 0 ? (
          <div className="space-y-2">
            <p className="text-xs font-medium text-muted-foreground">
              Violations <span className="tabular-nums">({violations.length})</span>
            </p>
            <ul className="space-y-2">
              {violations.map((v, i) => (
                <li
                  key={`${v.rule}-${i}`}
                  className="rounded-md border bg-muted/30 p-3"
                >
                  <div className="flex items-baseline justify-between gap-3">
                    <span className="inline-flex items-center gap-2 text-sm font-medium text-foreground">
                      <span aria-hidden className={cn("h-1.5 w-1.5 rounded-full", SEVERITY_DOT[v.severity])} />
                      {v.rule}
                    </span>
                    <span className="text-[11px] uppercase tracking-wide text-muted-foreground">
                      {v.severity}
                    </span>
                  </div>
                  <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
                    {v.description}
                  </p>
                </li>
              ))}
            </ul>
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
};

export default ComplianceCard;
