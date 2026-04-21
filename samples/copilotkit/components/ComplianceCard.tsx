/**
 * ComplianceCard — Renders compliance/attestation status inline in CopilotKit chat.
 *
 * Displays the compliance framework badge, overall status, an attestation
 * checklist, any violations, and the last audit date.
 */

import React from "react";

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

const STATUS_DISPLAY: Record<
  ComplianceStatus,
  { label: string; cls: string; ringCls: string }
> = {
  compliant: {
    label: "Compliant",
    cls: "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300",
    ringCls: "ring-emerald-500",
  },
  partial: {
    label: "Partial",
    cls: "bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300",
    ringCls: "ring-amber-500",
  },
  "non-compliant": {
    label: "Non-Compliant",
    cls: "bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300",
    ringCls: "ring-red-500",
  },
};

const ATTESTATION_ICON: Record<Attestation["status"], { icon: string; cls: string }> = {
  passed: { icon: "\u2713", cls: "text-emerald-500" },
  pending: { icon: "\u25CB", cls: "text-amber-500" },
  failed: { icon: "\u2717", cls: "text-red-500" },
};

const SEVERITY_BADGE: Record<Violation["severity"], string> = {
  critical:
    "bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300",
  high: "bg-orange-100 text-orange-700 dark:bg-orange-900/40 dark:text-orange-300",
  medium:
    "bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300",
  low: "bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-300",
};

const FRAMEWORK_COLORS: Record<string, string> = {
  HIPAA: "bg-blue-600 text-white",
  SOC2: "bg-indigo-600 text-white",
  GDPR: "bg-purple-600 text-white",
  ISO27001: "bg-teal-600 text-white",
  NIST: "bg-cyan-600 text-white",
  PCI_DSS: "bg-rose-600 text-white",
};

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
  const sd = STATUS_DISPLAY[status];
  const fwCls =
    FRAMEWORK_COLORS[framework.toUpperCase()] ??
    "bg-gray-700 text-white dark:bg-gray-600";

  return (
    <div className="w-full max-w-md overflow-hidden rounded-xl border border-gray-200 bg-white shadow-sm dark:border-gray-700 dark:bg-gray-800">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-gray-100 px-4 py-3 dark:border-gray-700">
        <div className="flex items-center gap-2">
          <span
            className={`inline-flex items-center rounded-md px-2.5 py-1 text-xs font-bold tracking-wide ${fwCls}`}
          >
            {framework}
          </span>
          <span
            className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ${sd.cls}`}
          >
            {sd.label}
          </span>
        </div>
        <span className="text-xs text-gray-400">
          Audited {lastAuditDate}
        </span>
      </div>

      {/* Attestation checklist */}
      {attestations.length > 0 && (
        <div className="border-b border-gray-100 px-4 py-3 dark:border-gray-700">
          <p className="mb-2 text-xs font-medium text-gray-500 dark:text-gray-400">
            Attestations
          </p>
          <ul className="space-y-1.5">
            {attestations.map((a) => {
              const ai = ATTESTATION_ICON[a.status];
              return (
                <li key={a.name} className="flex items-center gap-2 text-xs">
                  <span className={`font-bold ${ai.cls}`}>{ai.icon}</span>
                  <span className="flex-1 text-gray-700 dark:text-gray-300">
                    {a.name}
                  </span>
                  <span className="text-gray-400">{a.date}</span>
                </li>
              );
            })}
          </ul>
        </div>
      )}

      {/* Violations */}
      {violations.length > 0 && (
        <div className="px-4 py-3">
          <p className="mb-2 text-xs font-medium text-gray-500 dark:text-gray-400">
            Violations ({violations.length})
          </p>
          <ul className="space-y-2">
            {violations.map((v, i) => (
              <li
                key={`${v.rule}-${v.severity}-${i}`}
                className="rounded-lg border border-gray-100 p-2 dark:border-gray-700"
              >
                <div className="flex items-center gap-2">
                  <span
                    className={`inline-flex rounded-full px-2 py-0.5 text-[10px] font-bold uppercase ${SEVERITY_BADGE[v.severity]}`}
                  >
                    {v.severity}
                  </span>
                  <span className="text-xs font-semibold text-gray-900 dark:text-gray-100">
                    {v.rule}
                  </span>
                </div>
                <p className="mt-1 text-xs text-gray-600 dark:text-gray-400">
                  {v.description}
                </p>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Empty state */}
      {violations.length === 0 && attestations.length === 0 && (
        <div className="px-4 py-6 text-center text-xs text-gray-400">
          No attestations or violations recorded.
        </div>
      )}
    </div>
  );
};

export default ComplianceCard;
