/**
 * useLayerLensContext — Exposes dashboard state to CopilotKit as readable context.
 *
 * The parent component passes current dashboard state into this hook, and
 * CopilotKit makes it available to the AI assistant so it can give contextual
 * answers without the user having to describe what they are looking at.
 */

import { useCopilotReadable } from "@copilotkit/react-core";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface AppliedFilter {
  field: string;
  operator: "eq" | "neq" | "gt" | "lt" | "gte" | "lte" | "contains" | "in";
  value: string | number | string[];
}

export interface EvaluationResultSummary {
  evaluationId: string;
  name: string;
  passRate: number;
  totalCases: number;
  status: "running" | "completed" | "failed";
}

export interface RecentTrace {
  traceId: string;
  agentName: string;
  framework: string;
  status: "ok" | "error" | "timeout" | "running";
  timestamp: string;
}

export interface UserProfile {
  /** User's persona (e.g. "ml-engineer", "compliance-officer", "product-manager") */
  persona: string;
  /** Organization slug or name */
  org: string;
  /** Display name */
  displayName?: string;
}

export interface LayerLensContextParams {
  /** Current dashboard page/route (e.g. "/traces", "/evaluations/abc123") */
  currentPage?: string;
  /** Currently selected trace ID, if any */
  selectedTraceId?: string | null;
  /** Active filters applied in the current view */
  appliedFilters?: AppliedFilter[];
  /** Recent evaluation results visible in the dashboard */
  evaluationResults?: EvaluationResultSummary[];
  /** Recent traces visible in the dashboard */
  recentTraces?: RecentTrace[];
  /** Current user profile */
  userProfile?: UserProfile;
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

/**
 * Registers LayerLens dashboard state as CopilotKit readable context.
 *
 * Call this hook at the top level of your CopilotKit-wrapped dashboard page
 * and pass in the current UI state.  The AI assistant will automatically
 * have access to this context when formulating responses.
 *
 * @example
 * ```tsx
 * function DashboardPage() {
 *   useLayerLensContext({
 *     currentPage: "/traces",
 *     selectedTraceId: selectedId,
 *     appliedFilters: filters,
 *     recentTraces: traces,
 *     userProfile: { persona: "ml-engineer", org: "acme-corp" },
 *   });
 *   return <Dashboard />;
 * }
 * ```
 */
export function useLayerLensContext(params: LayerLensContextParams) {
  const {
    currentPage,
    selectedTraceId,
    appliedFilters,
    evaluationResults,
    recentTraces,
    userProfile,
  } = params;

  // -- Current navigation state --
  useCopilotReadable({
    description:
      "The current page/route the user is viewing in the LayerLens Stratix dashboard.",
    value: currentPage ?? "unknown",
  });

  // -- Selected trace --
  useCopilotReadable({
    description:
      "The trace ID currently selected or highlighted by the user, if any. " +
      "Null means no trace is selected.",
    value: selectedTraceId ?? null,
  });

  // -- Active filters --
  useCopilotReadable({
    description:
      "Filters the user has applied in the current dashboard view. " +
      "Each filter has a field, operator, and value.",
    value: appliedFilters ?? [],
  });

  // -- Evaluation results --
  useCopilotReadable({
    description:
      "Summary of recent evaluation results visible in the dashboard. " +
      "Includes evaluation ID, name, pass rate, total cases, and status.",
    value: evaluationResults ?? [],
  });

  // -- Recent traces --
  useCopilotReadable({
    description:
      "Recent traces visible in the dashboard list. " +
      "Each entry includes trace ID, agent name, framework, status, and timestamp.",
    value: recentTraces ?? [],
  });

  // -- User profile --
  useCopilotReadable({
    description:
      "The current user's profile including their persona " +
      "(e.g. ml-engineer, compliance-officer, product-manager) " +
      "and organization. Use this to tailor responses to the user's role.",
    value: userProfile ?? null,
  });
}

export default useLayerLensContext;
