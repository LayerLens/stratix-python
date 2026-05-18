/**
 * CopilotKit component library for LayerLens Stratix.
 *
 * Re-exports all card components that can be rendered inline
 * inside CopilotKit chat messages or in a side canvas via
 * ``useCoAgent``.
 *
 * Cards compose shadcn/ui primitives shipped under ``./ui/*``.
 * Consumers can import the cards alone, or pull the underlying
 * primitives directly from ``@layerlens/copilotkit-cards/ui/*``.
 */

export { EvaluationCard } from "./EvaluationCard";
export type {
  EvaluationCardProps,
  EvaluationStatus,
  Score,
  TrendPoint,
} from "./EvaluationCard";

export { TraceCard } from "./TraceCard";
export type { TraceCardProps, TraceStatus } from "./TraceCard";

export { JudgeVerdictCard } from "./JudgeVerdictCard";
export type {
  JudgeVerdictCardProps,
  Verdict,
  Severity,
  Evidence,
} from "./JudgeVerdictCard";

export { ComplianceCard } from "./ComplianceCard";
export type {
  ComplianceCardProps,
  ComplianceStatus,
  Attestation,
  Violation,
} from "./ComplianceCard";

export { MetricCard } from "./MetricCard";
export type {
  MetricCardProps,
  TrendDirection,
} from "./MetricCard";

export { MarkdownLite } from "./markdown-lite";
export type { MarkdownLiteProps } from "./markdown-lite";
