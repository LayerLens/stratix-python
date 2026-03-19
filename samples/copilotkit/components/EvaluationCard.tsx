/**
 * EvaluationCard — React component for displaying LayerLens evaluation results.
 *
 * Self-contained component designed for use with CopilotKit's render functions.
 * No build system required — can be used directly with CopilotKit's
 * useCopilotAction render callbacks.
 *
 * Usage:
 *   <EvaluationCard evaluation={evalData} onRerun={() => ...} />
 */

import React from "react";

interface EvaluationResult {
  prompt: string;
  score: number;
  duration_ms: number;
  passed: boolean;
}

interface EvaluationData {
  id: string;
  status: "pending" | "running" | "completed" | "failed";
  judge_name: string;
  dataset_name: string;
  created_at: string;
  pass_rate: number;
  average_score: number;
  total_samples: number;
  results?: EvaluationResult[];
}

interface EvaluationCardProps {
  evaluation: EvaluationData;
  onRerun?: () => void;
  onViewDetails?: (evalId: string) => void;
}

const statusColors: Record<string, string> = {
  completed: "#22c55e",
  failed: "#ef4444",
  running: "#f59e0b",
  pending: "#94a3b8",
};

export const EvaluationCard: React.FC<EvaluationCardProps> = ({
  evaluation,
  onRerun,
  onViewDetails,
}) => {
  const { id, status, judge_name, dataset_name, pass_rate, average_score, total_samples, results } =
    evaluation;
  const statusColor = statusColors[status] || "#94a3b8";
  const passRatePct = (pass_rate * 100).toFixed(1);
  const avgScorePct = (average_score * 100).toFixed(1);
  const passing = pass_rate >= 0.7;

  return (
    <div style={{ border: "1px solid #e2e8f0", borderRadius: 8, padding: 16, maxWidth: 480, fontFamily: "system-ui" }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
        <div>
          <h3 style={{ margin: 0, fontSize: 16 }}>Evaluation {id.slice(0, 8)}</h3>
          <span style={{ fontSize: 12, color: "#64748b" }}>{judge_name} / {dataset_name}</span>
        </div>
        <span style={{ background: statusColor, color: "#fff", borderRadius: 12, padding: "2px 10px", fontSize: 12, fontWeight: 600 }}>
          {status.toUpperCase()}
        </span>
      </div>

      {/* Metrics */}
      {status === "completed" && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8, marginBottom: 12 }}>
          <MetricBox label="Pass Rate" value={`${passRatePct}%`} highlight={passing} />
          <MetricBox label="Avg Score" value={`${avgScorePct}%`} />
          <MetricBox label="Samples" value={String(total_samples)} />
        </div>
      )}

      {/* Top results preview */}
      {results && results.length > 0 && (
        <div style={{ fontSize: 13, marginBottom: 12 }}>
          <strong>Top results:</strong>
          <table style={{ width: "100%", borderCollapse: "collapse", marginTop: 4 }}>
            <thead>
              <tr style={{ borderBottom: "1px solid #e2e8f0", textAlign: "left" }}>
                <th style={{ padding: "4px 0" }}>Prompt</th>
                <th style={{ padding: "4px 0", textAlign: "right" }}>Score</th>
                <th style={{ padding: "4px 0", textAlign: "center" }}>Pass</th>
              </tr>
            </thead>
            <tbody>
              {results.slice(0, 5).map((r, i) => (
                <tr key={i} style={{ borderBottom: "1px solid #f1f5f9" }}>
                  <td style={{ padding: "4px 0", maxWidth: 200, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {r.prompt}
                  </td>
                  <td style={{ padding: "4px 0", textAlign: "right" }}>{(r.score * 100).toFixed(0)}%</td>
                  <td style={{ padding: "4px 0", textAlign: "center" }}>{r.passed ? "Y" : "N"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Actions */}
      <div style={{ display: "flex", gap: 8 }}>
        {onRerun && (
          <button onClick={onRerun} style={{ padding: "6px 12px", borderRadius: 4, border: "1px solid #cbd5e1", background: "#fff", cursor: "pointer", fontSize: 13 }}>
            Re-run
          </button>
        )}
        {onViewDetails && (
          <button onClick={() => onViewDetails(id)} style={{ padding: "6px 12px", borderRadius: 4, border: "none", background: "#3b82f6", color: "#fff", cursor: "pointer", fontSize: 13 }}>
            View Details
          </button>
        )}
      </div>
    </div>
  );
};

const MetricBox: React.FC<{ label: string; value: string; highlight?: boolean }> = ({ label, value, highlight }) => (
  <div style={{ textAlign: "center", padding: 8, background: highlight ? "#f0fdf4" : "#f8fafc", borderRadius: 6 }}>
    <div style={{ fontSize: 18, fontWeight: 700, color: highlight ? "#16a34a" : "#1e293b" }}>{value}</div>
    <div style={{ fontSize: 11, color: "#64748b" }}>{label}</div>
  </div>
);

export default EvaluationCard;
