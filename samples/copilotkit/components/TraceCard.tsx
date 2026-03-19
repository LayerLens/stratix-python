/**
 * TraceCard — React component for displaying a LayerLens trace summary.
 *
 * Self-contained component for use with CopilotKit render callbacks.
 * Shows trace metadata, event timeline, and quick actions.
 *
 * Usage:
 *   <TraceCard trace={traceData} onReplay={() => ...} onInvestigate={() => ...} />
 */

import React, { useState } from "react";

interface TraceEvent {
  type: string;
  timestamp: string;
  duration_ms?: number;
  status?: string;
  error?: string;
}

interface TraceData {
  id: string;
  agent_id: string;
  model: string;
  status: "success" | "error" | "timeout" | "running";
  created_at: string;
  total_duration_ms: number;
  total_tokens?: number;
  events: TraceEvent[];
  tags?: string[];
}

interface TraceCardProps {
  trace: TraceData;
  onReplay?: (traceId: string) => void;
  onInvestigate?: (traceId: string) => void;
  onTagAdd?: (traceId: string, tag: string) => void;
}

const statusIcons: Record<string, string> = {
  success: "OK",
  error: "ERR",
  timeout: "TMO",
  running: "...",
};

const eventTypeColors: Record<string, string> = {
  input: "#3b82f6",
  output: "#22c55e",
  model_invoke: "#a855f7",
  tool_call: "#f59e0b",
  error: "#ef4444",
};

export const TraceCard: React.FC<TraceCardProps> = ({ trace, onReplay, onInvestigate, onTagAdd }) => {
  const [expanded, setExpanded] = useState(false);
  const { id, agent_id, model, status, created_at, total_duration_ms, total_tokens, events, tags } = trace;

  const duration = total_duration_ms < 1000 ? `${total_duration_ms}ms` : `${(total_duration_ms / 1000).toFixed(1)}s`;
  const errorCount = events.filter((e) => e.status === "error" || e.error).length;

  return (
    <div style={{ border: "1px solid #e2e8f0", borderRadius: 8, padding: 16, maxWidth: 480, fontFamily: "system-ui" }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
        <div>
          <h3 style={{ margin: 0, fontSize: 15 }}>
            <span style={{ fontFamily: "monospace", fontSize: 13 }}>{id.slice(0, 12)}</span>
          </h3>
          <span style={{ fontSize: 12, color: "#64748b" }}>{agent_id} | {model}</span>
        </div>
        <span style={{ fontWeight: 700, fontSize: 12, color: status === "success" ? "#16a34a" : "#ef4444" }}>
          {statusIcons[status] || status}
        </span>
      </div>

      {/* Quick stats */}
      <div style={{ display: "flex", gap: 16, fontSize: 13, color: "#475569", marginBottom: 8 }}>
        <span>Duration: <strong>{duration}</strong></span>
        <span>Events: <strong>{events.length}</strong></span>
        {total_tokens && <span>Tokens: <strong>{total_tokens}</strong></span>}
        {errorCount > 0 && <span style={{ color: "#ef4444" }}>Errors: <strong>{errorCount}</strong></span>}
      </div>

      {/* Tags */}
      {tags && tags.length > 0 && (
        <div style={{ display: "flex", gap: 4, marginBottom: 8, flexWrap: "wrap" }}>
          {tags.map((tag) => (
            <span key={tag} style={{ background: "#f1f5f9", borderRadius: 4, padding: "1px 6px", fontSize: 11, color: "#475569" }}>
              {tag}
            </span>
          ))}
        </div>
      )}

      {/* Event timeline (collapsible) */}
      <button
        onClick={() => setExpanded(!expanded)}
        style={{ background: "none", border: "none", cursor: "pointer", fontSize: 13, color: "#3b82f6", padding: 0, marginBottom: 8 }}
      >
        {expanded ? "Hide" : "Show"} event timeline
      </button>

      {expanded && (
        <div style={{ borderLeft: "2px solid #e2e8f0", paddingLeft: 12, marginBottom: 12, fontSize: 12 }}>
          {events.map((ev, i) => (
            <div key={i} style={{ marginBottom: 6, display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{ width: 8, height: 8, borderRadius: "50%", background: eventTypeColors[ev.type] || "#94a3b8", flexShrink: 0 }} />
              <span style={{ fontFamily: "monospace", color: "#64748b", minWidth: 90 }}>{ev.type}</span>
              {ev.duration_ms != null && <span style={{ color: "#94a3b8" }}>{ev.duration_ms}ms</span>}
              {ev.error && <span style={{ color: "#ef4444", fontSize: 11 }}>{ev.error}</span>}
            </div>
          ))}
        </div>
      )}

      {/* Actions */}
      <div style={{ display: "flex", gap: 8 }}>
        {onReplay && (
          <button onClick={() => onReplay(id)} style={{ padding: "6px 12px", borderRadius: 4, border: "1px solid #cbd5e1", background: "#fff", cursor: "pointer", fontSize: 13 }}>
            Replay
          </button>
        )}
        {onInvestigate && errorCount > 0 && (
          <button onClick={() => onInvestigate(id)} style={{ padding: "6px 12px", borderRadius: 4, border: "none", background: "#ef4444", color: "#fff", cursor: "pointer", fontSize: 13 }}>
            Investigate
          </button>
        )}
      </div>

      <div style={{ fontSize: 11, color: "#94a3b8", marginTop: 8 }}>{created_at}</div>
    </div>
  );
};

export default TraceCard;
