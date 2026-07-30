import * as vscode from "vscode";
import { TraceDetail } from "./client";

/**
 * Opens a webview for a LayerLens trace. The rendered HTML is intentionally
 * minimal — events as a scrollable list with expandable payloads. Real-time
 * updates / follow-along debugging can be layered on top later.
 */
export const TraceDocument = {
  async show(trace: TraceDetail): Promise<void> {
    const panel = vscode.window.createWebviewPanel(
      "layerlens.trace",
      `LayerLens: ${trace.name || trace.id}`,
      vscode.ViewColumn.Beside,
      { enableScripts: false, retainContextWhenHidden: true },
    );
    panel.webview.html = renderTrace(trace);
  },
};

function renderTrace(trace: TraceDetail): string {
  const events = trace.events
    .map((ev) => {
      const body = escapeHtml(JSON.stringify(ev, null, 2));
      const kind = escapeHtml(String(ev["event_type"] ?? ev["type"] ?? "event"));
      return `<details><summary><b>${kind}</b></summary><pre>${body}</pre></details>`;
    })
    .join("\n");
  return `<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
  body { font-family: var(--vscode-editor-font-family, monospace); padding: 12px; }
  h1 { font-size: 14px; margin: 0 0 8px; }
  details { margin: 4px 0; border-left: 2px solid var(--vscode-editorWidget-border, #888); padding-left: 8px; }
  summary { cursor: pointer; padding: 2px 0; }
  pre { margin: 4px 0; font-size: 11px; white-space: pre-wrap; }
</style></head>
<body>
  <h1>${escapeHtml(trace.name || trace.id)} <small>${escapeHtml(trace.id)}</small></h1>
  <p>Status: ${escapeHtml(trace.status ?? "unknown")} · Created: ${escapeHtml(trace.createdAt)}</p>
  ${events}
</body></html>`;
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) =>
    c === "&" ? "&amp;" : c === "<" ? "&lt;" : c === ">" ? "&gt;" : c === "\"" ? "&quot;" : "&#39;",
  );
}
