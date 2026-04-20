import * as vscode from "vscode";

export interface TraceSummary {
  id: string;
  name: string;
  createdAt: string;
  status?: string;
}

export interface TraceDetail extends TraceSummary {
  events: Array<Record<string, unknown>>;
}

/**
 * Minimal LayerLens API client used by the extension. Keeps the fetch surface
 * small so the extension can be extended incrementally without touching the UI.
 */
export class LayerLensClient {
  constructor(private output: vscode.OutputChannel) {}

  private config(): vscode.WorkspaceConfiguration {
    return vscode.workspace.getConfiguration("layerlens");
  }

  apiKey(): string | undefined {
    const cfg = this.config();
    return (cfg.get<string>("apiKey") || process.env.LAYERLENS_API_KEY) ?? undefined;
  }

  baseUrl(): string {
    return this.config().get<string>("apiBaseUrl") || "https://api.layerlens.ai";
  }

  orgId(): string | undefined {
    return this.config().get<string>("organizationId") || undefined;
  }

  projectId(): string | undefined {
    return this.config().get<string>("projectId") || undefined;
  }

  isConnected(): boolean {
    return Boolean(this.apiKey());
  }

  dashboardUrl(): string {
    const base = this.baseUrl().replace(/\/api\/?$/, "");
    return this.orgId() ? `${base}/enterprise/${this.orgId()}` : base;
  }

  async listTraces(): Promise<TraceSummary[]> {
    const apiKey = this.apiKey();
    if (!apiKey) return [];
    const url = `${this.baseUrl()}/v1/organizations/${this.orgId()}/traces`;
    try {
      const res = await fetch(url, { headers: { Authorization: `Bearer ${apiKey}` } });
      if (!res.ok) {
        this.output.appendLine(`listTraces ${res.status}: ${await res.text()}`);
        return [];
      }
      const json = (await res.json()) as { traces?: TraceSummary[] };
      return json.traces ?? [];
    } catch (err) {
      this.output.appendLine(`listTraces error: ${String(err)}`);
      return [];
    }
  }

  async getTrace(traceId: string): Promise<TraceDetail | undefined> {
    const apiKey = this.apiKey();
    if (!apiKey) return undefined;
    const url = `${this.baseUrl()}/v1/organizations/${this.orgId()}/traces?id=${encodeURIComponent(traceId)}`;
    try {
      const res = await fetch(url, { headers: { Authorization: `Bearer ${apiKey}` } });
      if (!res.ok) {
        this.output.appendLine(`getTrace ${res.status}: ${await res.text()}`);
        return undefined;
      }
      return (await res.json()) as TraceDetail;
    } catch (err) {
      this.output.appendLine(`getTrace error: ${String(err)}`);
      return undefined;
    }
  }
}
