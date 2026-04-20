import * as vscode from "vscode";
import { LayerLensClient, TraceSummary } from "./client";

export class TracesProvider implements vscode.TreeDataProvider<TraceItem> {
  private _onDidChangeTreeData = new vscode.EventEmitter<TraceItem | undefined>();
  readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

  constructor(private client: LayerLensClient) {}

  refresh(): void {
    this._onDidChangeTreeData.fire(undefined);
  }

  getTreeItem(element: TraceItem): vscode.TreeItem {
    return element;
  }

  async getChildren(): Promise<TraceItem[]> {
    const traces = await this.client.listTraces();
    return traces.map((t) => new TraceItem(t));
  }
}

export class TraceItem extends vscode.TreeItem {
  constructor(public readonly trace: TraceSummary) {
    super(trace.name || trace.id, vscode.TreeItemCollapsibleState.None);
    this.id = trace.id;
    this.description = new Date(trace.createdAt).toLocaleString();
    this.tooltip = `${trace.id}\n${trace.status ?? ""}`;
    this.command = {
      command: "layerlens.viewTrace",
      title: "View Trace",
      arguments: [trace.id],
    };
  }
}
