import * as vscode from "vscode";
import { LayerLensClient } from "./client";

export interface LayerLensStatusBar {
  update(): void;
  dispose(): void;
}

export function createStatusBar(client: LayerLensClient): LayerLensStatusBar {
  const item = vscode.window.createStatusBarItem(
    vscode.StatusBarAlignment.Right,
    100,
  );
  item.command = "layerlens.openDashboard";

  const update = () => {
    if (client.isConnected()) {
      item.text = `$(graph-line) LayerLens`;
      item.tooltip = `LayerLens: ${client.baseUrl()} (${client.orgId() ?? "no org"})`;
    } else {
      item.text = `$(debug-disconnect) LayerLens`;
      item.tooltip = "LayerLens: not connected — run 'LayerLens: Connect to Project'.";
    }
    item.show();
  };

  update();
  const watcher = vscode.workspace.onDidChangeConfiguration(
    (e: vscode.ConfigurationChangeEvent) => {
      if (e.affectsConfiguration("layerlens")) update();
    },
  );

  return {
    update,
    dispose() {
      watcher.dispose();
      item.dispose();
    },
  };
}
