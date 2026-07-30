import * as vscode from "vscode";
import { LayerLensClient } from "./client";
import { TracesProvider, TraceItem } from "./tracesProvider";
import { TraceDocument } from "./traceDocument";
import { registerLocalCommands } from "./localCommands";
import { createStatusBar } from "./statusBar";

export function activate(context: vscode.ExtensionContext): void {
  const output = vscode.window.createOutputChannel("LayerLens");
  const client = new LayerLensClient(output);

  const tracesProvider = new TracesProvider(client);
  const traceView = vscode.window.createTreeView("layerlens.traces", {
    treeDataProvider: tracesProvider,
    showCollapseAll: false,
  });
  context.subscriptions.push(traceView);

  const statusBar = createStatusBar(client);
  context.subscriptions.push(statusBar);

  const reg = (cmd: string, fn: (...args: any[]) => any) =>
    context.subscriptions.push(vscode.commands.registerCommand(cmd, fn));

  reg("layerlens.connect", async () => {
    const apiKey = await vscode.window.showInputBox({
      prompt: "LayerLens API key",
      password: true,
      ignoreFocusOut: true,
    });
    if (!apiKey) return;
    await vscode.workspace
      .getConfiguration("layerlens")
      .update("apiKey", apiKey, vscode.ConfigurationTarget.Global);
    const orgId = await vscode.window.showInputBox({
      prompt: "Organization ID (optional)",
      value: vscode.workspace.getConfiguration("layerlens").get<string>("organizationId") ?? "",
      ignoreFocusOut: true,
    });
    if (orgId !== undefined) {
      await vscode.workspace
        .getConfiguration("layerlens")
        .update("organizationId", orgId, vscode.ConfigurationTarget.Global);
    }
    statusBar.update();
    tracesProvider.refresh();
    await vscode.window.showInformationMessage("LayerLens connected.");
  });

  reg("layerlens.disconnect", async () => {
    await vscode.workspace
      .getConfiguration("layerlens")
      .update("apiKey", "", vscode.ConfigurationTarget.Global);
    statusBar.update();
    tracesProvider.refresh();
  });

  reg("layerlens.refreshTraces", () => tracesProvider.refresh());

  reg("layerlens.viewTrace", async (target?: string | TraceItem) => {
    let traceId: string | undefined;
    if (typeof target === "string") traceId = target;
    else if (target instanceof TraceItem) traceId = target.trace.id;
    else traceId = await vscode.window.showInputBox({ prompt: "Trace ID to view" });
    if (!traceId) return;
    const trace = await client.getTrace(traceId);
    if (!trace) {
      await vscode.window.showErrorMessage(`Trace ${traceId} not found.`);
      return;
    }
    await TraceDocument.show(trace);
  });

  reg("layerlens.openDashboard", async () => {
    await vscode.env.openExternal(vscode.Uri.parse(client.dashboardUrl()));
  });

  registerLocalCommands(context, output);
}

export function deactivate(): void {
  /* noop */
}
