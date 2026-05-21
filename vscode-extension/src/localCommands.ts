import * as vscode from "vscode";
import { spawn } from "child_process";
import { TraceItem } from "./tracesProvider";

/**
 * Commands that shell out to the local Python SDK (`layerlens.cli`). Keeping
 * these separate from remote/API commands so the extension remains useful
 * even when the user is offline or the dashboard is unreachable.
 */
export function registerLocalCommands(
  context: vscode.ExtensionContext,
  output: vscode.OutputChannel,
): void {
  const reg = (cmd: string, fn: (...args: any[]) => any) =>
    context.subscriptions.push(vscode.commands.registerCommand(cmd, fn));

  reg("layerlens.runEvaluation", async () => {
    const datasetId = await vscode.window.showInputBox({
      prompt: "Dataset ID to evaluate against",
      ignoreFocusOut: true,
      validateInput: validateIdentifier,
    });
    if (!datasetId) return;
    const targetModule = await vscode.window.showInputBox({
      prompt: "Python module path of the target function (e.g. myapp.eval:predict)",
      ignoreFocusOut: true,
      validateInput: validateModuleSpec,
    });
    if (!targetModule) return;
    await runLayerLensCli(
      ["evaluations", "run", "--dataset-id", datasetId, "--target", targetModule],
      output,
    );
  });

  reg("layerlens.replayTrace", async (item?: TraceItem) => {
    const traceId =
      item?.trace?.id ??
      (await vscode.window.showInputBox({
        prompt: "Trace ID to replay",
        ignoreFocusOut: true,
        validateInput: validateIdentifier,
      }));
    if (!traceId) return;
    const modelOverride = await vscode.window.showInputBox({
      prompt: "Model override (leave blank for exact replay)",
      ignoreFocusOut: true,
      validateInput: (v: string) => (v === "" ? undefined : validateIdentifier(v)),
    });
    const args = ["replay", "run", "--trace-id", traceId];
    if (modelOverride) args.push("--model-override", modelOverride);
    await runLayerLensCli(args, output);
  });

  reg("layerlens.generateSynthetic", async () => {
    const templateId = await vscode.window.showQuickPick(
      [
        "llm.chat.basic",
        "agent.tool_calling",
        "rag.retrieval",
        "multi_agent.handoff",
      ],
      { placeHolder: "Synthetic template" },
    );
    if (!templateId) return;
    const count = await vscode.window.showInputBox({
      prompt: "How many traces to generate",
      value: "10",
      validateInput: (v: string) =>
        /^\d+$/.test(v) && Number(v) > 0 ? undefined : "Enter a positive integer",
      ignoreFocusOut: true,
    });
    if (!count) return;
    await runLayerLensCli(
      ["synthetic", "generate", "--template", templateId, "--count", count],
      output,
    );
  });
}

/**
 * Identifiers (dataset/trace IDs, model names) are constrained to characters
 * safe for CLI arguments. This is defence-in-depth — `spawn` without a shell
 * already avoids shell-metacharacter interpretation, but we still refuse input
 * that would confuse the CLI's argparse (leading dashes, NUL bytes, etc.).
 */
function validateIdentifier(v: string): string | undefined {
  if (!/^[A-Za-z0-9][A-Za-z0-9._:@/-]{0,255}$/.test(v)) {
    return "Use letters, digits, and . _ : @ / - (must not start with '-').";
  }
  return undefined;
}

/** ``module.submodule:attr`` — dotted import path plus a single attribute. */
function validateModuleSpec(v: string): string | undefined {
  if (!/^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$/.test(v)) {
    return "Expected 'module.path:attr'.";
  }
  return undefined;
}

function runLayerLensCli(args: string[], output: vscode.OutputChannel): Promise<void> {
  const python =
    vscode.workspace.getConfiguration("layerlens").get<string>("pythonPath") || "python";
  const fullArgs = ["-m", "layerlens.cli", ...args];
  output.show(true);
  output.appendLine(`> ${python} ${fullArgs.join(" ")}`);

  return new Promise((resolve) => {
    const child = spawn(python, fullArgs, { shell: false });
    child.stdout.on("data", (chunk: Buffer) => output.append(chunk.toString()));
    child.stderr.on("data", (chunk: Buffer) => output.append(chunk.toString()));
    child.on("error", (err: Error) => {
      output.appendLine(`\n[layerlens] failed to spawn: ${err.message}`);
      resolve();
    });
    child.on("close", (code: number | null) => {
      output.appendLine(`\n[layerlens] exited with code ${code ?? "null"}`);
      resolve();
    });
  });
}
