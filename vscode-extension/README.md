# LayerLens — VS Code Extension

Trace viewer, debugger, and SDK integration for [LayerLens](https://layerlens.ai) inside VS Code.

## Features

- **Activity bar view** — LayerLens traces sidebar with refresh action.
- **Trace viewer** — open any trace by ID or pick one from the sidebar; webview renders events as expandable blocks.
- **Connect / disconnect** — stores API key and organization ID in VS Code user settings.
- **Status bar** — always-on connection indicator; click to open the dashboard.
- **Local workflows** — commands shell out to `python -m layerlens.cli` so replays, synthetic generation, and dataset-scoped evaluation runs work offline:
  - `LayerLens: Replay Trace Locally`
  - `LayerLens: Generate Synthetic Traces`
  - `LayerLens: Run Evaluation on Dataset`
- **Open Dashboard** — one-click jump to the LayerLens UI for the configured org.

## Install (dev)

```bash
cd vscode-extension
npm install
npm run compile
code --install-extension ./layerlens-vscode-0.1.0.vsix
```

## Configuration

| Setting | Default | Description |
|---|---|---|
| `layerlens.apiBaseUrl` | `https://api.layerlens.ai` | LayerLens API base URL. |
| `layerlens.apiKey` | _(empty)_ | API key. Prefer `LAYERLENS_API_KEY` env var. |
| `layerlens.organizationId` | _(empty)_ | Default tenant ID used for trace/evaluation endpoints. |
| `layerlens.projectId` | _(empty)_ | Default project ID. |
| `layerlens.pythonPath` | `python` | Interpreter used for local `layerlens.cli` invocations. |

## Structure

- `src/extension.ts` — activation + command registrations.
- `src/client.ts` — thin LayerLens REST client.
- `src/tracesProvider.ts` — explorer sidebar tree.
- `src/traceDocument.ts` — webview renderer for a single trace.
- `src/localCommands.ts` — replay / synthetic / evaluation commands that invoke the Python CLI.
- `src/statusBar.ts` — connection indicator in the status bar.
