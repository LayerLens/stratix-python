# CLI — Getting Started

The LayerLens Stratix CLI provides terminal access to all platform features: traces, judges, evaluations, integrations, scorers, evaluation spaces, bulk operations, and CI/CD helpers.

## Installation

Install the SDK with the `cli` extra:

```bash
pip install layerlens[cli] --extra-index-url https://sdk.layerlens.ai/package
```

If you already have `layerlens` installed, add the CLI extra:

```bash
pip install "layerlens[cli]" --extra-index-url https://sdk.layerlens.ai/package
```

For local development from a cloned repo:

```bash
pip install -e ".[cli]"
```

Verify the installation:

```bash
stratix --version
```

## Configuration

### API key

The CLI requires a LayerLens Stratix API key. Set it as an environment variable (recommended):

```bash
export LAYERLENS_STRATIX_API_KEY="your-api-key"
```

Or pass it per-command:

```bash
stratix --api-key "your-api-key" trace list
```

### Custom host

By default the CLI talks to `api.layerlens.ai`. Override with:

```bash
stratix --host my-instance.example.com trace list
stratix --host my-instance.example.com --port 8443 trace list
```

## Global options

Every command accepts these options:

| Option | Description |
| --- | --- |
| `--api-key` | API key (or set `LAYERLENS_STRATIX_API_KEY`) |
| `--host` | API host |
| `--port` | API port |
| `--format` | Output format: `table` (default) or `json` |
| `--verbose` / `-v` | Enable debug output |
| `--version` | Print version and exit |

## Output formats

The default output is a human-readable table:

```bash
stratix trace list
```

```
ID                                   Created              Filename         Evaluations
───────────────────────────────────────────────────────────────────────────────────────
a1b2c3d4-...                         2026-03-15 14:30     traces.jsonl     3
e5f6a7b8-...                         2026-03-14 09:12     batch_02.json    1
```

Switch to JSON for scripting:

```bash
stratix --format json trace list
```

```json
[
  {
    "id": "a1b2c3d4-...",
    "created_at": "2026-03-15T14:30:00Z",
    "filename": "traces.jsonl",
    ...
  }
]
```

## Shell completions

The CLI supports tab-completion for commands, options, and resource IDs.

```bash
# Print setup instructions for your shell
stratix completion bash
stratix completion zsh
stratix completion fish
stratix completion powershell
```

Follow the printed instructions to enable completions. After setup, you can tab-complete trace IDs, judge IDs, model names, and more.

## First commands

### List your traces

```bash
stratix trace list
```

### Run an evaluation

```bash
stratix evaluate run --model openai/gpt-4o --benchmark arc-agi-2 --wait
```

### Create a judge

```bash
stratix judge create --name "Response Quality" --goal "Rate accuracy and completeness" --model-id <MODEL_ID>
```

### Check integrations

```bash
stratix integration list
```

### Generate a CI report

```bash
stratix ci report -o summary.md
```

## Next steps

- [Command Reference](commands.md) — all commands and their options
- [Examples](examples.md) — 15 common workflows as copy-paste shell sessions
