# MCP Server

The Model Context Protocol (MCP) server exposes LayerLens capabilities as tools that can be
consumed by Claude, Cursor, VS Code Copilot, and any other MCP-compatible client. This enables
AI assistants to directly query traces, run evaluations, and manage judges through natural
language -- turning LayerLens into an interactive quality assurance co-pilot within your
development environment.

## Prerequisites

```bash
pip install layerlens --index-url https://sdk.layerlens.ai/package mcp
export LAYERLENS_STRATIX_API_KEY=your-api-key
```

## Quick Start

Run the server in stdio mode for use with Claude Code or other MCP clients:

```bash
python layerlens_server.py
```

The server will start and wait for MCP protocol messages on stdin/stdout.

## Available Tools

| Tool | Description |
|------|-------------|
| `list_traces` | List recent traces with optional filters (date range, status, model). |
| `get_trace` | Retrieve a single trace by ID, including all spans and metadata. |
| `run_evaluation` | Run a trace evaluation using a specified judge. |
| `get_evaluation` | Fetch evaluation results by evaluation ID. |
| `create_judge` | Create a new AI judge with custom criteria. |
| `list_judges` | List all available judges in the workspace. |

## Configuration

### Claude Code

Add the following to your MCP configuration file:

```json
{
  "mcpServers": {
    "layerlens": {
      "command": "python",
      "args": ["samples/mcp/layerlens_server.py"],
      "env": {
        "LAYERLENS_STRATIX_API_KEY": "your-api-key"
      }
    }
  }
}
```

### Other MCP Clients

Any MCP-compatible client can connect to this server using stdio transport. Consult your
client's documentation for the configuration format, and point it to `layerlens_server.py`
with the required environment variable.

## Expected Behavior

Once connected, the MCP client will discover the available tools and make them accessible
through its interface. For example, in Claude Code you can ask "list my recent traces" or
"evaluate trace abc-123 with the safety judge" and the assistant will invoke the
corresponding LayerLens tool.
