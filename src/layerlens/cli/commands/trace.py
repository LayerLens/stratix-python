from __future__ import annotations

import sys
import json

import click

from .._client import get_client, handle_errors
from .._formatter import to_dict, format_output
from .._completions import complete_trace

TRACE_COLUMNS = [
    ("id", "ID"),
    ("created_at", "Created"),
    ("filename", "Filename"),
    ("evaluations_count", "Evaluations"),
]


@click.group()
def trace() -> None:
    """Manage traces.

    \b
    Examples:
      stratix trace list
      stratix trace get <trace-id>
      stratix trace search "user login"
      stratix trace export <trace-id> --output trace.json
      stratix trace delete <trace-id> --yes
    """


@trace.command("list")
@click.option("--page", default=None, type=int, help="Page number.")
@click.option("--page-size", default=None, type=int, help="Results per page.")
@click.option("--source", default=None, help="Filter by source.")
@click.option("--status", default=None, help="Filter by status.")
@click.option("--sort-by", default=None, help="Sort field.")
@click.option("--sort-order", default=None, type=click.Choice(["asc", "desc"]), help="Sort order.")
@click.pass_context
@handle_errors
def list_traces(
    ctx: click.Context,
    page: int | None,
    page_size: int | None,
    source: str | None,
    status: str | None,
    sort_by: str | None,
    sort_order: str | None,
) -> None:
    """List traces with optional filtering and pagination.

    \b
    Examples:
      stratix trace list
      stratix trace list --page-size 10
      stratix trace list --source sdk --sort-by created_at --sort-order desc
    """
    client = get_client(ctx)
    result = client.traces.get_many(
        page=page,
        page_size=page_size,
        source=source,
        status=status,
        sort_by=sort_by,
        sort_order=sort_order,
    )
    if result is None or not result.traces:
        click.echo("No traces found.")
        return

    if ctx.obj["verbose"]:
        click.echo(f"Showing {result.count} of {result.total_count} traces", err=True)

    output = format_output(result.traces, ctx.obj["output_format"], TRACE_COLUMNS)
    click.echo(output)


@trace.command("get")
@click.argument("id", shell_complete=complete_trace)
@click.pass_context
@handle_errors
def get_trace(ctx: click.Context, id: str) -> None:
    """Get a trace by ID.

    \b
    Examples:
      stratix trace get abc123-def4-5678-ghij-klmnopqrstuv
      stratix trace get abc123 --format json
    """
    client = get_client(ctx)
    trace = client.traces.get(id)
    if trace is None:
        click.echo(f"Trace {id} not found.", err=True)
        sys.exit(1)

    output = format_output(trace, ctx.obj["output_format"])
    click.echo(output)


@trace.command("search")
@click.argument("query")
@click.option("--page", default=None, type=int, help="Page number.")
@click.option("--page-size", default=None, type=int, help="Results per page.")
@click.option("--source", default=None, help="Filter by source.")
@click.option("--status", default=None, help="Filter by status.")
@click.option("--sort-by", default=None, help="Sort field.")
@click.option("--sort-order", default=None, type=click.Choice(["asc", "desc"]), help="Sort order.")
@click.pass_context
@handle_errors
def search_traces(
    ctx: click.Context,
    query: str,
    page: int | None,
    page_size: int | None,
    source: str | None,
    status: str | None,
    sort_by: str | None,
    sort_order: str | None,
) -> None:
    """Search traces by query string.

    \b
    Examples:
      stratix trace search "user login"
      stratix trace search "error" --source sdk --page-size 5
    """
    client = get_client(ctx)
    result = client.traces.get_many(
        search=query,
        page=page,
        page_size=page_size,
        source=source,
        status=status,
        sort_by=sort_by,
        sort_order=sort_order,
    )
    if result is None or not result.traces:
        click.echo("No traces found matching your query.")
        return

    if ctx.obj["verbose"]:
        click.echo(f"Found {result.count} of {result.total_count} traces", err=True)

    output = format_output(result.traces, ctx.obj["output_format"], TRACE_COLUMNS)
    click.echo(output)


@trace.command("export")
@click.argument("id", shell_complete=complete_trace)
@click.option(
    "--output", "-o", "output_file", default=None, type=click.Path(), help="Output file path (default: stdout)."
)
@click.pass_context
@handle_errors
def export_trace(ctx: click.Context, id: str, output_file: str | None) -> None:
    """Export a trace as JSON.

    \b
    Examples:
      stratix trace export abc123
      stratix trace export abc123 --output trace.json
    """
    client = get_client(ctx)
    trace = client.traces.get(id)
    if trace is None:
        click.echo(f"Trace {id} not found.", err=True)
        sys.exit(1)

    json_str = json.dumps(to_dict(trace), indent=2, default=str)

    if output_file:
        with open(output_file, "w") as f:
            f.write(json_str)
        click.echo(f"Trace exported to {output_file}")
    else:
        click.echo(json_str)


@trace.command("delete")
@click.argument("id", shell_complete=complete_trace)
@click.option("--yes", "-y", is_flag=True, default=False, help="Skip confirmation prompt.")
@click.pass_context
@handle_errors
def delete_trace(ctx: click.Context, id: str, yes: bool) -> None:
    """Delete a trace by ID.

    \b
    Examples:
      stratix trace delete abc123
      stratix trace delete abc123 --yes
    """
    if not yes:
        click.confirm(f"Are you sure you want to delete trace {id}?", abort=True)

    client = get_client(ctx)
    success = client.traces.delete(id)
    if success:
        click.echo(f"Trace {id} deleted.")
    else:
        click.echo(f"Failed to delete trace {id}.", err=True)
        sys.exit(1)
