from __future__ import annotations

import sys

import click

from .._client import get_client, handle_errors
from .._formatter import format_output
from .._completions import complete_integration

INTEGRATION_COLUMNS = [
    ("id", "ID"),
    ("name", "Name"),
    ("type", "Type"),
    ("status", "Status"),
    ("created_at", "Created"),
]


@click.group()
def integration() -> None:
    """Manage integrations.

    \b
    Examples:
      layerlens integration list
      layerlens integration test <integration-id>
    """


@integration.command("list")
@click.option("--page", default=None, type=int, help="Page number.")
@click.option("--page-size", default=None, type=int, help="Results per page.")
@click.pass_context
@handle_errors
def list_integrations(ctx: click.Context, page: int | None, page_size: int | None) -> None:
    """List integrations with optional pagination.

    \b
    Examples:
      layerlens integration list
      layerlens integration list --page-size 10
    """
    client = get_client(ctx)
    result = client.integrations.get_many(page=page, page_size=page_size)
    if result is None or not result.integrations:
        click.echo("No integrations found.")
        return

    if ctx.obj["verbose"]:
        click.echo(f"Showing {result.count} of {result.total_count} integrations", err=True)

    output = format_output(result.integrations, ctx.obj["output_format"], INTEGRATION_COLUMNS)
    click.echo(output)


@integration.command("test")
@click.argument("id", shell_complete=complete_integration)
@click.pass_context
@handle_errors
def test_integration(ctx: click.Context, id: str) -> None:
    """Test an integration by ID.

    \b
    Examples:
      layerlens integration test abc123
      layerlens integration test abc123 --format json
    """
    client = get_client(ctx)
    result = client.integrations.test(id)
    if result is None:
        click.echo(f"Failed to test integration {id}.", err=True)
        sys.exit(1)

    if result.success:
        click.echo(f"Integration {id}: OK")
    else:
        click.echo(f"Integration {id}: FAILED")

    if result.message:
        click.echo(f"Message: {result.message}")

    if ctx.obj["output_format"] == "json":
        output = format_output(result, "json")
        click.echo(output)
