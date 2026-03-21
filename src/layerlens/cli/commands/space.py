from __future__ import annotations

import sys

import click

from .._client import get_client, handle_errors
from .._formatter import format_output

SPACE_COLUMNS = [
    ("id", "ID"),
    ("name", "Name"),
    ("visibility", "Visibility"),
    ("models_count", "Models"),
    ("benchmarks_count", "Benchmarks"),
    ("evaluations_count", "Evaluations"),
    ("created_at", "Created"),
]


@click.group()
def space() -> None:
    """Manage evaluation spaces.

    \b
    Examples:
      stratix space list
      stratix space get <space-id>
      stratix space create --name "My Space"
      stratix space delete <space-id>
    """


@space.command("list")
@click.option("--page", default=None, type=int, help="Page number.")
@click.option("--page-size", default=None, type=int, help="Results per page.")
@click.option("--sort-by", default=None, help="Sort field (e.g. weight, created_at).")
@click.option("--order", default=None, type=click.Choice(["asc", "desc"]), help="Sort order.")
@click.pass_context
@handle_errors
def list_spaces(
    ctx: click.Context, page: int | None, page_size: int | None, sort_by: str | None, order: str | None
) -> None:
    """List evaluation spaces with optional pagination.

    \b
    Examples:
      stratix space list
      stratix space list --page-size 10
      stratix space list --sort-by created_at --order desc
    """
    client = get_client(ctx)
    result = client.evaluation_spaces.get_many(page=page, page_size=page_size, sort_by=sort_by, order=order)
    if result is None or not result.evaluation_spaces:
        click.echo("No evaluation spaces found.")
        return

    if ctx.obj["verbose"]:
        click.echo(f"Showing {result.count} of {result.total_count} evaluation spaces", err=True)

    output = format_output(result.evaluation_spaces, ctx.obj["output_format"], SPACE_COLUMNS)
    click.echo(output)


@space.command("get")
@click.argument("id")
@click.pass_context
@handle_errors
def get_space(ctx: click.Context, id: str) -> None:
    """Get an evaluation space by ID or slug.

    \b
    Examples:
      stratix space get abc123
      stratix space get my-space-slug
      stratix space get abc123 --format json
    """
    client = get_client(ctx)
    s = client.evaluation_spaces.get(id)
    if s is None:
        click.echo(f"Evaluation space {id} not found.", err=True)
        sys.exit(1)

    output = format_output(s, ctx.obj["output_format"])
    click.echo(output)


@space.command("create")
@click.option("--name", required=True, help="Space name.")
@click.option("--description", default=None, help="Space description (max 500 chars).")
@click.option(
    "--visibility", default=None, type=click.Choice(["private", "public", "tenant"]), help="Visibility level."
)
@click.option("--dry-run", is_flag=True, default=False, help="Preview without executing.")
@click.pass_context
@handle_errors
def create_space(ctx: click.Context, name: str, description: str | None, visibility: str | None, dry_run: bool) -> None:
    """Create a new evaluation space.

    \b
    Examples:
      stratix space create --name "Production"
      stratix space create --name "Public Board" --visibility public
      stratix space create --name "Test" --dry-run
    """
    if dry_run:
        click.echo(f"[dry-run] Would create evaluation space: {name}")
        if visibility:
            click.echo(f"  Visibility: {visibility}")
        return

    client = get_client(ctx)
    s = client.evaluation_spaces.create(name=name, description=description, visibility=visibility)
    if s is None:
        click.echo("Failed to create evaluation space.", err=True)
        sys.exit(1)

    click.echo(f"Evaluation space created: {s.id}")
    output = format_output(s, ctx.obj["output_format"])
    click.echo(output)


@space.command("delete")
@click.argument("id")
@click.option("--yes", "-y", is_flag=True, default=False, help="Skip confirmation prompt.")
@click.option("--dry-run", is_flag=True, default=False, help="Preview without executing.")
@click.pass_context
@handle_errors
def delete_space(ctx: click.Context, id: str, yes: bool, dry_run: bool) -> None:
    """Delete an evaluation space by ID.

    \b
    Examples:
      stratix space delete abc123
      stratix space delete abc123 --yes
      stratix space delete abc123 --dry-run
    """
    if dry_run:
        click.echo(f"[dry-run] Would delete evaluation space {id}")
        return

    if not yes:
        click.confirm(f"Are you sure you want to delete evaluation space {id}?", abort=True)

    client = get_client(ctx)
    success = client.evaluation_spaces.delete(id)
    if success:
        click.echo(f"Evaluation space {id} deleted.")
    else:
        click.echo(f"Failed to delete evaluation space {id}.", err=True)
        sys.exit(1)
