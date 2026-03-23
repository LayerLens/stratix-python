from __future__ import annotations

import sys

import click

from .._client import get_client, handle_errors
from .._formatter import format_output

SCORER_COLUMNS = [
    ("id", "ID"),
    ("name", "Name"),
    ("model_name", "Model"),
    ("model_company", "Company"),
    ("created_at", "Created"),
]


@click.group()
def scorer() -> None:
    """Manage scorers.

    \b
    Examples:
      stratix scorer list
      stratix scorer get <scorer-id>
      stratix scorer create --name "Quality" --description "..." --model-id <id> --prompt "..."
      stratix scorer delete <scorer-id>
    """


@scorer.command("list")
@click.option("--page", default=None, type=int, help="Page number.")
@click.option("--page-size", default=None, type=int, help="Results per page.")
@click.pass_context
@handle_errors
def list_scorers(ctx: click.Context, page: int | None, page_size: int | None) -> None:
    """List scorers with optional pagination.

    \b
    Examples:
      stratix scorer list
      stratix scorer list --page-size 10
    """
    client = get_client(ctx)
    result = client.scorers.get_many(page=page, page_size=page_size)
    if result is None or not result.scorers:
        click.echo("No scorers found.")
        return

    if ctx.obj["verbose"]:
        click.echo(f"Showing {result.count} of {result.total_count} scorers", err=True)

    output = format_output(result.scorers, ctx.obj["output_format"], SCORER_COLUMNS)
    click.echo(output)


@scorer.command("get")
@click.argument("id")
@click.pass_context
@handle_errors
def get_scorer(ctx: click.Context, id: str) -> None:
    """Get a scorer by ID.

    \b
    Examples:
      stratix scorer get abc123
      stratix scorer get abc123 --format json
    """
    client = get_client(ctx)
    s = client.scorers.get(id)
    if s is None:
        click.echo(f"Scorer {id} not found.", err=True)
        sys.exit(1)

    output = format_output(s, ctx.obj["output_format"])
    click.echo(output)


@scorer.command("create")
@click.option("--name", required=True, help="Scorer name (3-64 chars).")
@click.option("--description", required=True, help="Scorer description (10-500 chars).")
@click.option("--model-id", required=True, help="Model ID to use for scoring.")
@click.option("--prompt", required=True, help="Scoring prompt.")
@click.option("--dry-run", is_flag=True, default=False, help="Preview without executing.")
@click.pass_context
@handle_errors
def create_scorer(ctx: click.Context, name: str, description: str, model_id: str, prompt: str, dry_run: bool) -> None:
    """Create a new scorer.

    \b
    Examples:
      stratix scorer create --name "Quality" --description "Evaluate quality" --model-id abc123 --prompt "Rate the quality..."
      stratix scorer create --name "Test" --description "Test scorer" --model-id abc123 --prompt "..." --dry-run
    """
    if dry_run:
        click.echo(f"[dry-run] Would create scorer: {name}")
        click.echo(f"  Model: {model_id}")
        click.echo(f"  Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        return

    client = get_client(ctx)
    s = client.scorers.create(name=name, description=description, model_id=model_id, prompt=prompt)
    if s is None:
        click.echo("Failed to create scorer.", err=True)
        sys.exit(1)

    click.echo(f"Scorer created: {s.id}")
    output = format_output(s, ctx.obj["output_format"])
    click.echo(output)


@scorer.command("delete")
@click.argument("id")
@click.option("--yes", "-y", is_flag=True, default=False, help="Skip confirmation prompt.")
@click.option("--dry-run", is_flag=True, default=False, help="Preview without executing.")
@click.pass_context
@handle_errors
def delete_scorer(ctx: click.Context, id: str, yes: bool, dry_run: bool) -> None:
    """Delete a scorer by ID.

    \b
    Examples:
      stratix scorer delete abc123
      stratix scorer delete abc123 --yes
      stratix scorer delete abc123 --dry-run
    """
    if dry_run:
        click.echo(f"[dry-run] Would delete scorer {id}")
        return

    if not yes:
        click.confirm(f"Are you sure you want to delete scorer {id}?", abort=True)

    client = get_client(ctx)
    success = client.scorers.delete(id)
    if success:
        click.echo(f"Scorer {id} deleted.")
    else:
        click.echo(f"Failed to delete scorer {id}.", err=True)
        sys.exit(1)
