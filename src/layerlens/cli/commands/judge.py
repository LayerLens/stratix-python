from __future__ import annotations

import sys

import click

from .._client import get_client, handle_errors
from .._formatter import format_output
from .._completions import complete_judge, complete_model, complete_trace

JUDGE_COLUMNS = [
    ("id", "ID"),
    ("name", "Name"),
    ("version", "Version"),
    ("run_count", "Runs"),
    ("created_at", "Created"),
]


@click.group()
def judge() -> None:
    """Manage judges.

    \b
    Examples:
      stratix judge list
      stratix judge get <judge-id>
      stratix judge create --name "Quality" --goal "Evaluate response quality"
      stratix judge test --judge-id <id> --trace-id <id>
    """


@judge.command("list")
@click.option("--page", default=None, type=int, help="Page number.")
@click.option("--page-size", default=None, type=int, help="Results per page.")
@click.pass_context
@handle_errors
def list_judges(ctx: click.Context, page: int | None, page_size: int | None) -> None:
    """List judges with optional pagination.

    \b
    Examples:
      stratix judge list
      stratix judge list --page-size 5
    """
    client = get_client(ctx)
    result = client.judges.get_many(page=page, page_size=page_size)
    if result is None or not result.judges:
        click.echo("No judges found.")
        return

    if ctx.obj["verbose"]:
        click.echo(f"Showing {result.count} of {result.total_count} judges", err=True)

    output = format_output(result.judges, ctx.obj["output_format"], JUDGE_COLUMNS)
    click.echo(output)


@judge.command("get")
@click.argument("id", shell_complete=complete_judge)
@click.pass_context
@handle_errors
def get_judge(ctx: click.Context, id: str) -> None:
    """Get a judge by ID.

    \b
    Examples:
      stratix judge get abc123
      stratix judge get abc123 --format json
    """
    client = get_client(ctx)
    j = client.judges.get(id)
    if j is None:
        click.echo(f"Judge {id} not found.", err=True)
        sys.exit(1)

    output = format_output(j, ctx.obj["output_format"])
    click.echo(output)


@judge.command("create")
@click.option("--name", required=True, help="Judge name.")
@click.option("--goal", required=True, help="Evaluation goal description.")
@click.option("--model-id", default=None, shell_complete=complete_model, help="Model ID for the judge.")
@click.pass_context
@handle_errors
def create_judge(ctx: click.Context, name: str, goal: str, model_id: str | None) -> None:
    """Create a new judge.

    \b
    Examples:
      stratix judge create --name "Quality" --goal "Evaluate response quality"
      stratix judge create --name "Safety" --goal "Check for harmful content" --model-id abc123
    """
    client = get_client(ctx)
    j = client.judges.create(name=name, evaluation_goal=goal, model_id=model_id)
    if j is None:
        click.echo("Failed to create judge.", err=True)
        sys.exit(1)

    click.echo(f"Judge created: {j.id}")
    output = format_output(j, ctx.obj["output_format"])
    click.echo(output)


@judge.command("test")
@click.option("--judge-id", required=True, shell_complete=complete_judge, help="Judge ID to test with.")
@click.option("--trace-id", required=True, shell_complete=complete_trace, help="Trace ID to evaluate.")
@click.pass_context
@handle_errors
def test_judge(ctx: click.Context, judge_id: str, trace_id: str) -> None:
    """Test a judge by evaluating a trace.

    Creates a trace evaluation using the specified judge and trace.

    \b
    Examples:
      stratix judge test --judge-id abc123 --trace-id def456
      stratix judge test --judge-id abc123 --trace-id def456 --format json
    """
    client = get_client(ctx)
    te = client.trace_evaluations.create(trace_id=trace_id, judge_id=judge_id)
    if te is None:
        click.echo("Failed to create trace evaluation.", err=True)
        sys.exit(1)

    click.echo(f"Trace evaluation created: {te.id}")
    output = format_output(te, ctx.obj["output_format"])
    click.echo(output)
