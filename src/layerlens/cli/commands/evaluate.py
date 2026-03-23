from __future__ import annotations

import sys

import click

from .._client import get_client, handle_errors, resolve_model, resolve_benchmark
from .._formatter import format_output
from .._completions import complete_model, complete_benchmark, complete_evaluation

EVALUATION_COLUMNS = [
    ("id", "ID"),
    ("status", "Status"),
    ("model_name", "Model"),
    ("benchmark_name", "Benchmark"),
    ("accuracy", "Accuracy"),
    ("submitted_at", "Submitted"),
]


@click.group()
def evaluate() -> None:
    """Manage evaluations.

    \b
    Examples:
      stratix evaluate list
      stratix evaluate get <evaluation-id>
      stratix evaluate run --model gpt-4 --benchmark mmlu --wait
    """


@evaluate.command("list")
@click.option("--page", default=None, type=int, help="Page number.")
@click.option("--page-size", default=None, type=int, help="Results per page.")
@click.option("--status", default=None, help="Filter by status (pending, in-progress, success, failure).")
@click.option(
    "--sort-by", default=None, type=click.Choice(["submitted_at", "accuracy", "average_duration"]), help="Sort field."
)
@click.option("--order", default=None, type=click.Choice(["asc", "desc"]), help="Sort order.")
@click.pass_context
@handle_errors
def list_evaluations(
    ctx: click.Context,
    page: int | None,
    page_size: int | None,
    status: str | None,
    sort_by: str | None,
    order: str | None,
) -> None:
    """List evaluations with optional filtering and pagination.

    \b
    Examples:
      stratix evaluate list
      stratix evaluate list --status success --sort-by accuracy --order desc
      stratix evaluate list --page-size 5
    """
    from ...models import EvaluationStatus

    client = get_client(ctx)

    eval_status = None
    if status:
        try:
            eval_status = EvaluationStatus(status)
        except ValueError:
            click.echo(f"Invalid status: {status}. Valid: {', '.join(s.value for s in EvaluationStatus)}", err=True)
            sys.exit(1)

    result = client.evaluations.get_many(
        page=page,
        page_size=page_size,
        status=eval_status,
        sort_by=sort_by,  # type: ignore[arg-type]
        order=order,  # type: ignore[arg-type]
    )
    if result is None or not result.evaluations:
        click.echo("No evaluations found.")
        return

    if ctx.obj["verbose"]:
        click.echo(
            f"Showing page {result.pagination.page} of {result.pagination.total_pages} ({result.pagination.total_count} total)",
            err=True,
        )

    output = format_output(result.evaluations, ctx.obj["output_format"], EVALUATION_COLUMNS)
    click.echo(output)


@evaluate.command("get")
@click.argument("id", shell_complete=complete_evaluation)
@click.pass_context
@handle_errors
def get_evaluation(ctx: click.Context, id: str) -> None:
    """Get an evaluation by ID.

    \b
    Examples:
      stratix evaluate get abc123
      stratix evaluate get abc123 --format json
    """
    client = get_client(ctx)
    evaluation = client.evaluations.get_by_id(id)
    if evaluation is None:
        click.echo(f"Evaluation {id} not found.", err=True)
        sys.exit(1)

    output = format_output(evaluation, ctx.obj["output_format"])
    click.echo(output)


@evaluate.command("run")
@click.option("--model", "model_id", required=True, shell_complete=complete_model, help="Model ID, key, or name.")
@click.option(
    "--benchmark", "benchmark_id", required=True, shell_complete=complete_benchmark, help="Benchmark ID, key, or name."
)
@click.option("--wait", is_flag=True, default=False, help="Wait for evaluation to complete.")
@click.pass_context
@handle_errors
def run_evaluation(ctx: click.Context, model_id: str, benchmark_id: str, wait: bool) -> None:
    """Run an evaluation with a model and benchmark.

    The --model and --benchmark options accept an ID, key, or name.

    \b
    Examples:
      stratix evaluate run --model gpt-4 --benchmark mmlu
      stratix evaluate run --model abc123-uuid --benchmark def456-uuid --wait
      stratix evaluate run --model "GPT-4" --benchmark "MMLU" --wait --format json
    """
    client = get_client(ctx)

    if ctx.obj["verbose"]:
        click.echo(f"Resolving model: {model_id}", err=True)

    model = resolve_model(client, model_id)
    if model is None:
        click.echo(f"Model not found: {model_id}", err=True)
        sys.exit(1)

    if ctx.obj["verbose"]:
        click.echo(f"Resolved model: {model.name} ({model.id})", err=True)
        click.echo(f"Resolving benchmark: {benchmark_id}", err=True)

    benchmark = resolve_benchmark(client, benchmark_id)
    if benchmark is None:
        click.echo(f"Benchmark not found: {benchmark_id}", err=True)
        sys.exit(1)

    if ctx.obj["verbose"]:
        click.echo(f"Resolved benchmark: {benchmark.name} ({benchmark.id})", err=True)

    click.echo(f"Creating evaluation: {model.name} x {benchmark.name}")

    evaluation = client.evaluations.create(model=model, benchmark=benchmark)
    if evaluation is None:
        click.echo("Failed to create evaluation.", err=True)
        sys.exit(1)

    click.echo(f"Evaluation created: {evaluation.id} (status: {evaluation.status})")

    if wait:
        click.echo("Waiting for completion...")
        evaluation = client.evaluations.wait_for_completion(evaluation)
        if evaluation is None:
            click.echo("Evaluation disappeared while waiting.", err=True)
            sys.exit(1)
        click.echo(f"Evaluation finished: {evaluation.status}")

    output = format_output(evaluation, ctx.obj["output_format"])
    click.echo(output)
