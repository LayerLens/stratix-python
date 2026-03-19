from __future__ import annotations

import sys
import json

import click

from .._client import get_client, handle_errors, resolve_model, resolve_benchmark
from .._formatter import format_output

EVALUATION_COLUMNS = [
    ("id", "ID"),
    ("status", "Status"),
    ("model_name", "Model"),
    ("benchmark_name", "Benchmark"),
]


@click.group()
def bulk() -> None:
    """Bulk operations.

    \b
    Examples:
      layerlens bulk eval --file jobs.jsonl
      layerlens bulk eval --model gpt-4 --benchmark mmlu --traces trace_ids.txt
    """


@bulk.command("eval")
@click.option(
    "--file",
    "file_path",
    type=click.Path(exists=True),
    help='JSONL file with evaluation jobs (each line: {"model": ..., "benchmark": ...}).',
)
@click.option("--model", "model_id", default=None, help="Model ID/name (use with --benchmark).")
@click.option("--benchmark", "benchmark_id", default=None, help="Benchmark ID/name (use with --model).")
@click.option("--judge-id", default=None, help="Judge ID (use with --traces).")
@click.option(
    "--traces", "traces_file", type=click.Path(exists=True), default=None, help="File with trace IDs (one per line)."
)
@click.option("--dry-run", is_flag=True, default=False, help="Preview without executing.")
@click.option("--wait", is_flag=True, default=False, help="Wait for all evaluations to complete.")
@click.pass_context
@handle_errors
def bulk_eval(
    ctx: click.Context,
    file_path: str | None,
    model_id: str | None,
    benchmark_id: str | None,
    judge_id: str | None,
    traces_file: str | None,
    dry_run: bool,
    wait: bool,
) -> None:
    """Run evaluations in bulk from a file or stdin.

    Three modes:
    1. JSONL file: each line is {"model": "<id>", "benchmark": "<id>"}
    2. Model + benchmark: run a single evaluation (optionally --wait)
    3. Judge + traces file: evaluate many traces with a judge

    \b
    Examples:
      layerlens bulk eval --file jobs.jsonl
      layerlens bulk eval --file jobs.jsonl --dry-run
      layerlens bulk eval --model gpt-4 --benchmark mmlu --wait
      layerlens bulk eval --judge-id <id> --traces trace_ids.txt
    """
    client = get_client(ctx)

    if file_path:
        with open(file_path) as f:
            jobs = []
            for line in f:
                line = line.strip()
                if line:
                    try:
                        jobs.append(json.loads(line))
                    except json.JSONDecodeError:
                        click.echo(f"Skipping invalid JSON line: {line}", err=True)

        if not jobs:
            click.echo("No valid jobs found in file.", err=True)
            sys.exit(1)

        if dry_run:
            click.echo(f"[dry-run] Would create {len(jobs)} evaluation(s):")
            for job in jobs:
                click.echo(f"  model={job.get('model')} benchmark={job.get('benchmark')}")
            return

        click.echo(f"Creating {len(jobs)} evaluation(s)...")
        evaluations = []
        for i, job in enumerate(jobs, 1):
            m = resolve_model(client, job["model"])
            b = resolve_benchmark(client, job["benchmark"])
            if m is None or b is None:
                click.echo(
                    f"  [{i}] SKIP - model={job.get('model')} or benchmark={job.get('benchmark')} not found", err=True
                )
                continue

            ev = client.evaluations.create(model=m, benchmark=b)
            if ev:
                click.echo(f"  [{i}] Created: {ev.id}")
                evaluations.append(ev)
            else:
                click.echo(f"  [{i}] FAIL", err=True)

        click.echo(f"\n{len(evaluations)} evaluation(s) created.")

        if wait and evaluations:
            click.echo("Waiting for completion...")
            for ev in evaluations:
                result = client.evaluations.wait_for_completion(ev)
                if result:
                    click.echo(f"  {result.id}: {result.status}")

    elif model_id and benchmark_id:
        model = resolve_model(client, model_id)
        if model is None:
            click.echo(f"Model not found: {model_id}", err=True)
            sys.exit(1)
        benchmark = resolve_benchmark(client, benchmark_id)
        if benchmark is None:
            click.echo(f"Benchmark not found: {benchmark_id}", err=True)
            sys.exit(1)

        if traces_file:
            click.echo("Error: --traces requires --judge-id, not --model/--benchmark.", err=True)
            sys.exit(1)

        else:
            if dry_run:
                click.echo(f"[dry-run] Would create evaluation: {model.name} x {benchmark.name}")
                return

            click.echo(f"Creating evaluation: {model.name} x {benchmark.name}")
            ev = client.evaluations.create(model=model, benchmark=benchmark)
            if ev is None:
                click.echo("Failed to create evaluation.", err=True)
                sys.exit(1)

            click.echo(f"Evaluation created: {ev.id}")
            if wait:
                click.echo("Waiting for completion...")
                ev = client.evaluations.wait_for_completion(ev)
                if ev:
                    click.echo(f"Evaluation finished: {ev.status}")

            output = format_output(ev, ctx.obj["output_format"])
            click.echo(output)
    elif judge_id and traces_file:
        # Mode 3: judge + traces file
        with open(traces_file) as f:
            trace_ids = [line.strip() for line in f if line.strip()]

        if not trace_ids:
            click.echo("No trace IDs found in file.", err=True)
            sys.exit(1)

        if dry_run:
            click.echo(f"[dry-run] Would create {len(trace_ids)} trace evaluation(s) with judge {judge_id}:")
            for tid in trace_ids:
                click.echo(f"  trace={tid}")
            return

        click.echo(f"Creating {len(trace_ids)} trace evaluation(s) with judge {judge_id}...")
        results = []
        for i, trace_id in enumerate(trace_ids, 1):
            te = client.trace_evaluations.create(trace_id=trace_id, judge_id=judge_id)
            if te:
                click.echo(f"  [{i}] Created: {te.id} (trace={trace_id})")
                results.append(te)
            else:
                click.echo(f"  [{i}] FAIL (trace={trace_id})", err=True)

        click.echo(f"\n{len(results)} trace evaluation(s) created.")
    else:
        click.echo("Provide --file, --model + --benchmark, or --judge-id + --traces.", err=True)
        sys.exit(1)
