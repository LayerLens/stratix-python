"""CLI entry points for synthetic trace generation."""

from __future__ import annotations

import sys
import json

import click

from ...synthetic import SyntheticDataBuilder


@click.group()
def synthetic() -> None:
    """Generate synthetic traces from templates."""


@synthetic.command("templates")
def templates() -> None:
    """List available templates."""
    for t in SyntheticDataBuilder().list_templates():
        click.echo(f"{t.id:30s} {t.category.value:15s} {t.title}")


@synthetic.command("generate")
@click.option("--template", "template_id", required=True)
@click.option("--count", type=int, default=10)
@click.option("--provider", default=None)
@click.option("--project-id", default=None)
@click.option("--organization-id", default=None)
@click.option(
    "--out",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help="Write traces to this JSONL file (default: stdout).",
)
def generate(
    template_id: str,
    count: int,
    provider: str | None,
    project_id: str | None,
    organization_id: str | None,
    out: str | None,
) -> None:
    """Generate N synthetic traces from TEMPLATE."""
    result = SyntheticDataBuilder().generate(
        template_id,
        count,
        provider_id=provider,
        project_id=project_id,
        organization_id=organization_id,
    )
    if result.errors:
        click.echo(f"errors: {result.errors}", err=True)
        sys.exit(1)
    sink = open(out, "w", encoding="utf-8") if out else sys.stdout
    try:
        for trace in result.traces:
            sink.write(json.dumps(trace.model_dump()) + "\n")
    finally:
        if out:
            sink.close()
    click.echo(
        f"generated {len(result.traces)} traces (job={result.job_id})",
        err=True,
    )
