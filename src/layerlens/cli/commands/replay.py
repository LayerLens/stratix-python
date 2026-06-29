"""CLI entry points for local replays (``layerlens replay ...``)."""

from __future__ import annotations

import json
from typing import Callable, cast

import click

from ...replay import ReplayRequest, ReplayController
from .._safe_loader import load_callable
from ...models.trace import Trace


def _load_callable(spec: str) -> Callable[..., Trace]:
    """Resolve ``module.submodule:attr`` into a callable (allowlist-guarded).

    See :mod:`layerlens.cli._safe_loader`: the loader refuses any stdlib/builtin
    root BEFORE importing, so a malicious ``--replay-fn`` (``posix:system``,
    ``os:system``, an import-side-effect module) cannot execute.
    """
    return cast(Callable[..., Trace], load_callable(spec, param_hint="--replay-fn"))


@click.group()
def replay() -> None:
    """Replay traces locally with overrides."""


@replay.command("run")
@click.option("--trace-id", required=True)
@click.option("--trace-file", type=click.Path(exists=True, dir_okay=False))
@click.option("--replay-fn", default=None, help="Callable 'module:attr' that replays a trace.")
@click.option("--model-override", default=None)
@click.option("--input-override", multiple=True, help="KEY=VALUE (repeatable).")
@click.option("--prompt-override", multiple=True, help="KEY=VALUE (repeatable).")
def run(
    trace_id: str,
    trace_file: str | None,
    replay_fn: str | None,
    model_override: str | None,
    input_override: tuple,
    prompt_override: tuple,
) -> None:
    """Run a single-trace replay and print the resulting diff."""
    if trace_file:
        with open(trace_file, "r", encoding="utf-8") as fh:
            trace_payload = json.load(fh)
        original = Trace(**trace_payload)
    else:
        original = Trace(
            id=trace_id,
            organization_id="local",
            project_id="local",
            created_at="local",
            filename=f"{trace_id}.json",
            data={},
        )

    request = ReplayRequest(
        trace_id=trace_id,
        model_override=model_override,
        input_overrides=dict(_kv(input_override)),
        prompt_overrides=dict(_kv(prompt_override)),
    )

    fn = _load_callable(replay_fn) if replay_fn else _echo_replay
    controller = ReplayController(fn)
    result = controller.run(original, request)
    click.echo(result.model_dump_json(indent=2))


def _kv(pairs: tuple) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for pair in pairs:
        if "=" not in pair:
            raise click.BadParameter(f"expected KEY=VALUE (got {pair!r})")
        k, v = pair.split("=", 1)
        out.append((k, v))
    return out


def _echo_replay(trace: Trace, _: ReplayRequest) -> Trace:
    """Fallback replay that returns the input trace unchanged."""
    return trace
