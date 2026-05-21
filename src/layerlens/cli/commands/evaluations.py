"""CLI entry points for local dataset-driven evaluation runs."""

from __future__ import annotations

import sys
import json
import importlib
from typing import Any, Dict, Callable, cast

import click

from ...datasets import Dataset, DatasetVisibility, InMemoryDatasetStore
from ...evaluation_runs import RunComparer, EvaluationRunner
from ...evaluation_runs.models import ScorerFn

# Stdlib modules that expose process-control primitives. Naming any of these
# as a ``--target`` or ``--scorer`` callable is almost certainly misuse or an
# injection attempt, so we refuse up-front rather than executing them.
_BLOCKED_MODULES = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "shutil",
        "builtins",
        "importlib",
        "runpy",
        "ctypes",
        "pty",
        "pickle",
        "marshal",
        "socket",
    }
)


def _load_callable(spec: str, *, param_hint: str = "--target") -> Callable[..., Any]:
    if ":" not in spec:
        raise click.BadParameter(f"expected 'module:attr' (got {spec!r})", param_hint=param_hint)
    module_name, attr = spec.split(":", 1)
    root = module_name.split(".", 1)[0]
    if root in _BLOCKED_MODULES:
        raise click.BadParameter(
            f"refusing to load callable from stdlib module {root!r}",
            param_hint=param_hint,
        )
    module = importlib.import_module(module_name)
    fn = getattr(module, attr, None)
    if fn is None or not callable(fn):
        raise click.BadParameter(f"{spec!r} is not callable", param_hint=param_hint)
    return cast(Callable[..., Any], fn)


@click.group()
def evaluations() -> None:
    """Run and compare dataset-scoped evaluations locally."""


@evaluations.command("run")
@click.option("--dataset-id", required=True)
@click.option(
    "--dataset-file",
    type=click.Path(exists=True, dir_okay=False),
    help="Load a dataset from a JSON file (list of {input, expected_output}).",
)
@click.option("--target", required=True, help="Callable 'module:attr'.")
@click.option(
    "--scorer",
    "scorers",
    multiple=True,
    help="Scorer 'name=module:attr' (repeatable). Default: 'exact' equality.",
)
@click.option(
    "--out",
    type=click.Path(dir_okay=False, writable=True),
    default=None,
    help="Write the run to this JSON file (default: stdout).",
)
def run(
    dataset_id: str,
    dataset_file: str | None,
    target: str,
    scorers: tuple,
    out: str | None,
) -> None:
    """Execute an evaluation run and print the aggregated results."""
    store = InMemoryDatasetStore()
    if dataset_file:
        with open(dataset_file, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        ds = store.create(Dataset(id=dataset_id, name=dataset_id, visibility=DatasetVisibility.PRIVATE))
        store.import_items(ds.id, raw)
    else:
        raise click.UsageError("remote dataset lookup is not yet implemented — pass --dataset-file")

    target_fn = _load_callable(target)
    scorer_map: Dict[str, ScorerFn] = {}
    if scorers:
        for spec in scorers:
            if "=" not in spec:
                raise click.BadParameter(f"expected name=module:attr (got {spec!r})")
            name, fn_spec = spec.split("=", 1)
            scorer_map[name] = cast(ScorerFn, _load_callable(fn_spec, param_hint="--scorer"))
    else:

        def _exact(actual: Any, expected: Any, _meta: Any) -> float:
            return 1.0 if actual == expected else 0.0

        scorer_map["exact"] = _exact

    run_obj = EvaluationRunner(store).run(dataset_id=ds.id, target=target_fn, scorers=scorer_map)
    payload = run_obj.model_dump_json(indent=2)
    if out:
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(payload)
    else:
        sys.stdout.write(payload + "\n")


@evaluations.command("compare")
@click.argument("baseline", type=click.Path(exists=True, dir_okay=False))
@click.argument("candidate", type=click.Path(exists=True, dir_okay=False))
@click.option("--score-tolerance", type=float, default=0.02)
def compare(baseline: str, candidate: str, score_tolerance: float) -> None:
    """Diff two previously-saved evaluation runs and exit non-zero on regression."""
    from ...evaluation_runs.models import EvaluationRun

    with open(baseline, "r", encoding="utf-8") as fh:
        base_run = EvaluationRun(**json.load(fh))
    with open(candidate, "r", encoding="utf-8") as fh:
        cand_run = EvaluationRun(**json.load(fh))

    cmp = RunComparer(score_tolerance=score_tolerance).compare(base_run, cand_run)
    click.echo(cmp.model_dump_json(indent=2))
    if cmp.is_regression:
        sys.exit(1)
