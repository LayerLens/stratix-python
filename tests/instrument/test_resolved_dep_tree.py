"""Resolved transitive-dependency-tree guard.

A direct dep with a permissive lower bound can pull in a tree that
quintuples install size. ``Requires-Dist`` only shows direct deps —
the actual install footprint is the TRANSITIVE closure of every
direct dep at the version pip's resolver picks.

This test compares the transitively-resolved package-name set for
``pip install layerlens`` (no extras) against a checked-in baseline
at ``tests/instrument/_baselines/resolved_dependencies.txt``.

Modes
-----

The test runs in one of two modes depending on environment:

1. **Offline / no-uv mode** (default for `pytest` runs without `uv` on
   PATH): the test only validates the baseline file's structure
   (sorted, normalized, no duplicates) and that every direct dep from
   ``pyproject.toml`` is also present in the resolved baseline (which
   it must be — direct deps always appear in their own resolved tree).

2. **Online mode** (when ``uv`` is on PATH AND
   ``LAYERLENS_RESOLVE_DEPS=1`` is set, OR running under CI): the test
   invokes ``uv pip compile`` to actually resolve the tree, then diffs
   the resolved name set against the baseline. Additions fail; removals
   pass with a hint to regenerate the baseline.

The CI workflow ``.github/workflows/dep-tree-guard.yaml`` always runs
in online mode. Local runs default to offline so devs without ``uv``
installed can still iterate on the test suite.
"""

from __future__ import annotations

import os
import re
import sys
import shutil
import subprocess
from typing import Set, List
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.9/3.10 fallback
    import tomli as tomllib


_REPO_ROOT: Path = Path(__file__).resolve().parents[2]
_PYPROJECT: Path = _REPO_ROOT / "pyproject.toml"
_BASELINE_PATH: Path = Path(__file__).resolve().parent / "_baselines" / "resolved_dependencies.txt"


def _normalize(name: str) -> str:
    """Normalize a distribution name per PEP 503."""
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def _split_name(requirement: str) -> str:
    """Extract the bare package name from a PEP 508 requirement line."""
    bare = re.split(r"[\s\[;<>=!~]", requirement, maxsplit=1)[0]
    return _normalize(bare)


def _read_baseline_names() -> List[str]:
    """Return the sorted list of normalized names in the baseline file."""
    raw = _BASELINE_PATH.read_text(encoding="utf-8").splitlines()
    out: List[str] = []
    for line in raw:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        out.append(_split_name(stripped))
    return out


def _read_pyproject_direct_deps() -> List[str]:
    """Return the raw ``[project] dependencies`` strings."""
    with _PYPROJECT.open("rb") as fh:
        data = tomllib.load(fh)
    deps = data.get("project", {}).get("dependencies", []) or []
    return [str(d).strip() for d in deps if isinstance(d, str)]


def _resolve_tree_via_uv(direct_deps: List[str]) -> Set[str]:
    """Invoke ``uv pip compile`` and return the resolved name set."""
    proc = subprocess.run(
        [
            "uv",
            "pip",
            "compile",
            "-q",
            "--no-header",
            "--no-annotate",
            "--no-strip-extras",
            "--universal",
            "-",
        ],
        input="\n".join(direct_deps).encode("utf-8"),
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"`uv pip compile` failed (exit {proc.returncode}):\n{stderr}")
    output = proc.stdout.decode("utf-8")

    names: Set[str] = set()
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        names.add(_split_name(line))
    return names


def _online_mode_requested() -> bool:
    """Return whether the test should perform a live resolve."""
    if shutil.which("uv") is None:
        return False
    if os.environ.get("CI") == "true":
        return True
    return os.environ.get("LAYERLENS_RESOLVE_DEPS") == "1"


def test_baseline_file_is_sorted_and_well_formed() -> None:
    """The baseline must be sorted, normalized, and free of duplicates."""
    names = _read_baseline_names()
    assert names, "Baseline file must contain at least one resolved package name."

    sorted_names = sorted(names)
    assert names == sorted_names, (
        "Baseline file must be sorted alphabetically by normalized package name.\n"
        f"  Expected: {sorted_names}\n"
        f"  Actual:   {names}"
    )

    # No duplicates.
    assert len(names) == len(set(names)), (
        f"Duplicate names in baseline: {sorted({n for n in names if names.count(n) > 1})}"
    )

    # Every line must already be in normalized form.
    for n in names:
        assert n == _normalize(n), f"Baseline contains non-normalized name {n!r}; expected {_normalize(n)!r}."


def test_baseline_includes_every_direct_dep() -> None:
    """Every direct dep in pyproject.toml must appear in the resolved baseline.

    This is a tautology in any consistent baseline (a package is always
    in its own resolved tree), but the check catches the case where a
    direct dep was added to pyproject.toml without regenerating the
    baseline.
    """
    direct_names = {_split_name(req) for req in _read_pyproject_direct_deps()}
    baseline_names = set(_read_baseline_names())
    missing = direct_names - baseline_names
    assert not missing, (
        f"Direct dep(s) in pyproject.toml not present in resolved baseline: "
        f"{sorted(missing)}.\n"
        f"  Run `python scripts/regen_dep_baselines.py` to refresh."
    )


@pytest.mark.skipif(
    not _online_mode_requested(),
    reason=(
        "Live dependency resolution requires `uv` on PATH and either "
        "CI=true or LAYERLENS_RESOLVE_DEPS=1. Skipping in offline mode."
    ),
)
def test_resolved_tree_matches_baseline() -> None:
    """The live-resolved tree MUST NOT add packages beyond the baseline."""
    direct_deps = _read_pyproject_direct_deps()
    resolved = _resolve_tree_via_uv(direct_deps)
    baseline = set(_read_baseline_names())

    added = resolved - baseline
    removed = baseline - resolved

    assert not added, (
        f"Resolved dependency tree added packages NOT in the baseline: "
        f"{sorted(added)}.\n"
        f"  This means a direct dep started pulling in new transitive deps.\n"
        f"  If the addition is acceptable, regenerate the baseline:\n"
        f"    python scripts/regen_dep_baselines.py\n"
        f"  Otherwise, tighten the version specifier on the offending direct dep."
    )

    if removed:
        # Removals are good news (less bloat) but we still report them so
        # devs can refresh the baseline. Don't fail the test; this is a
        # one-way ratchet that only blocks ADDITIONS.
        sys.stderr.write(
            f"\nNOTE: resolved tree no longer pulls in: {sorted(removed)}.\n"
            f"  Consider running `python scripts/regen_dep_baselines.py` "
            f"to tighten the baseline.\n"
        )
