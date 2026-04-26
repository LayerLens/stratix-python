"""Regenerate the dependency-guard baselines from ``pyproject.toml``.

This script is the canonical way to refresh the two baseline files at
``tests/instrument/_baselines/default_dependencies.txt`` and
``tests/instrument/_baselines/resolved_dependencies.txt``.

Run it AFTER making an intentional change to ``[project] dependencies``
in ``pyproject.toml`` (or after accepting an upstream transitive bloat
that you've reviewed and approved).

Requires ``uv`` (https://github.com/astral-sh/uv) on PATH. Install with
``curl -LsSf https://astral.sh/uv/install.sh | sh``.

Usage: ``python scripts/regen_dep_baselines.py``.

The generated files are deterministic (sorted, normalized) so diffs in
PRs are clean.
"""

from __future__ import annotations

import re
import sys
import shutil
import subprocess
from typing import Set, List
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.9/3.10 fallback
    import tomli as tomllib


_REPO_ROOT: Path = Path(__file__).resolve().parents[1]
_PYPROJECT: Path = _REPO_ROOT / "pyproject.toml"
_BASELINE_DIR: Path = _REPO_ROOT / "tests" / "instrument" / "_baselines"
_DEFAULT_BASELINE: Path = _BASELINE_DIR / "default_dependencies.txt"
_RESOLVED_BASELINE: Path = _BASELINE_DIR / "resolved_dependencies.txt"

_DEFAULT_HEADER: str = """\
# Baseline of REQUIRED runtime dependencies for `pip install layerlens`.
#
# Format: one PEP 508 requirement per line, sorted alphabetically by
# package name (PEP 503 normalized). Comments (lines starting with `#`)
# and blank lines are ignored.
#
# This file is consumed by tests/instrument/test_default_install.py to
# guard against accidental dependency additions in the SDK's default
# install set. Adding a line here represents a deliberate, reviewer-
# acknowledged decision to require a new transitive dependency for
# every `pip install layerlens` user.
#
# Adding a new heavy dependency? Put it behind an extra in
# `[project.optional-dependencies]` instead. Only widely-used,
# lightweight, dependency-stable packages belong in the default set.
#
# To regenerate after an intentional change:
#   1. Edit `[project] dependencies` in pyproject.toml.
#   2. Run: python scripts/regen_dep_baselines.py
#   3. Commit both pyproject.toml and this file in the same PR.
"""

_RESOLVED_HEADER: str = """\
# Baseline of TRANSITIVELY-RESOLVED package names for `pip install layerlens`.
#
# Format: one PEP 503 normalized package name per line, sorted
# alphabetically. Comments (lines starting with `#`) and blank lines
# are ignored. Versions are intentionally OMITTED — version drift in
# transitive deps is a separate concern (handled by the lockfile);
# this guard is purely about install-set BLOAT.
#
# This file is consumed by tests/instrument/test_resolved_dep_tree.py
# and `.github/workflows/dep-tree-guard.yaml` to guard against
# transitive bloat. A direct dep with a permissive lower bound can
# pull in a tree that quintuples install size; this baseline catches
# it.
#
# The CI workflow resolves the dependency tree from a clean
# environment (no extras), normalizes the package names, and diffs
# against this file:
#   - ADDITIONS fail the build.
#   - REMOVALS pass (transitive deps disappearing is good news).
#
# Adding a transitively-resolved dep here represents an explicit
# acknowledgement that the new transitive bloat is acceptable.
#
# To regenerate after an intentional change (e.g. bumping the floor
# of a direct dep, accepting a new transitive package):
#   1. Edit `[project] dependencies` in pyproject.toml as desired.
#   2. Run: python scripts/regen_dep_baselines.py
#   3. Commit pyproject.toml AND this file in the same PR.
"""


def _normalize(name: str) -> str:
    """Normalize a distribution name per PEP 503."""
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def _split_name(requirement: str) -> str:
    """Extract the bare package name from a PEP 508 requirement line."""
    bare = re.split(r"[\s\[;<>=!~]", requirement, maxsplit=1)[0]
    return _normalize(bare)


def _read_pyproject_default_deps() -> List[str]:
    """Return the raw ``[project] dependencies`` strings, sorted by name."""
    with _PYPROJECT.open("rb") as fh:
        data = tomllib.load(fh)
    deps = data.get("project", {}).get("dependencies", []) or []
    cleaned: List[str] = [str(d).strip() for d in deps if isinstance(d, str)]
    return sorted(cleaned, key=_split_name)


def _resolve_tree(direct_deps: List[str]) -> List[str]:
    """Return the sorted, deduplicated set of resolved package names.

    Uses ``uv pip compile`` in universal mode for deterministic,
    cross-platform output.
    """
    if shutil.which("uv") is None:
        raise RuntimeError(
            "`uv` is required to regenerate the resolved-tree baseline.\n"
            "Install: https://github.com/astral-sh/uv\n"
            "  curl -LsSf https://astral.sh/uv/install.sh | sh"
        )

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
        check=True,
    )
    output = proc.stdout.decode("utf-8")

    names: Set[str] = set()
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # `uv pip compile --universal` may emit `name==ver ; marker` —
        # we only need the name.
        names.add(_split_name(line))
    return sorted(names)


def _write_default_baseline(direct_deps: List[str]) -> None:
    body = "\n".join(direct_deps)
    _DEFAULT_BASELINE.write_text(_DEFAULT_HEADER + body + "\n", encoding="utf-8")


def _write_resolved_baseline(resolved_names: List[str]) -> None:
    body = "\n".join(resolved_names)
    _RESOLVED_BASELINE.write_text(_RESOLVED_HEADER + body + "\n", encoding="utf-8")


def main() -> int:
    direct_deps = _read_pyproject_default_deps()
    resolved_names = _resolve_tree(direct_deps)

    _BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    _write_default_baseline(direct_deps)
    _write_resolved_baseline(resolved_names)

    sys.stdout.write(f"Wrote {_DEFAULT_BASELINE.relative_to(_REPO_ROOT)} ({len(direct_deps)} direct deps)\n")
    sys.stdout.write(f"Wrote {_RESOLVED_BASELINE.relative_to(_REPO_ROOT)} ({len(resolved_names)} resolved names)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
