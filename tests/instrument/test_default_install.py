"""Default-install integrity guard.

Adding adapter extras to ``pyproject.toml`` MUST NOT change the runtime
dependency set installed by a plain ``pip install layerlens``. This
test reads ``[project] dependencies`` directly from ``pyproject.toml``
and asserts the required dependency list matches the canonical baseline
checked in at ``tests/instrument/_baselines/default_dependencies.txt``.

Two parallel checks run:

1. **Direct deps from pyproject.toml** vs. the checked-in baseline file.
   This is the load-bearing source of truth — what new SDK releases
   actually advertise as required.
2. **Installed metadata Requires-Dist** vs. the same baseline.
   Belt-and-suspenders: catches mismatch between source-of-truth and
   what the wheel actually ships.

If you add a new required dependency to ``[project] dependencies`` in
``pyproject.toml`` (rare and intentional), update the baseline file in
the same PR. If you add an extras group, no change is needed — extras
are not in ``Requires-Dist`` until a user opts in.
"""

from __future__ import annotations

import re
import sys
from typing import Set, Dict, List, Tuple
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.9/3.10 fallback
    import tomli as tomllib


_REPO_ROOT: Path = Path(__file__).resolve().parents[2]
_PYPROJECT: Path = _REPO_ROOT / "pyproject.toml"
_BASELINE_PATH: Path = Path(__file__).resolve().parent / "_baselines" / "default_dependencies.txt"


def _normalize(name: str) -> str:
    """Normalize a distribution name per PEP 503."""
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def _split_name(requirement: str) -> str:
    """Extract the bare package name from a PEP 508 requirement line."""
    # PEP 508 grammar: name[extras] specifier ; marker
    # We just need the name, which terminates at: whitespace, `[`, `;`,
    # `<`, `>`, `=`, `!`, `~`, or end-of-string.
    bare = re.split(r"[\s\[;<>=!~]", requirement, maxsplit=1)[0]
    return _normalize(bare)


def _read_baseline_file() -> Tuple[List[str], Dict[str, str]]:
    """Return (raw_lines, name->requirement) from the baseline file.

    Comments and blank lines are stripped from the returned data
    structures but the raw list preserves order for diagnostic output.
    """
    raw = _BASELINE_PATH.read_text(encoding="utf-8").splitlines()
    by_name: Dict[str, str] = {}
    for line in raw:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        by_name[_split_name(stripped)] = stripped
    return raw, by_name


def _read_pyproject_default_deps() -> Dict[str, str]:
    """Return name -> raw requirement string from ``[project] dependencies``."""
    with _PYPROJECT.open("rb") as fh:
        data = tomllib.load(fh)
    deps = data.get("project", {}).get("dependencies", []) or []
    out: Dict[str, str] = {}
    for req in deps:
        if not isinstance(req, str):
            continue
        out[_split_name(req)] = req.strip()
    return out


def _required_dist_names() -> Set[str]:
    """Read ``layerlens``'s installed metadata and return required dep names.

    Skips requirements gated by an ``extra ==`` marker — those are
    optional dependencies, not part of the default install set.
    """
    from importlib.metadata import distribution

    dist = distribution("layerlens")
    requires = dist.requires or []
    names: Set[str] = set()
    for req in requires:
        if "extra ==" in req:
            continue
        names.add(_split_name(req))
    return names


def test_pyproject_default_dependencies_match_baseline() -> None:
    """``[project] dependencies`` in pyproject.toml MUST equal the baseline."""
    pyproject_deps = _read_pyproject_default_deps()
    _, baseline_by_name = _read_baseline_file()

    pyproject_names = set(pyproject_deps)
    baseline_names = set(baseline_by_name)

    added = pyproject_names - baseline_names
    removed = baseline_names - pyproject_names

    assert not added, (
        f"New required dependency added to pyproject.toml that is NOT in the "
        f"checked-in baseline: {sorted(added)}.\n"
        f"  Baseline file: {_BASELINE_PATH}\n"
        f"  Either move the dep into an extras group in pyproject.toml,\n"
        f"  OR justify the addition in the PR description and update the\n"
        f"  baseline file in the same PR."
    )
    assert not removed, (
        f"Baseline lists dependencies not present in pyproject.toml: "
        f"{sorted(removed)}.\n"
        f"  Baseline file: {_BASELINE_PATH}\n"
        f"  If the removal is intentional, update the baseline file."
    )

    # Also verify the version specifier matches exactly. A silent bump of
    # a lower bound would be a behaviour change worth surfacing.
    for name in sorted(pyproject_names):
        assert pyproject_deps[name] == baseline_by_name[name], (
            f"Version specifier drift for `{name}`:\n"
            f"  pyproject.toml: {pyproject_deps[name]!r}\n"
            f"  baseline:       {baseline_by_name[name]!r}\n"
            f"  Update the baseline file if the bump is intentional."
        )


def test_installed_metadata_matches_baseline() -> None:
    """Installed wheel ``Requires-Dist`` MUST match the baseline name set."""
    actual = _required_dist_names()
    _, baseline_by_name = _read_baseline_file()
    expected = set(baseline_by_name)

    extra = actual - expected
    missing = expected - actual

    assert not extra, (
        f"Installed `layerlens` advertises required deps not in the baseline: "
        f"{sorted(extra)}.\n"
        f"  This means the built wheel diverged from pyproject.toml — investigate."
    )
    assert not missing, (
        f"Installed `layerlens` is missing baseline-required deps: "
        f"{sorted(missing)}.\n"
        f"  Reinstall the package: `pip install -e .`"
    )


def test_baseline_file_is_sorted_and_well_formed() -> None:
    """The baseline file must be sorted and have one requirement per line."""
    raw, by_name = _read_baseline_file()

    # Filter to the data lines and verify sort order.
    data_lines: List[str] = [line.strip() for line in raw if line.strip() and not line.strip().startswith("#")]
    sorted_data = sorted(data_lines, key=_split_name)
    assert data_lines == sorted_data, (
        "Baseline file must be sorted alphabetically by normalized package name.\n"
        f"  Expected order: {sorted_data}\n"
        f"  Actual order:   {data_lines}"
    )

    # No duplicate names.
    seen: Set[str] = set()
    for line in data_lines:
        name = _split_name(line)
        assert name not in seen, f"Duplicate dependency in baseline: {name}"
        seen.add(name)

    # by_name was populated, so the file is non-empty.
    assert by_name, "Baseline file must contain at least one dependency."
