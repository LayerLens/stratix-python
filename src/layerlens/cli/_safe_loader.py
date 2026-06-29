"""Safe dynamic-callable loader for the CLI (replay-fn / --target / --scorer).

The ``replay``/``evaluations`` commands resolve a ``module:attr`` string the
user passes on the command line into a callable that is then INVOKED. That is a
remote-code-execution surface: whatever the loader resolves runs in-process.

The old guard was a 12-name stdlib *denylist* (``os``, ``subprocess``, ...). A
denylist of a security boundary is the wrong shape — it fails OPEN for every
name the author did not think of:

* ``posix:system`` resolves ``posix.system`` which IS ``os.system`` (the C
  implementation), and ``posix`` was not on the list.
* ``importlib.import_module`` runs the target module's top-level code, so naming
  ANY importable module with an import-time side effect (``antigravity`` opens a
  browser, ``this`` prints, a malicious package's ``__init__`` runs) executes
  before any attribute is even looked up.

This module replaces the denylist with an ALLOWLIST resolved BEFORE any import:
a callable may only be loaded from a module that resolves (via
:func:`importlib.util.find_spec`, which does NOT execute the target module's own
top-level code) to a real source file located OUTSIDE the standard library, and
whose root is not a C builtin. In practice that means the caller's own
application package (or an installed third-party package) — never an stdlib
process-control primitive and never a frozen/builtin module. The membership
check runs first, so a denied root is never imported (no import-time side
effect can fire).
"""

from __future__ import annotations

import sys
import importlib
import sysconfig
import importlib.util
from typing import Any, Callable, cast

import click


def _stdlib_dirs() -> tuple[str, ...]:
    """Directories that hold the standard library for the running interpreter."""
    dirs: list[str] = []
    for key in ("stdlib", "platstdlib"):
        try:
            path = sysconfig.get_paths().get(key)
        except Exception:  # pragma: no cover - sysconfig is always present
            path = None
        if path:
            dirs.append(path)
    return tuple(dirs)


def _is_stdlib_root(root: str) -> bool:
    """True iff *root* is a standard-library or C-builtin top-level module.

    Uses ``find_spec`` on the ROOT name only — this never executes the target
    module's own top-level code (verified: ``find_spec('antigravity')`` does not
    open a browser), so it is safe to call on an untrusted spec BEFORE importing.
    A root that does not resolve to a real file (frozen/builtin like ``posix``,
    ``sys``, ``builtins``) is treated as stdlib and refused.
    """
    if root in sys.builtin_module_names:
        return True
    try:
        spec = importlib.util.find_spec(root)
    except (ImportError, AttributeError, ValueError):
        # A parent-less or otherwise unresolvable spec is not a loadable app
        # module — refuse it (fail closed).
        return True
    if spec is None:
        # Unknown module: let the later import_module raise a clear error rather
        # than masquerading as a stdlib refusal — but treat a missing origin as
        # non-loadable (fail closed) below.
        return False
    origin = getattr(spec, "origin", None)
    if origin is None or origin in ("built-in", "frozen"):
        # Frozen / builtin / namespace-without-file: no real source file ⇒ not an
        # application module. Fail closed.
        return True
    stdlib_dirs = _stdlib_dirs()
    return any(_within(origin, d) for d in stdlib_dirs)


def _within(path: str, directory: str) -> bool:
    """True iff *path* lives under *directory* (normalized, boundary-safe)."""
    import os

    try:
        path_r = os.path.realpath(path)
        dir_r = os.path.realpath(directory)
    except Exception:  # pragma: no cover - realpath rarely raises here
        return False
    return path_r == dir_r or path_r.startswith(dir_r + os.sep)


def load_callable(spec: str, *, param_hint: str = "--target") -> Callable[..., Any]:
    """Resolve ``module.submodule:attr`` into a callable, allowlist-guarded.

    Refuses, BEFORE importing anything, any spec whose root module is part of the
    standard library or is a C/frozen builtin. The membership check precedes
    ``import_module`` so a refused target's import-time side effects never run.
    """
    if ":" not in spec:
        raise click.BadParameter(f"expected 'module:attr' (got {spec!r})", param_hint=param_hint)
    module_name, attr = spec.split(":", 1)
    if not module_name or not attr:
        raise click.BadParameter(f"expected 'module:attr' (got {spec!r})", param_hint=param_hint)
    root = module_name.split(".", 1)[0]
    # ALLOWLIST gate — runs BEFORE any import so a denied root never executes.
    if _is_stdlib_root(root):
        raise click.BadParameter(
            f"refusing to load callable from non-application module {root!r} — "
            "only callables from your own/installed application packages may be "
            "loaded (stdlib and builtin modules are blocked to prevent code "
            "execution)",
            param_hint=param_hint,
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise click.BadParameter(f"could not import {module_name!r}: {exc}", param_hint=param_hint) from None
    fn = getattr(module, attr, None)
    if fn is None or not callable(fn):
        raise click.BadParameter(f"{spec!r} is not callable", param_hint=param_hint)
    return cast(Callable[..., Any], fn)
