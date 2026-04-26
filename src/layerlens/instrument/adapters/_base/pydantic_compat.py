"""Per-adapter Pydantic version compatibility declarations.

Round-2 deliberation item 20: surface each adapter's Pydantic v1 / v2 /
both compatibility so that importing a v2-only adapter under a v1-pinned
runtime fails fast with a clear message instead of producing a confusing
``ImportError`` deep inside the framework SDK.

Three values exist:

* :attr:`PydanticCompat.V1_ONLY` — adapter or its underlying framework
  uses Pydantic v1 idioms (``@root_validator``, ``model.dict()``,
  ``Config`` inner class) that break under v2.
* :attr:`PydanticCompat.V2_ONLY` — adapter or its underlying framework
  uses v2-only API surface (``@field_validator``, ``@model_validator``,
  ``model.model_dump()``, ``Annotated`` constraints, etc.). Pinning a v1
  Pydantic with this adapter raises at import.
* :attr:`PydanticCompat.V1_OR_V2` — adapter is Pydantic-version-agnostic.
  Either it imports nothing from ``pydantic`` directly, or it routes all
  Pydantic access through :mod:`layerlens._compat.pydantic`.

The :func:`requires_pydantic` helper is meant to be called at adapter
module import time after the version constant is declared::

    from layerlens.instrument.adapters._base.pydantic_compat import (
        PydanticCompat,
        requires_pydantic,
    )

    requires_pydantic(PydanticCompat.V2_ONLY)

If the runtime pydantic does not satisfy the declaration, the call
raises :class:`RuntimeError` with a message naming the adapter, the
required version, and the installed version.
"""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Optional

import pydantic

from layerlens._compat.pydantic import PYDANTIC_V2


class PydanticCompat(str, Enum):
    """Adapter declaration of which Pydantic major versions it supports."""

    V1_ONLY = "v1_only"
    V2_ONLY = "v2_only"
    V1_OR_V2 = "v1_or_v2"


def _runtime_pydantic_version() -> str:
    """Return the installed pydantic version string (e.g. ``"2.11.7"``)."""
    return str(getattr(pydantic, "VERSION", "unknown"))


def _caller_module_name() -> Optional[str]:
    """Best-effort lookup of the importing adapter's module name.

    Walks two frames up (past :func:`requires_pydantic`) and returns the
    ``__name__`` of the calling module. Used purely to make the
    :class:`RuntimeError` message actionable; never load-bearing.
    """
    frame = inspect.currentframe()
    if frame is None:
        return None
    try:
        outer = frame.f_back
        if outer is None:
            return None
        caller = outer.f_back
        if caller is None:
            return None
        return caller.f_globals.get("__name__")
    finally:
        del frame


def requires_pydantic(version: PydanticCompat) -> None:
    """Validate that the runtime Pydantic matches an adapter's declaration.

    Call from an adapter module's import path immediately after declaring
    its compatibility constant. Raises :class:`RuntimeError` with a clear,
    user-actionable message if the runtime Pydantic does not match.

    Args:
        version: The adapter's :class:`PydanticCompat` declaration.

    Raises:
        RuntimeError: If the runtime Pydantic version is incompatible
            with the declaration. The message identifies the calling
            adapter module so users can pin the correct extra.
    """
    if version is PydanticCompat.V1_OR_V2:
        return

    if version is PydanticCompat.V2_ONLY and not PYDANTIC_V2:
        caller = _caller_module_name() or "<unknown adapter>"
        raise RuntimeError(
            f"{caller} requires Pydantic v2 (declared {version.value}); "
            f"runtime is pydantic {_runtime_pydantic_version()}. "
            "Upgrade with `pip install 'pydantic>=2,<3'` or remove the "
            "adapter extra from your install set."
        )

    if version is PydanticCompat.V1_ONLY and PYDANTIC_V2:
        caller = _caller_module_name() or "<unknown adapter>"
        raise RuntimeError(
            f"{caller} requires Pydantic v1 (declared {version.value}); "
            f"runtime is pydantic {_runtime_pydantic_version()}. "
            "Pin with `pip install 'pydantic>=1.9,<2'` or remove the "
            "adapter extra from your install set."
        )


__all__ = [
    "PydanticCompat",
    "requires_pydantic",
]
