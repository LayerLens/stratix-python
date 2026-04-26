"""Pydantic v1/v2 dual-compatibility shim.

`stratix-python` pins ``pydantic>=1.9.0, <3``. The instrument layer must
work under both v1 and v2 because frameworks we adapt (LangChain, CrewAI,
Pydantic-AI, etc.) span both versions in customer environments.

This shim exposes a single set of names — ``BaseModel``, ``Field``,
``model_dump``, ``field_validator``, ``model_validator`` — that behave
identically under both versions. Callers must use these instead of
importing from ``pydantic`` directly so the v1/v2 boundary lives in
exactly one place.
"""

from __future__ import annotations

from typing import Any, Dict, Callable

import pydantic

PYDANTIC_V2: bool = pydantic.VERSION.startswith("2.")

# Re-exported public names. Adapter code imports from here, never from
# ``pydantic`` directly, so a future v3 (or rollback to v1) is a one-file change.
BaseModel = pydantic.BaseModel
Field = pydantic.Field


def model_dump(model: Any) -> Dict[str, Any]:
    """Return a dict representation of a Pydantic model under v1 or v2.

    v2 exposes ``model.model_dump()``; v1 exposes ``model.dict()``. Callers
    can also pass a plain ``dict`` (returned unchanged) or any other object
    (converted via ``str``) — matching the defensive pattern used by
    ``BaseAdapter`` when serializing event payloads of unknown shape.
    """
    if isinstance(model, dict):
        return model
    if PYDANTIC_V2 and hasattr(model, "model_dump"):
        result = model.model_dump()
        if isinstance(result, dict):
            return result
        return {"value": result}
    if hasattr(model, "dict"):
        result = model.dict()
        if isinstance(result, dict):
            return result
        return {"value": result}
    return {"raw": str(model)}


# Cast pydantic to Any inside the shim so we can call differently-shaped
# v1 and v2 entry points without the type checker objecting to the dead
# branch under whichever version is currently installed.
_pyd: Any = pydantic


def field_validator(*fields: str, mode: str = "after") -> Callable[..., Any]:
    """Cross-version field validator decorator.

    Under Pydantic v2, delegates to the real ``field_validator``. Under
    v1, delegates to ``pydantic.validator`` translating
    ``mode="before"`` to ``pre=True`` and ``mode="after"`` to
    ``pre=False``.

    Usage::

        from layerlens._compat.pydantic import BaseModel, field_validator

        class M(BaseModel):
            x: int

            @field_validator("x")
            @classmethod
            def _check_x(cls, v: int) -> int:
                ...
    """
    if PYDANTIC_V2:
        result = _pyd.field_validator(*fields, mode=mode)
        return result  # type: ignore[no-any-return]

    pre = mode == "before"

    def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        decorated: Callable[..., Any] = _pyd.validator(
            *fields, pre=pre, allow_reuse=True
        )(fn)
        return decorated

    return _decorator


def model_validator(mode: str = "after") -> Callable[..., Any]:
    """Cross-version model validator decorator.

    Under Pydantic v2, delegates to the real ``model_validator``. Under
    v1, delegates to ``pydantic.root_validator`` with the appropriate
    ``pre`` kwarg.
    """
    if PYDANTIC_V2:
        result = _pyd.model_validator(mode=mode)
        return result  # type: ignore[no-any-return]

    pre = mode == "before"

    def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        decorated: Callable[..., Any] = _pyd.root_validator(
            pre=pre, allow_reuse=True
        )(fn)
        return decorated

    return _decorator


__all__ = [
    "BaseModel",
    "Field",
    "PYDANTIC_V2",
    "field_validator",
    "model_dump",
    "model_validator",
]
