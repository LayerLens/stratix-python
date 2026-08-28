"""Decoding a 2xx body into a model, loudly.

Most resources hand ``cast_to`` to the transport and let ``_base_client`` raise
``APIResponseValidationError`` when the body does not fit. The paginated list
resources cannot: they have to read the body, derive ``pagination`` from it, and
only then validate. That post-processing step is where LAY-3765 and LAY-2772 both
landed, because each of those resources wrapped its validation in
``except Exception: return None`` (or ``except (ValueError, KeyError)``) and
returned nothing.

That is worse than a crash. ``get_all()`` treated the resulting None as *end of
pages*, so a customer received a short list with no error, no warning, and no way
to tell it apart from a complete one. And because ``pydantic.ValidationError`` is
a subclass of ``ValueError``, the narrower ``except (ValueError, KeyError)`` guards
swallowed schema drift just as thoroughly as a bare ``except``.

These helpers exist so those resources can validate and still raise the SDK's own
``APIResponseValidationError`` — carrying the response, the offending payload, and
the pydantic field paths, which is what makes a schema drift reportable instead of
guesswork.
"""

from __future__ import annotations

from typing import Any, Dict, Type, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from ._exceptions import APIResponseValidationError

ModelT = TypeVar("ModelT", bound=BaseModel)

# Cap on how much of a body is quoted into an exception message. The body itself
# is attached to the exception in full via `.body`; this only bounds the string a
# traceback prints, so a 500-row page does not bury the field path that matters.
_MAX_QUOTED_BODY = 512


def describe_errors(err: ValidationError) -> str:
    """Render a ValidationError as a compact list of ``field.path (type)`` entries."""
    parts = []
    for detail in err.errors():
        location = ".".join(str(piece) for piece in detail.get("loc", ())) or "<root>"
        parts.append(f"{location} ({detail.get('type', 'invalid')})")

    return ", ".join(parts)


def json_object(response: httpx.Response, *, endpoint: str) -> Dict[str, Any]:
    """Decode a body that must be a JSON object.

    Raises ``APIResponseValidationError`` rather than returning None: a 2xx
    response whose body is not the documented object is a server contract
    violation, and silently reporting it as "no data" is how a truncated list
    becomes indistinguishable from a complete one.
    """
    try:
        payload = response.json()
    except ValueError as err:
        raise APIResponseValidationError(
            response,
            response.text[:_MAX_QUOTED_BODY],
            message=f"{endpoint} returned a 2xx response whose body is not valid JSON",
        ) from err

    if not isinstance(payload, dict):
        raise APIResponseValidationError(
            response,
            payload,
            message=(f"{endpoint} returned a 2xx response whose body is {type(payload).__name__}, not a JSON object"),
        )

    return payload


def parse_model(
    model: Type[ModelT],
    payload: Any,
    *,
    response: httpx.Response,
    endpoint: str,
    detail: str = "",
) -> ModelT:
    """Validate ``payload`` against ``model`` or raise with the field paths named.

    ``detail`` locates the failure for the caller — a page number, a row index —
    because "the response did not validate" is not actionable on a 100-row page.
    """
    try:
        return model.model_validate(payload)
    except ValidationError as err:
        where = f" at {detail}" if detail else ""
        raise APIResponseValidationError(
            response,
            payload,
            message=(
                f"{endpoint} returned data the SDK could not parse as "
                f"{model.__name__}{where}: {describe_errors(err)}. "
                "This usually means the API's response shape changed; please report it "
                "with this message and your layerlens version."
            ),
        ) from err
