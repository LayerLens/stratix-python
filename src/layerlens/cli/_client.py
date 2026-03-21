from __future__ import annotations

import re
import sys
import functools
import traceback as tb
from typing import Any, Callable

import click

from .._client import Stratix
from .._exceptions import StratixError, NotFoundError, AuthenticationError


def get_client(ctx: click.Context) -> Stratix:
    """Create a Stratix client from CLI context options."""
    try:
        return Stratix(
            api_key=ctx.obj.get("api_key"),
            base_url=ctx.obj.get("base_url"),
        )
    except StratixError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def handle_errors(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator that catches SDK errors and prints user-friendly messages."""

    @functools.wraps(fn)
    @click.pass_context
    def wrapper(ctx: click.Context, *args: Any, **kwargs: Any) -> Any:
        try:
            return ctx.invoke(fn, *args, **kwargs)
        except AuthenticationError:
            click.echo("Error: Invalid or missing API key.", err=True)
            sys.exit(1)
        except NotFoundError as e:
            click.echo(f"Error: Resource not found. {e}", err=True)
            sys.exit(1)
        except StratixError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
        except click.exceptions.Exit:
            raise
        except Exception as e:
            if ctx.obj.get("verbose"):
                tb.print_exc()
            click.echo(f"Unexpected error: {e}", err=True)
            sys.exit(1)

    return wrapper


_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)


def _is_uuid(value: str) -> bool:
    return bool(_UUID_RE.match(value))


def resolve_model(client: Stratix, identifier: str) -> Any:
    """Resolve a model by ID, key, or name."""
    # Try by ID first if it looks like a UUID
    if _is_uuid(identifier):
        model = client.models.get_by_id(identifier)
        if model:
            return model

    # Try by key
    model = client.models.get_by_key(identifier)
    if model:
        return model

    # Try by name
    models = client.models.get(name=identifier)
    if models:
        return models[0]

    return None


def resolve_benchmark(client: Stratix, identifier: str) -> Any:
    """Resolve a benchmark by ID, key, or name."""
    # Try by ID first if it looks like a UUID
    if _is_uuid(identifier):
        benchmark = client.benchmarks.get_by_id(identifier)
        if benchmark:
            return benchmark

    # Try by key
    benchmark = client.benchmarks.get_by_key(identifier)
    if benchmark:
        return benchmark

    # Try by name
    benchmarks = client.benchmarks.get(name=identifier)
    if benchmarks:
        return benchmarks[0]

    return None
