from __future__ import annotations

from typing import Any, List

import click


def _get_client_silent(ctx: click.Context) -> Any:
    """Try to create a Stratix client for autocompletion, returning None on failure."""
    try:
        from .._client import Stratix

        api_key = ctx.params.get("api_key") or None
        base_url = None
        host = ctx.params.get("host")
        port = ctx.params.get("port")
        if host:
            scheme = "https" if port in (None, 443) else "http"
            if port and port not in (80, 443):
                base_url = f"{scheme}://{host}:{port}/api/v1"
            else:
                base_url = f"{scheme}://{host}/api/v1"

        return Stratix(api_key=api_key, base_url=base_url)
    except Exception:
        return None


def complete_trace(
    ctx: click.Context, _param: click.Parameter, incomplete: str
) -> List[click.shell_completion.CompletionItem]:
    """Autocomplete trace IDs."""
    client = _get_client_silent(ctx)
    if not client:
        return []
    try:
        resp = client.traces.get_many(search=incomplete if incomplete else None, page_size=20)
        if resp and resp.traces:
            return [
                click.shell_completion.CompletionItem(t.id, help=t.filename)
                for t in resp.traces
                if t.id.startswith(incomplete)
            ]
    except Exception:
        pass
    return []


def complete_judge(
    ctx: click.Context, _param: click.Parameter, incomplete: str
) -> List[click.shell_completion.CompletionItem]:
    """Autocomplete judge IDs."""
    client = _get_client_silent(ctx)
    if not client:
        return []
    try:
        resp = client.judges.get_many(page_size=50)
        if resp and resp.judges:
            return [
                click.shell_completion.CompletionItem(j.id, help=j.name)
                for j in resp.judges
                if j.id.startswith(incomplete) or j.name.lower().startswith(incomplete.lower())
            ]
    except Exception:
        pass
    return []


def complete_model(
    ctx: click.Context, _param: click.Parameter, incomplete: str
) -> List[click.shell_completion.CompletionItem]:
    """Autocomplete model IDs, keys, and names."""
    client = _get_client_silent(ctx)
    if not client:
        return []
    try:
        models = client.models.get()
        if models:
            items = []
            for m in models:
                if (
                    m.id.startswith(incomplete)
                    or m.key.lower().startswith(incomplete.lower())
                    or m.name.lower().startswith(incomplete.lower())
                ):
                    items.append(click.shell_completion.CompletionItem(m.key, help=m.name))
            return items
    except Exception:
        pass
    return []


def complete_benchmark(
    ctx: click.Context, _param: click.Parameter, incomplete: str
) -> List[click.shell_completion.CompletionItem]:
    """Autocomplete benchmark IDs, keys, and names."""
    client = _get_client_silent(ctx)
    if not client:
        return []
    try:
        benchmarks = client.benchmarks.get()
        if benchmarks:
            items = []
            for b in benchmarks:
                if (
                    b.id.startswith(incomplete)
                    or b.key.lower().startswith(incomplete.lower())
                    or b.name.lower().startswith(incomplete.lower())
                ):
                    items.append(click.shell_completion.CompletionItem(b.key, help=b.name))
            return items
    except Exception:
        pass
    return []


def complete_evaluation(
    ctx: click.Context, _param: click.Parameter, incomplete: str
) -> List[click.shell_completion.CompletionItem]:
    """Autocomplete evaluation IDs."""
    client = _get_client_silent(ctx)
    if not client:
        return []
    try:
        resp = client.evaluations.get_many(page_size=20)
        if resp and resp.evaluations:
            return [
                click.shell_completion.CompletionItem(
                    e.id,
                    help=f"{getattr(e, 'model_name', '?')} / {getattr(e, 'benchmark_name', '?')}",
                )
                for e in resp.evaluations
                if e.id.startswith(incomplete)
            ]
    except Exception:
        pass
    return []


def complete_integration(
    ctx: click.Context, _param: click.Parameter, incomplete: str
) -> List[click.shell_completion.CompletionItem]:
    """Autocomplete integration IDs."""
    client = _get_client_silent(ctx)
    if not client:
        return []
    try:
        resp = client.integrations.get_many(page_size=50)
        if resp and resp.integrations:
            return [
                click.shell_completion.CompletionItem(i.id, help=i.name)
                for i in resp.integrations
                if i.id.startswith(incomplete) or i.name.lower().startswith(incomplete.lower())
            ]
    except Exception:
        pass
    return []
