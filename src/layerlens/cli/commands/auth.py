"""Authentication commands: login, logout, whoami."""

from __future__ import annotations

import sys

import click


@click.command()
@click.pass_context
def login(ctx: click.Context) -> None:
    """Authenticate with LayerLens using email and password."""
    from .._auth import (
        LoginError,
        cli_login,
        load_credentials,
    )

    existing = load_credentials()
    if existing and existing.get("access_token"):
        if not click.confirm("You are already logged in. Re-authenticate?", default=False):
            return

    base_url = ctx.obj.get("base_url") if ctx.obj else None

    email = click.prompt("  Email")
    password = click.prompt("  Password", hide_input=True)

    try:
        creds = cli_login(email, password, base_url=base_url)
        user = creds.get("user") or {}
        name = user.get("given_name") or user.get("name") or email
        click.echo(f"\n  Logged in as {name}", err=True)
    except LoginError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)


@click.command()
def logout() -> None:
    """Clear stored authentication credentials."""
    from .._auth import load_credentials, clear_credentials

    if load_credentials() is None:
        click.echo("Not currently logged in.", err=True)
        return

    clear_credentials()
    click.echo("Logged out successfully.", err=True)


@click.command()
def whoami() -> None:
    """Display the currently authenticated user."""
    import os

    from .._auth import (
        get_user_info,
        get_valid_token,
        load_credentials,
    )

    # Check env-var API key first
    env_key = os.environ.get("LAYERLENS_API_KEY")
    if env_key:
        click.echo("Authenticated via LAYERLENS_API_KEY environment variable.", err=True)
        return

    creds = load_credentials()
    if creds is None:
        click.echo("Not logged in. Run `layerlens login` to authenticate.", err=True)
        sys.exit(1)

    token = get_valid_token()
    if token is None:
        click.echo("Session expired. Run `layerlens login` to re-authenticate.", err=True)
        sys.exit(1)

    info = get_user_info(token)
    if info is None:
        click.echo("Authenticated (could not fetch user details).", err=True)
        return

    email = info.get("email", "unknown")
    name = info.get("name") or info.get("given_name", "")
    sub = info.get("sub", "")

    click.echo(f"  Email: {email}")
    if name:
        click.echo(f"  Name:  {name}")
    if sub:
        click.echo(f"  ID:    {sub}")
