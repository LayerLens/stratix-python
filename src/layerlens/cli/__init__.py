from __future__ import annotations

import sys


def main() -> None:
    # Handle --version before importing click so it works without the [cli] extra
    if len(sys.argv) > 1 and sys.argv[1] in ("--version", "-v"):
        from .._version import __version__

        print(f"layerlens {__version__}")  # noqa: T201
        sys.exit(0)

    try:
        import click  # noqa: F401
    except ImportError:
        print(  # noqa: T201
            "CLI dependencies not installed. Install them with:\n\n  pip install layerlens[cli]\n"
        )
        sys.exit(1)

    from ._app import cli

    cli()


__all__ = ["main"]
