from __future__ import annotations

import sys

from ._version import __version__


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in ("--version", "-v"):
        print(f"layerlens {__version__}")
        sys.exit(0)

    print(f"layerlens {__version__}")
    print("See https://layerlens.gitbook.io/stratix-python-sdk for documentation.")
    print("\nUsage:")
    print("  layerlens --version   Show version")
