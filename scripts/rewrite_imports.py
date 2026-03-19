#!/usr/bin/env python3
"""Rewrite stratix.* imports to layerlens.instrument.* in all .py files under a directory."""

import re
import sys
from pathlib import Path

# Ordered most-specific first so longer prefixes match before shorter ones
REWRITES = [
    ("stratix.core.events.", "layerlens.instrument.schema.events."),
    ("stratix.core.", "layerlens.instrument.schema."),
    ("stratix.sdk.python.adapters.", "layerlens.instrument.adapters."),
    ("stratix.sdk.python.exporters.", "layerlens.instrument.exporters."),
    ("stratix.sdk.python.simulators.", "layerlens.instrument.simulators."),
    ("stratix.sdk.python.", "layerlens.instrument."),
]


def rewrite_file(path: Path) -> bool:
    """Rewrite imports in a single file. Returns True if changes were made."""
    text = path.read_text()
    original = text
    for old, new in REWRITES:
        text = text.replace(old, new)
    if text != original:
        path.write_text(text)
        return True
    return False


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <directory> [--dry-run]")
        sys.exit(1)

    target = Path(sys.argv[1])
    dry_run = "--dry-run" in sys.argv

    if not target.is_dir():
        print(f"Error: {target} is not a directory")
        sys.exit(1)

    changed = 0
    for py_file in sorted(target.rglob("*.py")):
        if dry_run:
            text = py_file.read_text()
            for old, new in REWRITES:
                if old in text:
                    print(f"  Would rewrite: {py_file}")
                    changed += 1
                    break
        else:
            if rewrite_file(py_file):
                print(f"  Rewrote: {py_file}")
                changed += 1

    print(f"\n{'Would rewrite' if dry_run else 'Rewrote'} {changed} files")


if __name__ == "__main__":
    main()
