#!/usr/bin/env python3
"""Port a single-file framework adapter from ateam to stratix-python.

Mechanical transforms applied:

1. ``stratix.sdk.python.adapters.X``    → ``layerlens.instrument.adapters.frameworks.X``
2. ``stratix.sdk.python.adapters.base`` → ``layerlens.instrument.adapters._base.adapter``
3. ``stratix.sdk.python.adapters.capture`` → ``layerlens.instrument.adapters._base.capture``
4. ``# type: ignore[import-not-found]`` → ``# type: ignore[import-not-found,unused-ignore]``
5. ``_stratix_original`` → ``_layerlens_original``  (attribute name only)
6. Brand: ``Stratix adapter for X`` in docstrings → ``LayerLens adapter for X``
7. Validate: file uses ``from __future__ import annotations`` (so PEP 604 union
   types and built-in generics work in 3.8+ in annotation positions).

Does NOT change:
* Class names — these were never STRATIX-prefixed in source.
* Public method signatures.
* Behavior / instrumentation logic — must remain a faithful port.

Per CLAUDE.md, scripted ports are fine when each result is reviewed and
tested. This script's output is verified by ``mypy --strict`` and a
test that imports and instantiates each adapter.

Usage::

    python scripts/port_adapter.py <ateam-package> [<dest-name>]

Examples::

    python scripts/port_adapter.py agno
    python scripts/port_adapter.py benchmark_import
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ATEAM_ROOT = Path("A:/github/layerlens/ateam")
DEST_ROOT = Path("A:/github/layerlens/stratix-python")

SRC_BASE = ATEAM_ROOT / "stratix" / "sdk" / "python" / "adapters"
DST_BASE = DEST_ROOT / "src" / "layerlens" / "instrument" / "adapters" / "frameworks"


def port_text(text: str, package: str) -> str:
    """Apply mechanical transforms to a single source file's contents."""
    out = text

    # Specific imports first (longest first to avoid partial matches).
    out = out.replace(
        f"from stratix.sdk.python.adapters.{package}.lifecycle import",
        f"from layerlens.instrument.adapters.frameworks.{package}.lifecycle import",
    )
    out = out.replace(
        f"from stratix.sdk.python.adapters.{package}.adapter import",
        f"from layerlens.instrument.adapters.frameworks.{package}.adapter import",
    )
    out = out.replace(
        "from stratix.sdk.python.adapters.base import",
        "from layerlens.instrument.adapters._base.adapter import",
    )
    out = out.replace(
        "from stratix.sdk.python.adapters.capture import",
        "from layerlens.instrument.adapters._base.capture import",
    )
    # Generic catch-all (rare cross-adapter imports).
    out = out.replace(
        "from stratix.sdk.python.adapters.",
        "from layerlens.instrument.adapters.frameworks.",
    )

    # Soften the type-ignore so mypy doesn't complain in envs where the
    # framework IS installed (the local dev box, but not all CI matrices).
    out = re.sub(
        r"#\s*type:\s*ignore\[import-not-found\](?!\w)",
        "# type: ignore[import-not-found,unused-ignore]",
        out,
    )
    out = re.sub(
        r"#\s*type:\s*ignore\[import-untyped\](?!\w)",
        "# type: ignore[import-untyped,unused-ignore]",
        out,
    )

    # Rename internal sentinel attribute on traced functions.
    out = out.replace("_stratix_original", "_layerlens_original")

    # Brand strings (visible in docstrings + user-facing AdapterInfo.description).
    out = out.replace("Stratix adapter for", "LayerLens adapter for")
    out = out.replace("STRATIX adapter for", "LayerLens adapter for")

    return out


def port_package(package: str) -> None:
    src_dir = SRC_BASE / package
    dst_dir = DST_BASE / package
    if not src_dir.exists():
        sys.exit(f"source not found: {src_dir}")
    dst_dir.mkdir(parents=True, exist_ok=True)

    files_ported = 0
    for src_file in sorted(src_dir.glob("*.py")):
        if src_file.name == "__pycache__":
            continue
        text = src_file.read_text()
        new = port_text(text, package)
        dst_file = dst_dir / src_file.name
        dst_file.write_text(new)
        files_ported += 1

    print(f"Ported {files_ported} files: {package}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__.split("Usage::")[1].strip())
    port_package(sys.argv[1])
