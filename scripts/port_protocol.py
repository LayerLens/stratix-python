#!/usr/bin/env python3
"""Port protocol adapters from ateam to stratix-python.

Handles both:
* Subdirectory protocols: ``a2a/``, ``agui/``, ``mcp/`` — like the
  framework script.
* Flat files: ``ap2.py``, ``a2ui.py``, ``ucp.py``, ``certification.py``,
  plus shared support files (``base.py``, ``exceptions.py``, etc.).

Mechanical transforms identical to scripts/port_adapter.py.
"""

from __future__ import annotations

import re
from pathlib import Path

ATEAM_ROOT = Path("A:/github/layerlens/ateam")
DEST_ROOT = Path("A:/github/layerlens/stratix-python")

SRC_BASE = ATEAM_ROOT / "stratix" / "sdk" / "python" / "adapters" / "protocols"
DST_BASE = DEST_ROOT / "src" / "layerlens" / "instrument" / "adapters" / "protocols"


def port_text(text: str) -> str:
    out = text
    out = out.replace(
        "from stratix.sdk.python.adapters.protocols.",
        "from layerlens.instrument.adapters.protocols.",
    )
    out = out.replace(
        "from stratix.sdk.python.adapters.base import",
        "from layerlens.instrument.adapters._base.adapter import",
    )
    out = out.replace(
        "from stratix.sdk.python.adapters.capture import",
        "from layerlens.instrument.adapters._base.capture import",
    )
    out = out.replace(
        "from stratix.sdk.python.adapters.trace_container import",
        "from layerlens.instrument.adapters._base.trace_container import",
    )
    # Catch-all for cross-adapter imports.
    out = out.replace(
        "from stratix.sdk.python.adapters.",
        "from layerlens.instrument.adapters.frameworks.",
    )
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
    out = out.replace("_stratix_original", "_layerlens_original")
    out = out.replace("Stratix adapter for", "LayerLens adapter for")
    out = out.replace("STRATIX adapter for", "LayerLens adapter for")
    return out


def port_subdirectory(name: str) -> int:
    """Port a subdirectory protocol (a2a, agui, mcp)."""
    src_dir = SRC_BASE / name
    dst_dir = DST_BASE / name
    if not src_dir.exists():
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for src_file in sorted(src_dir.glob("*.py")):
        text = src_file.read_text()
        (dst_dir / src_file.name).write_text(port_text(text))
        n += 1
    return n


def port_flat_file(name: str) -> int:
    """Port a flat file (ap2.py, a2ui.py, ucp.py, etc.)."""
    src_file = SRC_BASE / f"{name}.py"
    if not src_file.exists():
        return 0
    text = src_file.read_text()
    (DST_BASE / f"{name}.py").write_text(port_text(text))
    return 1


if __name__ == "__main__":
    DST_BASE.mkdir(parents=True, exist_ok=True)
    total = 0
    # Shared support files (top-level under protocols/).
    for flat in ["base", "exceptions", "health", "connection_pool"]:
        n = port_flat_file(flat)
        if n:
            print(f"Ported flat: {flat}.py")
            total += n
    # Single-file protocol adapters.
    for flat in ["ap2", "a2ui", "ucp", "certification"]:
        n = port_flat_file(flat)
        if n:
            print(f"Ported flat: {flat}.py")
            total += n
    # Subdirectory protocol adapters.
    for sub in ["a2a", "agui", "mcp"]:
        n = port_subdirectory(sub)
        if n:
            print(f"Ported {n} files: {sub}/")
            total += n
    print(f"Total files ported: {total}")
