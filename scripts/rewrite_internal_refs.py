#!/usr/bin/env python3
"""Second-pass rewriter: fix internal module references after file renames."""

import sys
from pathlib import Path

# SDK core modules renamed with _ prefix
REWRITES = [
    # SDK core files (within layerlens.instrument.)
    ("layerlens.instrument.core", "layerlens.instrument._core"),
    ("layerlens.instrument.context", "layerlens.instrument._context"),
    ("layerlens.instrument.decorators", "layerlens.instrument._decorators"),
    ("layerlens.instrument.emit", "layerlens.instrument._emit"),
    ("layerlens.instrument.cost", "layerlens.instrument._cost"),
    ("layerlens.instrument.enforcement", "layerlens.instrument._enforcement"),
    ("layerlens.instrument.state", "layerlens.instrument._state"),
    # Adapter base files
    ("layerlens.instrument.adapters.base", "layerlens.instrument.adapters._base"),
    ("layerlens.instrument.adapters.capture", "layerlens.instrument.adapters._capture"),
    ("layerlens.instrument.adapters.registry", "layerlens.instrument.adapters._registry"),
    ("layerlens.instrument.adapters.sinks", "layerlens.instrument.adapters._sinks"),
    ("layerlens.instrument.adapters.trace_container", "layerlens.instrument.adapters._trace_container"),
    ("layerlens.instrument.adapters.replay_models", "layerlens.instrument.adapters._replay_models"),
    # Exporter files
    ("layerlens.instrument.exporters.base", "layerlens.instrument.exporters._base"),
    ("layerlens.instrument.exporters.otel_metrics", "layerlens.instrument.exporters._otel_metrics"),
    ("layerlens.instrument.exporters.otel", "layerlens.instrument.exporters._otel"),
]

# Sort by length descending so longer matches come first (e.g., otel_metrics before otel)
REWRITES.sort(key=lambda x: len(x[0]), reverse=True)


def rewrite_file(path: Path) -> bool:
    text = path.read_text()
    original = text
    for old, new in REWRITES:
        # Only replace when followed by word boundary characters (import context)
        # Simple string replacement is fine since these are unique enough
        text = text.replace(old, new)
    if text != original:
        path.write_text(text)
        return True
    return False


def main() -> None:
    target = Path(sys.argv[1])
    changed = 0
    for py_file in sorted(target.rglob("*.py")):
        if rewrite_file(py_file):
            print(f"  Rewrote: {py_file}")
            changed += 1
    print(f"\nRewrote {changed} files (internal refs)")


if __name__ == "__main__":
    main()
