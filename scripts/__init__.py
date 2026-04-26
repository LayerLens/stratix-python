"""Repository-level CLI scripts.

Marked as a package so the tests in
``tests/instrument/adapters/test_manifest_consistency.py`` can import
``scripts.emit_adapter_manifest`` directly without ``mypy .`` flagging
the file as both a top-level module (when scanned via the directory
walk) and a package member (when imported by the test). The package
contains no runtime exports — every module here is intended to be run
as ``python scripts/<name>.py``.
"""

from __future__ import annotations
