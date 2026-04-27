"""Vendored snapshots of types from the ateam ``stratix`` package.

These modules are deliberately *frozen* copies of select types from the
``stratix`` package (see ``A:/github/layerlens/ateam``) so that the
LayerLens instrumentation layer can reference them without taking a
runtime dependency on ateam.

Each module records the source SHA at the top. To refresh a vendored
module:

1. Re-copy the file from
   ``A:/github/layerlens/ateam/stratix/<path>``.
2. Apply the Python 3.9 / Pydantic 2 compatibility shims described in
   the comment header of each file.
3. Update the ``Source SHA`` line.
4. Re-run ``pytest tests/instrument`` and ``mypy --strict
   src/layerlens/instrument/_vendored/``.

Do **not** modify these files to add new fields — vendored types must
match ateam's wire shape exactly. New behavior belongs in the adapters
that consume them.
"""

from __future__ import annotations

__all__: list[str] = []
