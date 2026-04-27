"""Compatibility shims for Python and library version differences.

The instrument layer must run on Python 3.8+ and Pydantic 1.9+ or 2.x.
Modules in this package centralize the conditional imports and polyfills
so adapter code can be written against a single, stable surface.
"""

from __future__ import annotations
