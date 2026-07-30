"""Per-protocol registry. Reuses ``FrameworkCase``. Protocols are LLM-free,
but they DO carry content (message text, tool args, payment details), so each
content-emitting protocol also runs the ``redaction`` variant — the scenarios
embed ``SENTINEL`` in their content fields and the harness asserts it never
reaches the trace payload under ``capture_content=False`` (LAY-3578 / N7).
"""

from __future__ import annotations

from typing import Tuple

from . import _protocol_scenarios as ps
from ._framework_registry import FrameworkCase

# Built-in protocols (agui/a2ui/ap2/ucp) need no external package -> import_name
# "layerlens" always resolves. mcp/a2a may require their optional package; the
# test importorskips it (and they are exercised in a Python 3.10+ venv).
PROTOCOLS: Tuple[FrameworkCase, ...] = (
    FrameworkCase(
        id="agui", import_name="layerlens", runner=ps.run_agui, supports_redaction=True, install_hint="built-in"
    ),
    FrameworkCase(
        # a2ui emits only ids/counts and a KEYED HMAC of the action context
        # (per-instance random key, never emitted) — no cleartext content to
        # redact, and the digest is not reversible (P3 fix, LAY-3578).
        id="a2ui",
        import_name="layerlens",
        runner=ps.run_a2ui,
        supports_redaction=False,
        install_hint="built-in",
    ),
    FrameworkCase(
        id="ap2", import_name="layerlens", runner=ps.run_ap2, supports_redaction=True, install_hint="built-in"
    ),
    FrameworkCase(
        id="ucp", import_name="layerlens", runner=ps.run_ucp, supports_redaction=True, install_hint="built-in"
    ),
    FrameworkCase(
        id="mcp",
        import_name="mcp",
        runner=ps.run_mcp,
        supports_redaction=True,
        install_hint="layerlens[mcp] (py>=3.10)",
    ),
    FrameworkCase(
        id="a2a",
        import_name="a2a",
        runner=ps.run_a2a,
        supports_redaction=True,
        install_hint="layerlens[a2a] (py>=3.10)",
    ),
)
