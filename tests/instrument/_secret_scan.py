"""Runtime secret guard for uploaded telemetry (credential-sprawl net).

Imports the production scrubber's pattern set (so the CI guard and the prod
redactor can never drift) and asserts no secret-shaped value reaches any
uploaded event. Wired into the autouse ``_enforce_schema_lock`` fixture so it
runs over every adapter unit suite with zero per-test wiring. Pairs with
``layerlens.instrument._secret_scrub.safe_error`` (the production fix).
"""

from __future__ import annotations

from typing import Any, Dict, List

from layerlens.instrument._secret_scrub import find_secrets


def _walk_strings(value: Any, path: str, out: List[tuple[str, str]]) -> None:
    if isinstance(value, str):
        for name in find_secrets(value):
            out.append((path, name))
    elif isinstance(value, dict):
        for k, v in value.items():
            _walk_strings(v, f"{path}.{k}", out)
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            _walk_strings(v, f"{path}[{i}]", out)


def scan_for_secrets(events: List[Dict[str, Any]]) -> None:
    """Assert no uploaded event carries a secret-shaped value.

    Secrets leak independently of ``capture_content`` (a provider exception that
    echoes an API key is uploaded under the default config), so this guard runs
    on every recorded trace regardless of redaction settings.
    """
    hits: List[str] = []
    for event in events:
        found: List[tuple[str, str]] = []
        _walk_strings(event.get("payload") or {}, event.get("event_type", "?"), found)
        for path, name in found:
            hits.append(f"{path} matched secret pattern {name!r}")
    if hits:
        raise AssertionError(
            "secret-shaped value(s) reached an uploaded event (scrub at the emit site via "
            "layerlens.instrument._secret_scrub.safe_error):\n  " + "\n  ".join(hits)
        )
