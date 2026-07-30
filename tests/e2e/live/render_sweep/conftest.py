"""Report capture for the customer-run render sweep.

Accumulates one row per (stem, record) and writes a Markdown ``.report/`` at the
end of the session — the shipped-vs-flagged render evidence for Objective 2.
"""

from __future__ import annotations

import os
import json
import datetime
from typing import Any, Dict, List

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPORT_DIR = os.path.join(_HERE, ".report")

_ROWS: List[Dict[str, Any]] = []


@pytest.fixture
def render_report():
    """Return a callable that records one render-sweep row."""

    def _record(row: Dict[str, Any]) -> None:
        _ROWS.append(row)

    return _record


def _write_report() -> str:
    os.makedirs(_REPORT_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")
    by_kind: Dict[str, int] = {}
    n_flag = 0
    for r in _ROWS:
        by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + 1
        if r["problems"]:
            n_flag += 1
    stems = sorted({r["stem"] for r in _ROWS})
    lines = [
        "# Customer-run render sweep — report",
        "",
        f"Generated: {ts}",
        "",
        f"- stems swept: **{len(stems)}**",
        f"- records checked: **{len(_ROWS)}**",
        f"- render violations: **{n_flag}**",
        f"- kind breakdown (records): {json.dumps(by_kind)}",
        "",
        "| stem | rec | kind | server nodes | edges | agent | framework | status | sealed | render |",
        "|------|-----|------|--------------|-------|-------|-----------|--------|--------|--------|",
    ]
    for r in sorted(_ROWS, key=lambda x: (x["stem"], x["record"])):
        verdict = "OK" if not r["problems"] else ("FLAG: " + "; ".join(r["problems"]))
        nodes = ",".join(r["server_nodes"]) if r["server_nodes"] else "—"
        lines.append(
            f"| {r['stem']} | {r['record']} | {r['kind']} | {nodes} | {r['edges']} | "
            f"{r['agent'] or '—'} | {r['framework'] or '—'} | {r['status'] or '—'} | "
            f"{'yes' if r['sealed'] else ''} | {verdict} |"
        )
    path = os.path.join(_REPORT_DIR, f"render_sweep_{ts}.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    # Also a machine-readable copy.
    with open(os.path.join(_REPORT_DIR, f"render_sweep_{ts}.json"), "w") as f:
        json.dump({"generated": ts, "rows": _ROWS}, f, indent=2)
    return path


def pytest_sessionfinish(session: pytest.Session) -> None:
    if _ROWS:
        path = _write_report()
        tr = session.config.pluginmanager.get_plugin("terminalreporter")
        if tr is not None:
            tr.write_line(f"\n[render-sweep] wrote report: {path}")
