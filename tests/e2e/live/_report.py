"""Render harness result rows into the L3 report (markdown file + terminal summary).

The markdown report is the artifact an engineer scans for the manual L3 pass: one
row per provider x variant with a deep-link to the trace in the LayerLens UI, so the
human step is "open the link, confirm the UI renders this already-verified trace
sensibly" rather than hunting through the dashboard.
"""

from __future__ import annotations

import os
import datetime as _dt
from typing import Any, Dict, List

_DEFAULT_APP_BASE = "https://app.layerlens.ai"


def _report_dir() -> str:
    return os.path.join(os.path.dirname(__file__), ".report")


def ui_link(row: Dict[str, Any]) -> str:
    """Best-effort deep-link to the trace in the LayerLens UI.

    The exact UI path is not yet confirmed, so this is templated from
    LAYERLENS_APP_BASE_URL; the raw trace_id/org/project are always in the report
    so the trace is locatable even if the path differs.
    """
    base = os.environ.get("LAYERLENS_APP_BASE_URL", _DEFAULT_APP_BASE).rstrip("/")
    trace_id = row.get("trace_id") or ""
    return f"{base}/traces/{trace_id}"


def write_markdown_report(rows: List[Dict[str, Any]]) -> str:
    os.makedirs(_report_dir(), exist_ok=True)
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = os.path.join(_report_dir(), f"live-run-{stamp}.md")

    lines: List[str] = []
    lines.append(f"# Live adapter verification — {stamp}")
    lines.append("")
    lines.append(f"{len(rows)} verified trace(s). Open each link and run the L3 checklist in README.md.")
    lines.append("")
    lines.append(
        "| Provider | Variant | Model | Events | Types | Tools | Cost (USD) | Redaction | Attestation | Data echo | Trace |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        types = ", ".join(f"{k}:{v}" for k, v in sorted(r.get("event_types", {}).items()))
        link = ui_link(r)
        trace_cell = f"[{r.get('trace_id', '')}]({link})"
        lines.append(
            "| {provider} | {variant} | {model} | {n} | {types} | {tools} | {cost} | {red} | {att} | {echo} | {trace} |".format(
                provider=r.get("provider", ""),
                variant=r.get("variant", ""),
                model=r.get("model") or "-",
                n=r.get("n_events", 0),
                types=types or "-",
                tools=r.get("tool_calls", 0),
                cost=f"{r.get('total_cost_usd', 0):.6f}",
                red="yes" if r.get("redaction_ok") else "-",
                att="ok" if r.get("attestation_ok") else "FAIL",
                echo="yes" if r.get("data_has_events") else "no",
                trace=trace_cell,
            )
        )
    lines.append("")
    lines.append(f"_org={rows[0].get('org_id')} project={rows[0].get('project_id')}_" if rows else "")
    lines.append("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path


def terminal_summary_lines(rows: List[Dict[str, Any]], path: str) -> List[str]:
    out: List[str] = []
    for r in rows:
        n_events = r.get("n_events")
        if n_events is None:
            n_events = r.get("event_count")  # self-flushing rows
        out.append(
            "  {name}/{variant}: {n} events, {tools} tool.call, ${cost:.6f}  ->  {link}".format(
                name=r.get("provider") or r.get("framework", ""),
                variant=r.get("variant", ""),
                n=n_events if n_events is not None else "?",
                tools=r.get("tool_calls", 0),
                cost=r.get("total_cost_usd", 0),
                link=ui_link(r),
            )
        )
    out.append(f"  report: {path}")
    return out
