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


def _app_base() -> str:
    """The UI origin that actually serves the API this run uploaded to.

    Defaulting to production is wrong for the local docker stack, which is where
    these lanes normally run: every row then deep-links to app.layerlens.ai for a
    trace that only exists in local Mongo — a dead link that also reads as though
    a local run had published to prod. Follow the upload target unless told
    otherwise (the local FE is served natively on :3000).
    """
    explicit = os.environ.get("LAYERLENS_APP_BASE_URL")
    if explicit:
        return explicit.rstrip("/")
    api = os.environ.get("LAYERLENS_STRATIX_BASE_URL", "")
    if "localhost" in api or "127.0.0.1" in api:
        return "http://localhost:3000"
    return _DEFAULT_APP_BASE


def ui_link(row: Dict[str, Any]) -> str:
    """Best-effort deep-link to the trace in the LayerLens UI.

    The exact UI path is not yet confirmed, so this is templated from the app
    base; the raw trace_id/org/project are always in the report so the trace is
    locatable even if the path differs.
    """
    base = _app_base()
    trace_id = row.get("trace_id") or ""
    return f"{base}/traces/{trace_id}"



def _cost_cell(row: Dict[str, Any]) -> str:
    """Render spend, distinguishing "no price is knowable" from "it was free".

    ``total_cost_usd`` is None when the trace carries no priced cost.record —
    an unpriced local model, or an adapter (marvin) that surfaces no usage at
    all. Rendering that as "$0.000000" claims the run was free and hides real
    untracked spend behind the same string a genuinely free ollama lane prints.
    """
    cost = row.get("total_cost_usd")
    if cost is None:
        return "n/a"
    return f"{cost:.6f}"


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
                # Framework rows key this as ``framework`` (the terminal summary
                # already falls back; the table did not, so every framework lane
                # rendered a nameless row).
                provider=r.get("provider") or r.get("framework", ""),
                variant=r.get("variant", ""),
                model=r.get("model") or "-",
                n=r.get("n_events", r.get("event_count")) or 0,
                types=types or "-",
                tools=r.get("tool_calls", 0),
                cost=_cost_cell(r),
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
            "  {name}/{variant}: {n} events, {tools} tool.call, {cost}  ->  {link}".format(
                name=r.get("provider") or r.get("framework", ""),
                variant=r.get("variant", ""),
                n=n_events if n_events is not None else "?",
                tools=r.get("tool_calls", 0),
                cost=("cost n/a" if r.get("total_cost_usd") is None else "$%.6f" % r["total_cost_usd"]),
                link=ui_link(r),
            )
        )
    out.append(f"  report: {path}")
    return out
