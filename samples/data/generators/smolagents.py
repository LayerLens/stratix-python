"""ADP-W2 Family-B recorder for the ``smolagents`` adapter (record-real-once).

Records TWO real, fully-instrumented HuggingFace ``smolagents`` runs and writes
each as a sealed real-trace fixture under ``samples/data/traces/industry/``:

* ``generate_smolagents_single`` -> ``media_smolagents_newsroom.jsonl``: a single
  newsroom ``newsroom_research_agent`` (a ``ToolCallingAgent``) that runs a real
  two-tool research loop — ``search_news`` then ``read_article`` — and writes a
  short story brief. Renders a single honest agent node (Agent column =
  ``newsroom_research_agent``) with the real ``model.invoke`` / ``cost.record`` /
  ``tool.call`` / ``tool.result`` events of the loop.

* ``generate_smolagents_multi`` -> ``media_smolagents_research_team.jsonl``: a
  genuine multi-agent run — a ``newsroom_editor`` manager delegates to two named
  managed sub-agents, ``research_agent`` (which runs the ``search_news`` /
  ``read_article`` tools) and ``story_writer`` (which writes the brief), via
  smolagents' built-in managed-agent delegation. The adapter recursively
  instruments the sub-agents, so the trace carries a real ``agent.handoff`` per
  delegation (newsroom_editor -> research_agent, newsroom_editor -> story_writer)
  and three distinct honest agent identities — it renders as a multi-agent DAG
  (>=2 agent nodes + handoff edges, Agent column ``multi-agent``).

Both are recorded through the REAL ``SmolAgentsAdapter`` (step-callbacks +
``agent.run`` wrapper): the adapter builds a per-run collector and flushes it on
run-end, and the flush is observed via the ``_generate_fixtures`` capture seam
(``set_trace_observer`` + a no-op ``enqueue_upload``) so the sealed payload —
real per-step ``model.invoke``/``cost.record`` (gpt-4o-mini pricing is applied,
so ``cost.record.cost_usd`` is a real non-None figure) + intact attestation
chain — is captured but never uploaded during generation. The samples upload the
captured fixtures themselves at run time.

Nothing is fabricated: the Framework column shows ``smolagents`` (the framework
that really ran), the token/cost fields are real, and the multi-agent nodes and
handoff edges are the real named sub-agents / delegations the framework emitted.
smolagents requires agent names to be valid Python identifiers (they are exposed
to the manager as callable tools), so the honest agent names use underscores.

The recorded ``model.invoke`` events carry per-step token totals via
``token_usage`` (non-streaming step callbacks); smolagents has no token-stream/
delta path, so this is the only — and non-lossy — path.
"""

from __future__ import annotations

import json
import os
import sys

# Reuse the record-real-once seam from the sibling ``_generate_fixtures`` module
# (single source of truth for the fixture writer + capture config + model name).
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.dirname(_HERE)        # samples/data
_SAMPLES = os.path.dirname(_DATA)     # samples
_REPO = os.path.dirname(_SAMPLES)
for _p in (os.path.join(_REPO, "src"), _SAMPLES, _DATA):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# This module is named ``smolagents.py`` (to match the adapter). When the file is
# run directly, Python inserts its own directory at ``sys.path[0]``, which would
# shadow the real ``smolagents`` package for the function-local ``import
# smolagents``. Drop this module's own directory from the path so the framework
# import always resolves to the installed package (a no-op when imported as
# ``generators.smolagents``, since the package dir is not on the path then).
sys.path[:] = [_p for _p in sys.path if os.path.abspath(_p or ".") != _HERE]

from layerlens import Stratix  # noqa: E402
import layerlens.instrument._collector as _collector_mod  # noqa: E402
from layerlens.instrument._collector import set_trace_observer  # noqa: E402

import _generate_fixtures as _gf  # noqa: E402

_write = _gf._write
_CAPTURE = _gf._CAPTURE
OPENAI_MODEL = _gf.OPENAI_MODEL


# --------------------------------------------------------------------------
# A small, realistic (non-sensitive) local-news wire the research tools read
# from, so the ``search_news`` / ``read_article`` tool loop returns genuine
# grounded content for the story brief.
# --------------------------------------------------------------------------
_NEWS_WIRE = {
    "TR-101": {
        "id": "TR-101",
        "headline": "City council approves $40M transit expansion",
        "source": "Metro Wire",
        "date": "2026-07-12",
        "body": (
            "The city council voted 7-2 on Tuesday to fund a $40 million transit "
            "expansion. The plan adds a new light-rail line and 12 electric buses, "
            "and is projected to cut average cross-town commute times by 18%. "
            "Construction is slated to begin in spring 2026 with full service by 2027."
        ),
    },
    "TR-102": {
        "id": "TR-102",
        "headline": "Transit advocates praise plan, cite equity gains",
        "source": "Metro Wire",
        "date": "2026-07-12",
        "body": (
            "Transit-equity groups welcomed the expansion, noting the new line "
            "connects three historically underserved neighborhoods to the downtown "
            "job core. A coalition spokesperson said reliable service could raise "
            "job access for roughly 60,000 residents while the electric fleet cuts "
            "corridor emissions."
        ),
    },
    "TR-103": {
        "id": "TR-103",
        "headline": "Some residents question the transit budget timeline",
        "source": "Metro Wire",
        "date": "2026-07-13",
        "body": (
            "A minority of council members and residents raised concerns about the "
            "expansion's funding timeline, warning that the 2027 service target "
            "depends on a state matching grant that has not yet been finalized."
        ),
    },
}


def _search_news_impl(query: str) -> str:
    """Return matching headlines (id + headline + source) from the local wire."""
    q = (query or "").lower()
    hits = [
        {"id": r["id"], "headline": r["headline"], "source": r["source"], "date": r["date"]}
        for r in _NEWS_WIRE.values()
        if any(tok in r["headline"].lower() or tok in r["body"].lower()
               for tok in ("transit", "light-rail", "bus", "expansion") if tok in q)
    ]
    if not hits:  # keyword fallback so the loop always has grounded results
        hits = [
            {"id": r["id"], "headline": r["headline"], "source": r["source"], "date": r["date"]}
            for r in _NEWS_WIRE.values()
        ]
    return json.dumps(hits)


def _read_article_impl(article_id: str) -> str:
    """Return the full article body for a wire id (from ``search_news``)."""
    rec = _NEWS_WIRE.get((article_id or "").strip().upper())
    if rec is None:
        return json.dumps({"id": article_id, "found": False, "message": "No such article."})
    return json.dumps(rec)


# --------------------------------------------------------------------------
# Adapter-driven capture: SmolAgentsAdapter wraps ``agent.run`` and flushes its
# own collector on run-end; we register it, drive a REAL ``agent.run``, and
# observe the flushed payload (self-flushing adapter, like crewai/autogen).
# --------------------------------------------------------------------------
def _capture_smolagents(client: Stratix, agent, task: str) -> dict:
    from layerlens.instrument.adapters.frameworks.smolagents import SmolAgentsAdapter

    captured: dict = {}
    set_trace_observer(lambda p: captured.setdefault("payload", p))
    orig = _collector_mod.enqueue_upload
    _collector_mod.enqueue_upload = lambda *a, **k: None
    adapter = SmolAgentsAdapter(client, capture_config=_CAPTURE)
    adapter.connect(target=agent)
    try:
        agent.run(task)
    finally:
        try:
            adapter.disconnect()
        except Exception:
            pass
        set_trace_observer(None)
        _collector_mod.enqueue_upload = orig
    payload = captured.get("payload")
    if not payload:
        raise RuntimeError("no payload captured for smolagents run")
    return payload


# --------------------------------------------------------------------------
# Single agent + a real two-tool research loop (media newsroom research)
# --------------------------------------------------------------------------
def generate_smolagents_single(client: Stratix) -> dict:
    """Record a single ``newsroom_research_agent`` running a real search/read loop."""
    from smolagents import ToolCallingAgent, OpenAIServerModel, tool

    @tool
    def search_news(query: str) -> str:
        """Search the local-news wire for a topic and return matching headlines.

        Args:
            query: The topic to search for (e.g. "transit expansion").
        """
        return _search_news_impl(query)

    @tool
    def read_article(article_id: str) -> str:
        """Fetch the full text of a wire article by its id (from search_news).

        Args:
            article_id: The wire id returned by search_news (e.g. "TR-101").
        """
        return _read_article_impl(article_id)

    model = OpenAIServerModel(model_id=OPENAI_MODEL)
    agent = ToolCallingAgent(
        tools=[search_news, read_article],
        model=model,
        max_steps=4,
        name="newsroom_research_agent",
        description=(
            "A newsroom research assistant that searches the wire and reads the "
            "relevant article, then drafts a short, sourced story brief."
        ),
    )

    task = (
        "Research the city transit-expansion story. First use search_news to find "
        "the relevant headlines, then use read_article to read the most relevant "
        "article, and finally write a factual 2-3 sentence news brief citing the "
        "key figures (funding amount and service year)."
    )
    payload = _capture_smolagents(client, agent, task)
    payload["tags"] = [
        "layerlens-sample",
        "industry",
        "media",
        "newsroom-research",
        "tool-use",
    ]
    events = payload.get("events", [])
    tools = sorted(
        {(e.get("payload") or {}).get("tool_name") for e in events
         if e.get("event_type") == "tool.call"}
        - {None}
    )
    mi = [e for e in events if e.get("event_type") == "model.invoke"]
    cr = [e for e in events if e.get("event_type") == "cost.record"]
    print("  smolagents single (newsroom_research_agent, tool-use)  "
          "events=%d tools=%s model.invoke=%d cost.record=%d"
          % (len(events), tools, len(mi), len(cr)))
    print("  ->", _write([payload], "industry", "media_smolagents_newsroom"), "\n")
    return payload


# --------------------------------------------------------------------------
# Multi-agent: manager delegates to two named managed sub-agents (newsroom team)
# --------------------------------------------------------------------------
def generate_smolagents_multi(client: Stratix) -> dict:
    """Record a genuine multi-agent newsroom team: a ``newsroom_editor`` manager
    delegates research to ``research_agent`` (real tools) and writing to
    ``story_writer`` via smolagents managed-agent delegation (real handoffs)."""
    from smolagents import ToolCallingAgent, OpenAIServerModel, tool

    @tool
    def search_news(query: str) -> str:
        """Search the local-news wire for a topic and return matching headlines.

        Args:
            query: The topic to search for (e.g. "transit expansion").
        """
        return _search_news_impl(query)

    @tool
    def read_article(article_id: str) -> str:
        """Fetch the full text of a wire article by its id (from search_news).

        Args:
            article_id: The wire id returned by search_news (e.g. "TR-101").
        """
        return _read_article_impl(article_id)

    model = OpenAIServerModel(model_id=OPENAI_MODEL)
    research_agent = ToolCallingAgent(
        tools=[search_news, read_article],
        model=model,
        max_steps=4,
        name="research_agent",
        description=(
            "Researches a news topic: runs search_news then read_article and "
            "returns the key sourced facts (figures, dates, sources)."
        ),
    )
    story_writer = ToolCallingAgent(
        tools=[],
        model=model,
        max_steps=2,
        name="story_writer",
        description=(
            "Writes a concise, publishable 2-3 sentence news brief from the facts "
            "provided to it."
        ),
    )
    newsroom_editor = ToolCallingAgent(
        tools=[],
        model=model,
        max_steps=6,
        managed_agents=[research_agent, story_writer],
        name="newsroom_editor",
        description=(
            "The newsroom editor: assigns research to research_agent, then the "
            "write-up to story_writer, and returns the final brief."
        ),
    )

    task = (
        "Produce a short, sourced news brief on the city transit-expansion story. "
        "First delegate to research_agent to gather the key facts (funding amount, "
        "service year, and who benefits). Then delegate to story_writer to write "
        "the final 2-3 sentence brief from those facts. Return the finished brief."
    )
    payload = _capture_smolagents(client, newsroom_editor, task)
    payload["tags"] = [
        "layerlens-sample",
        "industry",
        "media",
        "newsroom-research",
        "multi-agent",
    ]
    events = payload.get("events", [])
    idents = sorted(
        {(e.get("payload") or {}).get("agent_name") for e in events
         if (e.get("payload") or {}).get("agent_name")}
        - {None}
    )
    handoffs = [
        (
            (e.get("payload") or {}).get("from_agent"),
            (e.get("payload") or {}).get("to_agent"),
        )
        for e in events
        if e.get("event_type") == "agent.handoff"
    ]
    mi = [e for e in events if e.get("event_type") == "model.invoke"]
    cr = [e for e in events if e.get("event_type") == "cost.record"]
    print("  smolagents multi (newsroom_editor -> research_agent/story_writer)  "
          "events=%d agents=%s handoffs=%s model.invoke=%d cost.record=%d"
          % (len(events), idents, handoffs, len(mi), len(cr)))
    print("  ->", _write([payload], "industry", "media_smolagents_research_team"), "\n")
    return payload


if __name__ == "__main__":  # pragma: no cover - manual regeneration entrypoint
    _client = Stratix()
    generate_smolagents_single(_client)
    generate_smolagents_multi(_client)
