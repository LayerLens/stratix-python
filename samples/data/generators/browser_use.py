"""ADP-PORT Family-B recorder for the ``browser_use`` adapter (record-real-once).

Records ONE real, fully-instrumented ``browser-use`` run and seals BOTH shipped
artifacts from that single run:

* ``samples/data/traces/industry/travel_browseruse_research.jsonl`` — the
  Family-B sample fixture. A corporate travel-desk ``trip-research-agent`` (a
  real ``browser_use.Agent`` backed by ``ChatOpenAI``) DRIVES A REAL HEADLESS
  CHROMIUM over CDP to read the travel desk's Trip Options Board and pick the
  cheapest nonstop Boston->Lisbon package under the client's $1,400 cap that
  keeps free cancellation. It renders a single honest agent node (Agent column
  = ``trip-research-agent``) with the real per-action ``tool.call`` events the
  browser really executed, the real history-level ``model.invoke`` /
  ``cost.record``, and the real ``environment.config`` snapshot.

  NOTE: the action PATH is the model's own choice and is not deterministic — a
  re-record may reach the board by clicking rather than extracting, and may take
  a different number of steps/tokens. That is expected of a real run. The sealed
  fixture and the recorded-corpus test are regenerated together from ONE run, so
  the test's real token/action constants must be refreshed whenever this is
  re-run (a stale constant correctly turns the corpus test red).

* ``tests/fixtures/recorded/browser_use/travel_research.json`` — the LAY-3614
  recorded-corpus fixture: the SAME run's real ``AgentHistoryList``, serialized
  with browser-use's OWN ``model_dump()`` (plus its ``UsageSummary``, which that
  custom dump drops). This is an ``object``-transport fixture — the adapter's
  real INPUT, recorded UPSTREAM of the adapter's history walk, which is the
  parser under test. ``tests/instrument/adapters/frameworks/test_browser_use_recorded.py``
  re-materializes it through browser-use's own ``AgentHistoryList.load_from_dict``
  against the real action registry and replays it through the real adapter — so
  a browser-use rename of a history/usage field fails CI, with no browser, no
  network, and no spend.

WHAT IS REAL HERE (and what is local) — read this before trusting the fixture
-----------------------------------------------------------------------------
Everything in the recorded trace is genuine:

* a REAL headless Chromium (the playwright-cached ``chromium-1217`` build)
  is launched and driven over REAL CDP by browser-use itself;
* the page is fetched over a REAL HTTP request and parsed into a REAL DOM,
  which browser-use serializes and feeds to the model;
* the model is a REAL ``gpt-4o-mini`` call — every action the agent takes is
  the model's own decision, and the token counts are the real
  ``AgentHistoryList.usage`` figures browser-use's token service recorded;
* the final answer is the model's real reading of the real rendered page.

The one deliberately-local part is the PAGE ITSELF: the Trip Options Board is
served by a throwaway ``http.server`` bound to loopback for the duration of the
run. That is a real page over real HTTP — not a stub, not a mock, not a
hand-built history — but it is *our* page rather than a third party's. This is
intentional and it is the honest trade: a fixture that must be regenerable on
demand cannot depend on a live travel site's markup, rate limits, bot walls, or
prices, all of which change without notice. Pointing the same agent at a public
booking site would make the recording unreproducible and, worse, would silently
re-record a DIFFERENT scenario every time the site changed. An internal
fare/options board behind the corporate travel desk is also the realistic
deployment for this workload. Nothing about the browse, the DOM, the model
call, the actions, or the tokens is simulated.

WHY gpt-4o-mini AND NOT THE FREE LOCAL OLLAMA
---------------------------------------------
browser-use drives the browser by making the model emit a STRICT JSON action
schema every step (``{"navigate": {...}}``, ``{"done": {...}}``, ...) against a
large serialized-DOM prompt. The local ``llama3:8b`` does not hold that schema
reliably: it flails, retries, and burns steps without ever reaching ``done``,
which would produce either no fixture or a dishonest failed one. This is the
documented "genuinely needs a real provider" case, and the run is deliberately
kept to a couple of steps (``initial_actions`` navigates deterministically
instead of paying the model to figure out the URL), so the real spend is a
fraction of a cent.

The recording uses the REAL ``BrowserUseAdapter`` (it wraps the bound
``agent.run`` and walks the REAL ``AgentHistoryList``); the flush is observed
through the ``_generate_fixtures`` capture seam (``set_trace_observer`` + a
no-op ``enqueue_upload``) so the sealed payload — real ``tool.call`` per real
browser action, real ``model.invoke`` from ``history.usage``, real
``cost.record`` (browser-use's own ``total_cost`` is 0.0 because
``calculate_cost`` is off, so the adapter honestly omits ``cost_usd`` and the
shared price-on-emit chokepoint derives it from the REAL prompt/completion
counts at real gpt-4o-mini rates), plus an intact attestation chain — is
captured but NEVER uploaded during generation. The sample uploads the sealed
fixture itself at run time.
"""

from __future__ import annotations

import asyncio
import datetime
import functools
import http.server
import importlib.metadata
import importlib.util
import json
import os
import socketserver
import sys
import tempfile
import threading

# Reuse the record-real-once seam from the sibling ``_generate_fixtures`` module
# (single source of truth for the fixture writer + capture config + model name).
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.dirname(_HERE)        # samples/data
_SAMPLES = os.path.dirname(_DATA)     # samples
_REPO = os.path.dirname(_SAMPLES)
for _p in (os.path.join(_REPO, "src"), _SAMPLES, _DATA):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# This module is named ``browser_use.py`` (to match the adapter). When the file
# is run directly, Python inserts its own directory at ``sys.path[0]``, which
# would shadow the real ``browser_use`` package for the function-local
# ``import browser_use``. Drop this module's own directory from the path so the
# framework import always resolves to the installed package (a no-op when
# imported as ``generators.browser_use``, since the package dir is not on the
# path then).
sys.path[:] = [_p for _p in sys.path if os.path.abspath(_p or ".") != _HERE]

from layerlens import Stratix  # noqa: E402
import layerlens.instrument._collector as _collector_mod  # noqa: E402
from layerlens.instrument._collector import set_trace_observer  # noqa: E402

import _generate_fixtures as _gf  # noqa: E402

_write = _gf._write
_CAPTURE = _gf._CAPTURE
OPENAI_MODEL = _gf.OPENAI_MODEL

#: Where the LAY-3614 recorded-corpus fixture for browser_use is sealed.
_CORPUS = os.path.join(_REPO, "tests", "fixtures", "recorded", "browser_use")

#: Loopback port for the throwaway board server. Bound only for the run.
_PORT = int(os.environ.get("BROWSER_USE_SAMPLE_PORT", "8731"))

#: The real headless Chromium browser-use drives over CDP. Overridable so the
#: recorder is not welded to one machine's playwright cache.
_CHROME = os.environ.get(
    "BROWSER_USE_SAMPLE_CHROME",
    os.path.expanduser(
        "~/Library/Caches/ms-playwright/chromium-1217/chrome-mac-arm64/"
        "Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing"
    ),
)


# --------------------------------------------------------------------------
# The REAL page the agent browses: a corporate travel desk's Trip Options
# Board. Boring, static, and deterministic on purpose — the agent's job is to
# apply the client's real constraints to it, which is the actual business task.
#
# The correct answer is NW-101 ($1,290): NW-102 is cheaper but has a stop,
# NW-103 is nonstop but non-refundable, NW-104 is nonstop/refundable but
# dearer, and NW-105 blows the $1,400 cap. A model that just grabs the smallest
# number gets it wrong — so the recorded answer is evidence the agent really
# read and reasoned over the rendered page.
# --------------------------------------------------------------------------
_BOARD_HTML = """<!doctype html>
<html>
  <head><title>Northwind Travel Desk — Trip Options Board</title></head>
  <body>
    <h1>Northwind Travel Desk — Trip Options Board</h1>
    <p>Client: Meridian Analytics · Request TRQ-4482 · Boston (BOS) &rarr; Lisbon (LIS) · October 2026</p>
    <table border="1" cellpadding="6">
      <thead>
        <tr>
          <th>Option</th><th>Route</th><th>Airline</th><th>Stops</th>
          <th>Depart</th><th>Return</th><th>Hotel</th><th>Cancellation</th><th>Total (USD)</th>
        </tr>
      </thead>
      <tbody>
        <tr><td>NW-101</td><td>BOS &rarr; LIS</td><td>TAP Air Portugal</td><td>Nonstop</td>
            <td>2026-10-12</td><td>2026-10-18</td><td>Hotel Baixa (4&#9733;)</td>
            <td>Free cancellation</td><td>1290</td></tr>
        <tr><td>NW-102</td><td>BOS &rarr; LIS</td><td>Azores Airlines</td><td>1 stop (PDL)</td>
            <td>2026-10-12</td><td>2026-10-18</td><td>Alfama Inn (3&#9733;)</td>
            <td>Free cancellation</td><td>1105</td></tr>
        <tr><td>NW-103</td><td>BOS &rarr; LIS</td><td>TAP Air Portugal</td><td>Nonstop</td>
            <td>2026-10-14</td><td>2026-10-20</td><td>Chiado Suites (4&#9733;)</td>
            <td>Non-refundable</td><td>1210</td></tr>
        <tr><td>NW-104</td><td>BOS &rarr; LIS</td><td>United</td><td>Nonstop</td>
            <td>2026-10-13</td><td>2026-10-19</td><td>Principe Real (4&#9733;)</td>
            <td>Free cancellation</td><td>1375</td></tr>
        <tr><td>NW-105</td><td>BOS &rarr; LIS</td><td>Delta</td><td>Nonstop</td>
            <td>2026-10-15</td><td>2026-10-21</td><td>Belem Garden (5&#9733;)</td>
            <td>Free cancellation</td><td>1580</td></tr>
      </tbody>
    </table>
    <p><small>Fares held for 24h. Corporate policy: nonstop preferred; refundable fares
    required for client-billable travel.</small></p>
  </body>
</html>
"""

#: The travel desk's real booking brief — the client's constraints, verbatim.
TASK = (
    "You are researching trip options for client Meridian Analytics (request TRQ-4482), "
    "Boston (BOS) to Lisbon (LIS) in October 2026. Read the Trip Options Board that is "
    "already open in the browser and pick the single best option that satisfies ALL of "
    "the client's requirements: the flight must be NONSTOP, the booking must have FREE "
    "CANCELLATION (the trip is client-billable), and the total must be UNDER $1,400. Of "
    "the options that qualify, choose the CHEAPEST. Report the option code, the airline, "
    "the hotel, and the total price, and state in one line why the cheaper options on the "
    "board were rejected."
)

AGENT_NAME = "trip-research-agent"


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    """Serves the board without spraying request logs over the recorder output."""

    def log_message(self, fmt, *args):  # noqa: A003 - stdlib signature
        pass


def _serve_board() -> "socketserver.TCPServer":
    """Serve the Trip Options Board over REAL HTTP on loopback, for this run."""
    directory = tempfile.mkdtemp(prefix="travel-board-")
    with open(os.path.join(directory, "index.html"), "w") as f:
        f.write(_BOARD_HTML)
    handler = functools.partial(_QuietHandler, directory=directory)
    socketserver.TCPServer.allow_reuse_address = True
    server = socketserver.TCPServer(("127.0.0.1", _PORT), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _build_agent(url: str):
    """A REAL browser_use.Agent driving a REAL headless Chromium over CDP."""
    from browser_use import Agent
    from browser_use.browser.profile import BrowserProfile
    from browser_use.llm import ChatOpenAI

    return Agent(
        task=TASK,
        llm=ChatOpenAI(model=OPENAI_MODEL),
        browser_profile=BrowserProfile(headless=True, executable_path=_CHROME),
        # Navigate deterministically rather than paying the model to discover the
        # URL — the research task is reading the board, not finding it.
        initial_actions=[{"navigate": {"url": url, "new_tab": False}}],
        # The board is text; vision would only add image tokens and cost.
        use_vision=False,
        enable_planning=False,
        use_judge=False,
        max_actions_per_step=2,
    )


def _scrubber():
    """The corpus scrubber from ``tests/instrument/_recorded.py``.

    Imported by path rather than as ``tests.instrument._recorded`` so the
    generator does not require the test package to be importable from the
    samples tree.
    """
    path = os.path.join(_REPO, "tests", "instrument", "_recorded.py")
    spec = importlib.util.spec_from_file_location("_layerlens_recorded_scrub", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _seal_corpus(history) -> str:
    """Seal the REAL ``AgentHistoryList`` as a LAY-3614 ``object`` corpus fixture.

    Serialized with browser-use's OWN ``AgentHistoryList.model_dump()`` so the
    committed body is exactly the framework's own wire shape. That custom dump
    emits ONLY ``history`` — it silently drops ``usage`` — so the real
    ``UsageSummary`` is dumped alongside it explicitly; without it the replay
    could not assert the real token counts at all.
    """
    scrub = _scrubber().scrub
    if not history.history:
        raise RuntimeError("no history recorded — the agent never took a step")
    if history.usage is None:
        raise RuntimeError("no usage recorded — refusing to seal a corpus with no real tokens")

    body = history.model_dump()
    body["usage"] = history.usage.model_dump()
    fixture = {
        "provenance": {
            "provider": "openai",
            "sdk_version": "browser-use %s" % (_browser_use_version() or "unknown"),
            "model": OPENAI_MODEL,
            "scenario": "travel_research",
            "captured_at": datetime.datetime.now(datetime.timezone.utc)
            .replace(microsecond=0)
            .isoformat(),
            "note": (
                "The real AgentHistoryList from a REAL headless-chromium browser-use run "
                "(see samples/data/generators/browser_use.py) — the same run that produced "
                "samples/data/traces/industry/travel_browseruse_research.jsonl. Recorded "
                "UPSTREAM of the BrowserUseAdapter's history walk (the parser under test): "
                "real ActionModel dumps, real ActionResult outcomes, real per-step URLs, and "
                "the real history-level UsageSummary browser-use's token service recorded."
            ),
        },
        "transport": "object",
        "sdk": "browser_use",
        "response": scrub(body),
    }
    os.makedirs(_CORPUS, exist_ok=True)
    path = os.path.join(_CORPUS, "travel_research.json")
    with open(path, "w") as f:
        json.dump(fixture, f, indent=1)
        f.write("\n")
    return path


def _browser_use_version():
    try:
        return importlib.metadata.version("browser-use")
    except importlib.metadata.PackageNotFoundError:
        return None


def _capture_browser_use(client: Stratix, agent):
    """Drive the REAL run through the REAL BrowserUseAdapter and seal the flush.

    The adapter wraps the bound ``agent.run`` and walks the real
    ``AgentHistoryList`` in a ``finally``; ``_end_run`` flushes the collector,
    which the observer seam catches while ``enqueue_upload`` is a no-op — so the
    recording never uploads or pollutes an org.

    Returns ``(payload, history)`` — the sealed trace and the REAL
    ``AgentHistoryList`` the SAME run produced, so both shipped artifacts come
    from one real browse rather than two divergent recordings.
    """
    from layerlens.instrument.adapters.frameworks.browser_use import BrowserUseAdapter

    captured: dict = {}
    set_trace_observer(lambda p: captured.setdefault("payload", p))
    orig = _collector_mod.enqueue_upload
    _collector_mod.enqueue_upload = lambda *a, **k: None
    adapter = BrowserUseAdapter(client, capture_config=_CAPTURE)
    # A real browser_use Agent declares no name of its own, so the honest agent
    # identity is the one the PRODUCER declares here — the same seam a customer
    # uses via ``instrument_browser_use(agent, agent_name=...)``.
    adapter.connect(target=agent, agent_name=AGENT_NAME)

    async def _drive():
        try:
            # max_steps: read the board and answer. The navigate is an initial
            # action, so the model's own steps are the reading + the done.
            return await agent.run(4)
        finally:
            await agent.close()

    try:
        history = asyncio.run(_drive())
    finally:
        try:
            adapter.disconnect()
        except Exception:
            pass
        set_trace_observer(None)
        _collector_mod.enqueue_upload = orig

    payload = captured.get("payload")
    if not payload:
        raise RuntimeError("no payload captured for browser_use run")
    return payload, history


def generate_browser_use_single(client: Stratix) -> dict:
    """Record the ``trip-research-agent`` really browsing the Trip Options Board.

    Named ``_single`` (not ``_research``) so the ``_generate_fixtures`` W2 loader
    discovers it by its existing ``generate_<adapter>_<single|multi>`` convention
    — adding ``"browser_use"`` to ``_W2_ADAPTERS`` is then the only wiring needed.
    There is no ``_multi``: browser-use drives ONE browser agent per run and
    declares no multi-agent topology, so a second agent would have to be invented.
    """
    server = _serve_board()
    url = "http://127.0.0.1:%d/index.html" % _PORT
    try:
        payload, history = _capture_browser_use(client, _build_agent(url))
    finally:
        server.shutdown()
        server.server_close()

    payload["tags"] = [
        "layerlens-sample",
        "industry",
        "travel",
        "trip-research",
        "browser-automation",
    ]

    events = payload.get("events", [])
    tools = [
        (e.get("payload") or {}).get("tool_name")
        for e in events
        if e.get("event_type") == "tool.call"
    ]
    mi = [e for e in events if e.get("event_type") == "model.invoke"]
    cr = [e for e in events if e.get("event_type") == "cost.record"]
    out = next((e for e in events if e.get("event_type") == "agent.output"), None)
    print(
        "  browser_use (trip-research-agent, real headless browse)  "
        "events=%d actions=%s model.invoke=%d cost.record=%d"
        % (len(events), tools, len(mi), len(cr))
    )
    if mi:
        p = mi[0]["payload"]
        print(
            "    model=%s provider=%s tokens=%s/%s/%s"
            % (
                p.get("model"),
                p.get("provider"),
                p.get("tokens_prompt"),
                p.get("tokens_completion"),
                p.get("tokens_total"),
            )
        )
    if cr:
        print("    cost_usd=%s" % cr[0]["payload"].get("cost_usd"))
    if out:
        print("    answer=%r" % (out["payload"].get("output_text") or "")[:120])
    print("  ->", _write([payload], "industry", "travel_browseruse_research"))
    # Seal the SAME run's real AgentHistoryList as the LAY-3614 corpus body.
    print("  ->", _seal_corpus(history), "\n")
    return payload


if __name__ == "__main__":  # pragma: no cover - manual regeneration entrypoint
    generate_browser_use_single(Stratix())
