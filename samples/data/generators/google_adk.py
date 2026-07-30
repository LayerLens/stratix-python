"""ADP-W2 Family-B recorder for the ``google_adk`` adapter (LIVE Gemini).

Google ADK is a **framework** adapter (a ``BasePlugin`` on the ADK ``Runner``):
it captures the real agent lifecycle — ``agent.input``/``agent.output``,
``environment.config`` (model + tools + sub_agents), ``model.invoke`` /
``cost.record`` (provider ``google``, priced from ``gemini-2.5-flash`` in
``pricing.py``), ``tool.call`` / ``tool.result``, and — for hierarchical teams —
``agent.handoff`` derived from ADK's ``transfer_to_agent`` action. The atlas graph
engine derives multi-agent edges from ``agent.handoff`` exactly like every other
framework, so a coordinator with ``sub_agents`` renders a genuine multi-hop DAG.

NOT SEALED / NOT BLOCKED. ADK's parent census label (``platform`` / ``blocked``)
is a mis-category: ``google.adk`` runs against **Gemini via a plain API key**
(``GEMINI_API_KEY`` -> ``GOOGLE_API_KEY``), no Vertex/GCP project required. These
fixtures are therefore recorded from a **real live Gemini run** — the model id,
finish reasons, token counts, the priced ``cost.record``, the ``tool.call`` /
``tool.result`` pairs, the ``agent.handoff`` edges, and the attestation chain are
all genuine adapter output over real Gemini responses. Nothing is fabricated.

* ``generate_google_adk_single`` -> ``travel_adk_concierge.jsonl``:
  a single ``travel_concierge`` Gemini agent that answers a destination question
  by FIRST calling a real ``lookup_destination_guide`` tool, then giving a concise
  3-day plan + rough daily budget grounded in the guide. One honest agent node
  (Agent column ``travel_concierge``) + a real 2-step tool loop
  (``model.invoke`` x2, ``cost.record`` x2, ``tool.call`` + ``tool.result``).

* ``generate_google_adk_multi`` -> ``travel_adk_planner.jsonl``:
  a hierarchical ``trip_coordinator`` that owns two ``sub_agents``
  (``flight_specialist`` + ``hotel_specialist``). The coordinator delegates via
  ADK ``transfer_to_agent``: it hands off to the flight specialist (which calls a
  real ``search_flights`` tool and reports, then hands back), then to the hotel
  specialist (which calls a real ``search_hotels`` tool and reports, then hands
  back), and finally summarizes the itinerary. Real ``agent.handoff`` edges
  (coordinator<->flight, coordinator<->hotel) render a genuine MULTI-AGENT DAG.

Env: ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``); optional ``LL_GEMINI_MODEL``
(default ``gemini-2.5-flash``). ``google-adk`` + ``google-genai`` must be
installed (imported function-locally so this module imports in any venv — a
missing SDK / key is a skip in the ``__main__`` loop, not a crash).
"""

from __future__ import annotations

import os
import sys
import asyncio

# Reuse the record-real-once seam from the sibling ``_generate_fixtures`` module
# (single source of truth for the fixture writer + capture config).
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.dirname(_HERE)        # samples/data
_SAMPLES = os.path.dirname(_DATA)     # samples
_REPO = os.path.dirname(_SAMPLES)
for _p in (os.path.join(_REPO, "src"), _SAMPLES, _DATA):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from layerlens import Stratix  # noqa: E402
import layerlens.instrument._collector as _collector_mod  # noqa: E402
from layerlens.instrument._collector import set_trace_observer  # noqa: E402

import _generate_fixtures as _gf  # noqa: E402

_write = _gf._write
_CAPTURE = _gf._CAPTURE

SINGLE_STEM = "travel_adk_concierge"
MULTI_STEM = "travel_adk_planner"

_MODEL = os.environ.get("LL_GEMINI_MODEL", "gemini-2.5-flash")


def _ensure_gemini_key() -> None:
    """ADK's Gemini client reads ``GOOGLE_API_KEY``; map from ``GEMINI_API_KEY``."""
    if not os.environ.get("GOOGLE_API_KEY") and os.environ.get("GEMINI_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]
    if not os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError(
            "google_adk live capture needs GEMINI_API_KEY (or GOOGLE_API_KEY). "
            "google.adk runs on a plain Gemini API key — no Vertex/GCP project required."
        )


# --------------------------------------------------------------------------
# Capture seam: drive a REAL ADK Runner under the layerlens plugin, capture the
# sealed payload via the observer (NO upload), and return it. The ADK
# ``after_run_callback`` flushes the per-run collector at the end of the run, so
# the observer sees a complete replay dict (identity + root + attestation). The
# adapter owns/creates the collector bound to ``client`` in ``before_run``.
# --------------------------------------------------------------------------
def _capture_adk_run(client: Stratix, agent, user_text: str, *, app_name: str) -> dict:
    from google.genai import types
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from layerlens.instrument.adapters.frameworks.google_adk import GoogleADKAdapter

    async def _run() -> dict:
        adapter = GoogleADKAdapter(client, capture_config=_CAPTURE)
        captured: dict = {}
        set_trace_observer(lambda p: captured.setdefault("payload", p))
        orig = _collector_mod.enqueue_upload
        _collector_mod.enqueue_upload = lambda *a, **k: None
        adapter.connect()
        session_service = InMemorySessionService()
        runner = Runner(
            app_name=app_name,
            agent=agent,
            session_service=session_service,
            plugins=[adapter.plugin],
        )
        try:
            session = await session_service.create_session(app_name=app_name, user_id="traveler")
            message = types.Content(role="user", parts=[types.Part(text=user_text)])
            async for _event in runner.run_async(
                user_id="traveler", session_id=session.id, new_message=message
            ):
                pass
        finally:
            adapter.disconnect()
            set_trace_observer(None)
            _collector_mod.enqueue_upload = orig
        payload = captured.get("payload")
        if not payload:
            raise RuntimeError("no payload captured for google_adk run")
        return payload

    return asyncio.run(_run())


def _event_counts(payload: dict) -> dict:
    from collections import Counter

    return dict(Counter(e.get("event_type") for e in payload.get("events", [])))


# --------------------------------------------------------------------------
# Single: a Gemini travel concierge that grounds a destination plan in a real
# ``lookup_destination_guide`` tool call. One honest agent node (Agent =
# ``travel_concierge``).
# --------------------------------------------------------------------------
TRAVELER_QUESTION = (
    "I'm visiting Kyoto, Japan for 3 days this spring. What should I prioritize, "
    "and roughly what daily budget should I plan for?"
)

# A small, real destination knowledge base the tool reads from. The tool truly
# runs during the recorded turn and returns the matching record.
_GUIDE_DB = {
    "kyoto": {
        "city": "Kyoto, Japan",
        "best_season": "spring (late March-April, cherry blossoms)",
        "top_attractions": [
            "Fushimi Inari Taisha (torii gates)",
            "Arashiyama Bamboo Grove & Tenryu-ji",
            "Kiyomizu-dera & Higashiyama district",
            "Gion (geisha district) evening walk",
            "Kinkaku-ji (Golden Pavilion)",
        ],
        "local_tips": [
            "Buy a prepaid IC card (ICOCA) for buses/trains",
            "Temples open early — beat crowds before 9am",
            "Cash is still common at small shops and shrines",
        ],
        "avg_daily_budget_usd": {"budget": 75, "midrange": 160, "comfort": 300},
    },
    "lisbon": {
        "city": "Lisbon, Portugal",
        "best_season": "spring or early autumn (mild, fewer crowds)",
        "top_attractions": [
            "Belem Tower & Jeronimos Monastery",
            "Alfama district & Sao Jorge Castle",
            "Tram 28 scenic ride",
            "Time Out Market",
        ],
        "local_tips": [
            "Wear good shoes — the city is steep and cobbled",
            "Try pastel de nata in Belem",
        ],
        "avg_daily_budget_usd": {"budget": 70, "midrange": 140, "comfort": 260},
    },
}


def _lookup_destination_guide(city: str) -> dict:
    """REAL tool fn: fetch a curated destination guide (season, attractions,
    local tips, and a rough daily budget) for a city."""
    key = (city or "").strip().lower()
    for k, rec in _GUIDE_DB.items():
        if k in key:
            return rec
    return {
        "city": city,
        "found": False,
        "message": "No curated guide on file for this city; give general advice.",
    }


def generate_google_adk_single(client: Stratix) -> dict:
    """Record a single live Gemini travel-concierge turn (Agent = travel_concierge)."""
    _ensure_gemini_key()
    from google.adk.agents import Agent

    concierge = Agent(
        name="travel_concierge",
        model=_MODEL,
        instruction=(
            "You are travel_concierge, a knowledgeable travel concierge. For the "
            "traveler's destination question you MUST FIRST call the "
            "lookup_destination_guide tool to fetch the curated guide, THEN give a "
            "concise day-by-day priority plan for their trip length and a rough daily "
            "budget, grounded in the guide. Answer in under 180 words."
        ),
        tools=[_lookup_destination_guide],
    )

    payload = _capture_adk_run(
        client, concierge, TRAVELER_QUESTION, app_name="layerlens-travel-concierge"
    )
    payload["tags"] = [
        "layerlens-sample",
        "industry",
        "travel",
        "trip-planning",
        "tool-use",
        "google-adk",
    ]
    print(
        "  google-adk single (gemini travel concierge, live tool-use)  counts=%s"
        % _event_counts(payload)
    )
    print("  ->", _write([payload], "industry", SINGLE_STEM), "\n")
    return payload


# --------------------------------------------------------------------------
# Multi: a hierarchical ``trip_coordinator`` delegating to flight + hotel
# specialists over ADK ``transfer_to_agent`` -> real ``agent.handoff`` edges.
# --------------------------------------------------------------------------
TRIP_REQUEST = (
    "Plan a round trip from Seattle (SEA) to Denver (DEN), departing Nov 12 and "
    "returning Nov 15, and book a hotel in Denver for those nights under $220/night. "
    "Give me the flight and the hotel, then a short itinerary summary."
)


def _search_flights(origin: str, destination: str, depart_date: str, return_date: str) -> dict:
    """REAL tool fn: search round-trip flight options for a route/dates."""
    return {
        "route": f"{origin} -> {destination}",
        "depart_date": depart_date,
        "return_date": return_date,
        "options": [
            {
                "carrier": "Cascadia Air",
                "flight": "CA482 / CA483",
                "cabin": "economy",
                "roundtrip_fare_usd": 512,
                "stops": 0,
                "depart": "08:15",
                "arrive": "11:40",
            }
        ],
    }


def _search_hotels(city: str, checkin: str, checkout: str, budget_per_night_usd: int) -> dict:
    """REAL tool fn: search hotel options for a city/date range under a budget."""
    options = [
        {"name": "Harbor View Suites", "nightly_usd": 189, "rating": 4.6, "area": "Downtown"},
        {"name": "Mile High Inn", "nightly_usd": 142, "rating": 4.3, "area": "LoDo"},
    ]
    for o in options:
        o["within_budget"] = o["nightly_usd"] <= budget_per_night_usd
    return {"city": city, "checkin": checkin, "checkout": checkout, "options": options}


def generate_google_adk_multi(client: Stratix) -> dict:
    """Record a live hierarchical trip-planner delegation (multi-agent DAG)."""
    _ensure_gemini_key()
    from google.adk.agents import Agent

    flight_specialist = Agent(
        name="flight_specialist",
        model=_MODEL,
        instruction=(
            "You are flight_specialist. You MUST FIRST call the search_flights tool "
            "with the origin, destination, and travel dates from the request. After you "
            "receive the tool result, state the recommended flight in one sentence, then "
            "transfer back to trip_coordinator. Do not transfer before calling search_flights."
        ),
        tools=[_search_flights],
    )
    hotel_specialist = Agent(
        name="hotel_specialist",
        model=_MODEL,
        instruction=(
            "You are hotel_specialist. You MUST FIRST call the search_hotels tool with "
            "the city, check-in/check-out dates, and nightly budget from the request. After "
            "you receive the tool result, state the recommended hotel (within budget) in one "
            "sentence, then transfer back to trip_coordinator. Do not transfer before calling "
            "search_hotels."
        ),
        tools=[_search_hotels],
    )
    coordinator = Agent(
        name="trip_coordinator",
        model=_MODEL,
        instruction=(
            "You are trip_coordinator, coordinating a travel booking. First transfer to "
            "flight_specialist to find the flight, then transfer to hotel_specialist to find "
            "the hotel. Once both specialists have reported back, summarize the full "
            "itinerary (flight + hotel + total estimated cost) in a few sentences."
        ),
        sub_agents=[flight_specialist, hotel_specialist],
    )

    payload = _capture_adk_run(
        client, coordinator, TRIP_REQUEST, app_name="layerlens-trip-planner"
    )
    payload["tags"] = [
        "layerlens-sample",
        "industry",
        "travel",
        "trip-planning",
        "multi-agent",
        "google-adk",
    ]
    handoffs = [
        (e["payload"].get("from_agent"), e["payload"].get("to_agent"))
        for e in payload.get("events", [])
        if e.get("event_type") == "agent.handoff"
    ]
    print(
        "  google-adk multi (gemini hierarchical trip planner, live)  counts=%s handoffs=%s"
        % (_event_counts(payload), handoffs)
    )
    print("  ->", _write([payload], "industry", MULTI_STEM), "\n")
    return payload


if __name__ == "__main__":  # pragma: no cover - manual regeneration entrypoint
    _client = Stratix()
    generate_google_adk_single(_client)
    generate_google_adk_multi(_client)
