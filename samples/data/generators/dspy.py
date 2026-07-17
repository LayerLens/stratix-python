"""ADP-PORT Family-B recorder for the ``dspy`` adapter (record-real-once).

Records ONE real, fully-instrumented ``dspy`` run and seals TWO artifacts from
that single run:

* ``samples/data/traces/industry/education_dspy_tutor.jsonl`` — the sealed
  real-trace fixture the Family-B sample (``samples/industry/education_dspy_tutor.py``)
  uploads. A ``TutorRAG`` (a real ``dspy.Module``) answers an enrolled student's
  question about Bessel's correction over the STAT-101 Week-3 lecture notes: it
  calls a real ``dspy.Tool`` (``search_course_material``) to retrieve the
  relevant note excerpts, then a real ``dspy.ChainOfThought`` grounds the tutoring
  answer in ONLY those excerpts and cites them. Renders one honest agent node
  (Agent column = ``TutorRAG``, Framework = ``dspy``, Status = ok) with the real
  nested module boundary (TutorRAG -> ChainOfThought -> Predict), the real
  ``tool.call`` for the retrieval, and the real ``model.invoke`` / ``cost.record``
  of the LM turn.

* ``tests/fixtures/recorded/dspy/default.json`` — the LAY-3614 recorded-corpus
  fixture (``transport: http``): the RAW ollama ``/api/chat`` response body that
  this same run really received, captured UPSTREAM of dspy's parser through a
  recording ``httpx`` transport. ``tests/instrument/adapters/frameworks/test_dspy_recorded.py``
  replays it back through a real ``dspy.LM`` so dspy's real ``ChatAdapter`` parses
  the real ``[[ ## field ## ]]`` blocks and the real ``DSPyAdapter`` emits off it —
  no creds, no network, no spend.

TRANSPORT — ollama ``llama3:8b`` (FREE, local). The scenario is honestly served by
it: llama3:8b produces dspy's strict ``[[ ## field ## ]]`` ChatAdapter format on
the first attempt for this signature (verified before recording — no JSONAdapter
fallback, no ``AdapterParseError``), and returns real ``prompt_eval_count`` /
``eval_count`` which litellm normalizes into the real ``usage`` block dspy keeps
on ``LM.history``. No paid provider is needed, so none is used.

HONEST COST NOTE — ``llama3:8b`` is not in the SDK pricing table (it is a local
model with no provider tariff), so the real ``cost.record`` this run emits carries
real token counts and ``cost_usd = None``. That is the truth of a local run and is
sealed as such; a dollar figure is NOT invented for it. The adapter's
``_split_model_id`` is still exercised end-to-end — the sealed events carry
``model="llama3:8b"`` + ``provider="ollama_chat"`` split off litellm's
``ollama_chat/llama3:8b`` id, which is the split the pricing lookup depends on.

Everything is a genuine run: the Framework column shows ``dspy`` (the framework
that really ran), the token counts are the ones ollama really reported, the
retrieval is a real scored search over the real notes corpus, and the attestation
chain is the collector's own.
"""

from __future__ import annotations

import os
import sys
import json
import datetime

# Reuse the record-real-once seam from the sibling ``_generate_fixtures`` module
# (single source of truth for the fixture writer + capture config).
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.dirname(_HERE)  # samples/data
_SAMPLES = os.path.dirname(_DATA)  # samples
_REPO = os.path.dirname(_SAMPLES)
for _p in (os.path.join(_REPO, "src"), _SAMPLES, _DATA):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# This module is named ``dspy.py`` (to match the adapter). When the file is run
# directly, Python inserts its own directory at ``sys.path[0]``, which would
# shadow the real ``dspy`` package for the function-local ``import dspy``. Drop
# this module's own directory from the path so the framework import always
# resolves to the installed package (a no-op when imported as ``generators.dspy``,
# since the package dir is not on the path then).
sys.path[:] = [_p for _p in sys.path if os.path.abspath(_p or ".") != _HERE]

from layerlens import Stratix  # noqa: E402
import layerlens.instrument._collector as _collector_mod  # noqa: E402
from layerlens.instrument._collector import set_trace_observer  # noqa: E402

import _generate_fixtures as _gf  # noqa: E402

_write = _gf._write
_CAPTURE = _gf._CAPTURE
OLLAMA_MODEL = _gf.OLLAMA_MODEL

#: Where the LAY-3614 recorded-corpus fixture for dspy is sealed.
_CORPUS = os.path.join(_REPO, "tests", "fixtures", "recorded", "dspy")

#: litellm addresses a local ollama through its ``ollama_chat/`` provider prefix;
#: the adapter's ``_split_model_id`` splits it back into model + provider.
_DSPY_MODEL_ID = "ollama_chat/%s" % OLLAMA_MODEL


# --------------------------------------------------------------------------
# The real course material the tutor retrieves over: a genuine excerpt set from
# an intro-statistics course's Week-3 lecture notes (non-sensitive, authored for
# the sample). The student's question is the single most-asked question of that
# week, so the retrieval has real work to do — the corpus deliberately contains
# near-miss notes (mean, standard deviation, CLT) that must NOT be cited.
# --------------------------------------------------------------------------
_COURSE_ID = "STAT-101"

_COURSE_NOTES = [
    {
        "note_id": "STAT101-W3-N1",
        "week": 3,
        "title": "The sample mean",
        "text": (
            "The sample mean x-bar is the sum of the observations divided by n. It is an "
            "unbiased estimator of the population mean mu: its expected value equals mu for "
            "any sample size. The sample mean is the balance point of the data."
        ),
    },
    {
        "note_id": "STAT101-W3-N2",
        "week": 3,
        "title": "Sample variance and the n-1 divisor",
        "text": (
            "The sample variance s-squared divides the sum of squared deviations by n-1, not "
            "by n. Dividing by n yields a biased estimator that systematically UNDERESTIMATES "
            "the population variance, because the deviations are measured about the sample "
            "mean x-bar rather than the unknown population mean mu, and the sum of squared "
            "deviations is minimized at x-bar. Dividing by n-1 corrects that downward bias "
            "exactly, making s-squared an unbiased estimator of sigma-squared."
        ),
    },
    {
        "note_id": "STAT101-W3-N3",
        "week": 3,
        "title": "Degrees of freedom and Bessel's correction",
        "text": (
            "The n-1 divisor is known as Bessel's correction, and n-1 is the degrees of "
            "freedom of the sample variance. Intuition: once x-bar is fixed, only n-1 of the "
            "n deviations can vary freely — the last one is determined because the deviations "
            "must sum to zero. So the sample supplies n-1, not n, independent pieces of "
            "information about spread."
        ),
    },
    {
        "note_id": "STAT101-W3-N4",
        "week": 3,
        "title": "Standard deviation",
        "text": (
            "The sample standard deviation s is the square root of the sample variance and is "
            "reported in the original units of the data. Note that s is NOT an unbiased "
            "estimator of sigma even though s-squared is unbiased for sigma-squared, because "
            "the square root is a non-linear function."
        ),
    },
    {
        "note_id": "STAT101-W4-N1",
        "week": 4,
        "title": "The Central Limit Theorem",
        "text": (
            "For a large enough sample size, the sampling distribution of the sample mean is "
            "approximately normal regardless of the shape of the population distribution, with "
            "standard error sigma over the square root of n."
        ),
    },
]

_STUDENT_QUESTION = (
    "On problem set 2 I lost points for computing the variance with n in the denominator. "
    "Why do we divide by n-1 instead of n when we compute the sample variance, and what "
    "does 'degrees of freedom' have to do with it?"
)

#: English words that carry no retrieval signal — dropped before scoring so the
#: overlap score reflects the real statistical terms of the question.
_STOPWORDS = frozenset(
    """a an and are as at be by do for from how i in is it its of on or so that the this to we
    what when why with you your not do i lost points problem set""".split()
)


def _tokenize(text: str) -> list:
    out = []
    word = []
    for ch in (text or "").lower():
        if ch.isalnum() or ch == "-":
            word.append(ch)
        elif word:
            out.append("".join(word))
            word = []
    if word:
        out.append("".join(word))
    return [w for w in out if w and w not in _STOPWORDS]


def _render_note(note: dict) -> str:
    return "[%s] %s\n%s" % (note["note_id"], note["title"], note["text"])


# --------------------------------------------------------------------------
# The real DSPy program. Declared at module scope so ``TutorRAG.__module__`` is
# this module (NOT ``dspy.*``) — which is exactly what the adapter's
# ``_honest_agent_name`` discriminates on to decide that ``TutorRAG`` is a
# developer-declared agent while ``ChainOfThought``/``Predict`` are framework
# primitives that must NOT reach the Agent column.
# --------------------------------------------------------------------------
def _build_program():
    """Build the real ``TutorRAG`` program (dspy imported function-locally)."""
    import dspy

    class TutorAnswer(dspy.Signature):
        """Answer an enrolled student's question about the course using ONLY the
        retrieved course material. Explain the reasoning like a tutor, not like a
        textbook. If the material does not answer the question, say so rather than
        guessing. Cite the note ids you actually used."""

        course_material: str = dspy.InputField(
            desc="Excerpts retrieved from the course lecture notes, each tagged with its note id."
        )
        question: str = dspy.InputField(desc="The student's question.")
        answer: str = dspy.OutputField(
            desc="A clear tutoring answer grounded strictly in the retrieved material."
        )
        citations: str = dspy.OutputField(
            desc="Comma-separated note ids from the material that support the answer."
        )

    class TutorRAG(dspy.Module):
        """A course tutoring assistant: retrieve the relevant lecture notes, then
        answer the student's question grounded in them."""

        def __init__(self, notes: list) -> None:
            super().__init__()
            self._notes = notes
            # A REAL dspy.Tool, so the retrieval crosses dspy's callback bus
            # (on_tool_start/on_tool_end) and lands as a real ``tool.call``.
            self.search_course_material = dspy.Tool(
                self._search,
                name="search_course_material",
                desc="Retrieve the most relevant excerpts from the course lecture notes for a query.",
            )
            self.answer = dspy.ChainOfThought(TutorAnswer)

        def _search(self, query: str, k: int = 3) -> str:
            """Retrieve the k lecture-note excerpts most relevant to the query.

            Args:
                query: The student's question or a search phrase.
                k: How many excerpts to return.
            """
            terms = _tokenize(query)
            scored = []
            for note in self._notes:
                haystack = _tokenize(note["title"] + " " + note["text"])
                # Real term-overlap score, length-normalized so a long note does
                # not win on volume alone.
                hits = sum(1 for t in terms if t in haystack)
                if not hits:
                    continue
                scored.append((hits / (len(haystack) ** 0.5), note))
            scored.sort(key=lambda pair: pair[0], reverse=True)
            top = [note for _score, note in scored[:k]]
            if not top:
                return "No course material matched the query."
            return "\n\n".join(_render_note(n) for n in top)

        def forward(self, question: str) -> "dspy.Prediction":
            material = self.search_course_material(query=question, k=3)
            return self.answer(course_material=material, question=question)

    return TutorRAG(_COURSE_NOTES)


# --------------------------------------------------------------------------
# Recording transport: forwards to the REAL local ollama and captures the raw
# HTTP interaction on the way back, so ONE run yields both the sealed trace and
# the LAY-3614 corpus fixture (the provider body, recorded upstream of dspy's
# parser — the one thing we do not control).
# --------------------------------------------------------------------------
class _RecordingTransport:
    """An ``httpx`` transport that proxies to the real ollama and records bodies."""

    def __init__(self) -> None:
        import httpx

        self._inner = httpx.HTTPTransport()
        self.interactions: list = []

    def handle_request(self, request):  # noqa: ANN001 - httpx transport protocol
        import httpx

        response = self._inner.handle_request(request)
        body = response.read()
        try:
            parsed = json.loads(body)
        except ValueError:
            parsed = None
        self.interactions.append(
            {
                "request": {"method": request.method, "path": request.url.path},
                "response": {
                    "status_code": response.status_code,
                    "json": parsed,
                    "text": None if parsed is not None else body.decode("utf-8", "replace"),
                    "headers": {
                        k: v
                        for k, v in response.headers.items()
                        if k.lower() in ("content-type", "date")
                    },
                },
            }
        )
        # The stream is consumed by read(); hand back an equivalent response.
        return httpx.Response(
            response.status_code,
            headers=response.headers,
            content=body,
            request=request,
        )

    def close(self) -> None:
        self._inner.close()


def _seal_corpus(transport: _RecordingTransport, *, dspy_version: str) -> str:
    """Write the recorded raw ollama bodies as a LAY-3614 ``http`` corpus fixture."""
    scrub = _scrubber().scrub
    if not transport.interactions:
        raise RuntimeError("no HTTP interaction recorded — the LM call never hit the transport")
    fixture = {
        "provenance": {
            "provider": "ollama",
            "sdk_version": dspy_version,
            "model": OLLAMA_MODEL,
            "scenario": "default",
            "captured_at": datetime.datetime.now(datetime.timezone.utc)
            .replace(microsecond=0)
            .isoformat(),
            "note": (
                "Raw ollama /api/chat body from a REAL dspy TutorRAG run (see "
                "samples/data/generators/dspy.py). Recorded upstream of dspy's ChatAdapter "
                "parser so the replay exercises dspy's real parse of a real body."
            ),
        },
        "transport": "http",
        "interactions": scrub(transport.interactions),
    }
    os.makedirs(_CORPUS, exist_ok=True)
    path = os.path.join(_CORPUS, "default.json")
    with open(path, "w") as f:
        json.dump(fixture, f, indent=1)
        f.write("\n")
    return path


def _scrubber():
    """The corpus scrubber from ``tests/instrument/_recorded.py``.

    Imported by path rather than as ``tests.instrument._recorded`` so the
    generator does not require the test package to be importable from the
    samples tree.
    """
    import importlib.util

    path = os.path.join(_REPO, "tests", "instrument", "_recorded.py")
    spec = importlib.util.spec_from_file_location("_layerlens_recorded_scrub", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# The recorder
# --------------------------------------------------------------------------
def generate_dspy_single(client: Stratix) -> dict:
    """Record the real ``TutorRAG`` tutoring run and seal both artifacts."""
    import httpx
    import dspy
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    from layerlens.instrument.adapters.frameworks.dspy import DSPyAdapter

    # Real litellm HTTP client, but through a transport that records the real
    # ollama bodies on their way back (the corpus capture rides the real run).
    recorder = _RecordingTransport()
    http_client = HTTPHandler()
    http_client.client = httpx.Client(transport=recorder, timeout=180.0)

    lm = dspy.LM(
        _DSPY_MODEL_ID,
        api_base=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        api_key="",
        client=http_client,
        # No cache: a cached hit would return a body the transport never saw, so
        # the corpus fixture would not correspond to this run's model.invoke.
        cache=False,
    )
    dspy.configure(lm=lm)
    program = _build_program()

    captured: dict = {}
    set_trace_observer(lambda p: captured.setdefault("payload", p))
    orig = _collector_mod.enqueue_upload
    _collector_mod.enqueue_upload = lambda *a, **k: None
    adapter = DSPyAdapter(client, capture_config=_CAPTURE)
    adapter.connect()
    try:
        prediction = program(question=_STUDENT_QUESTION)
    finally:
        try:
            adapter.disconnect()
        except Exception:
            pass
        set_trace_observer(None)
        _collector_mod.enqueue_upload = orig
        recorder.close()

    payload = captured.get("payload")
    if not payload:
        raise RuntimeError("no payload captured for dspy TutorRAG run")
    payload["tags"] = [
        "layerlens-sample",
        "industry",
        "education",
        "tutoring",
        "rag",
    ]

    events = payload.get("events", [])
    mi = [e for e in events if e.get("event_type") == "model.invoke"]
    cr = [e for e in events if e.get("event_type") == "cost.record"]
    tools = sorted(
        {(e.get("payload") or {}).get("tool_name") for e in events if e.get("event_type") == "tool.call"}
        - {None}
    )
    idents = sorted(
        {(e.get("payload") or {}).get("agent_name") for e in events if (e.get("payload") or {}).get("agent_name")}
    )
    modules = sorted(
        {(e.get("payload") or {}).get("module_type") for e in events if (e.get("payload") or {}).get("module_type")}
    )
    # An LM turn that produced no model.invoke means the run never reached the
    # provider (or the model was unresolvable) — sealing that as a "real" tutor
    # trace would be a lie, so fail loudly instead.
    if not mi:
        raise RuntimeError("dspy run emitted no model.invoke — refusing to seal a modelless tutor fixture")

    print(
        "  dspy single (TutorRAG, education tutoring RAG)  "
        "events=%d agents=%s modules=%s tools=%s model.invoke=%d cost.record=%d"
        % (len(events), idents, modules, tools, len(mi), len(cr))
    )
    print("     answer:", str(getattr(prediction, "answer", ""))[:120].replace("\n", " "), "...")
    print("     cites :", str(getattr(prediction, "citations", "")))
    print("  ->", _write([payload], "industry", "education_dspy_tutor"))
    print("  ->", _seal_corpus(recorder, dspy_version=str(getattr(dspy, "__version__", "?"))), "\n")
    return payload


if __name__ == "__main__":  # pragma: no cover - manual regeneration entrypoint
    generate_dspy_single(Stratix())
