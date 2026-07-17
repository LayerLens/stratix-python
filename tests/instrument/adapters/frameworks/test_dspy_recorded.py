"""Recorded-real-response replay for the DSPy framework (LAY-3614).

Drives a REAL ``dspy.Module`` (the ``TutorRAG`` program of the education Family-B
sample) whose ``dspy.LM`` is backed by litellm's real ``ollama_chat`` transformer
over ``httpx.MockTransport`` serving the captured ollama ``/api/chat`` body, with
the real ``DSPyAdapter`` attached via its first-party callback bus. This exercises
the full path — real ollama body -> real litellm ``ollama_chat`` transformation ->
real dspy ``ChatAdapter`` parse of the real ``[[ ## field ## ]]`` blocks -> real
``LM.history`` usage -> real adapter -> emitted events — which the unit doubles
(hand-built LM stand-ins in ``test_dspy.py``) never combine with a real provider
body.

The fixture is the raw ollama body recorded UPSTREAM of dspy's parser during the
SAME real run that produced ``samples/data/traces/industry/education_dspy_tutor.jsonl``
(see ``samples/data/generators/dspy.py``) — so this asserts our adapter against a
response shape we do not control.

The strong tells that the real provider shape flowed through:

* ``model.invoke`` reports ``tokens_prompt/completion/total = 620/138/758``. Those
  are ollama's real ``prompt_eval_count``/``eval_count`` off the recorded body,
  which litellm normalizes into the ``usage`` block dspy appends to ``LM.history``
  and which the adapter's ``_probe_usage`` correlates back by output identity.
  Nothing in the test hands the adapter a token count.
* The ``Prediction`` fields (``reasoning``/``answer``/``citations``) come from
  dspy's real ChatAdapter parse of the recorded body's field blocks, so the
  ``agent.output`` prediction is the recorded model's real answer.
* ``model``/``provider`` are ``llama3:8b`` / ``ollama_chat``, i.e. the adapter's
  ``_split_model_id`` really split litellm's ``ollama_chat/llama3:8b`` id.

HONEST COST NOTE — ``llama3:8b`` is a local model with no provider tariff and is
deliberately absent from the SDK pricing table, so the real ``cost.record`` this
replay emits carries real tokens and NO ``cost_usd``. That is asserted as the
truth of a local run: the recorded corpus proves the token accounting is real and
that the adapter does not invent a price for an unpriced model. The priced branch
of ``_split_model_id`` (``openai/gpt-4o-mini`` -> ``gpt-4o-mini`` -> a real
``cost_usd``) is covered by the unit suite, not by this local-body corpus.
"""

from __future__ import annotations

from typing import Any, Dict

import httpx
import pytest

pytest.importorskip("dspy")  # skips in the base venv (not installed there)
pytest.importorskip("litellm")

# The adapter targets dspy >= 3; the unified dev env resolves an older dspy (2.6.x)
# that can't co-resolve with the rest of the all-features set. The pinned matrix
# venv (dspy==3.2.1) is the authoritative lane.
from importlib.metadata import version as _pkg_version  # noqa: E402

if int(_pkg_version("dspy").split(".")[0]) < 3:
    pytest.skip("dspy adapter targets dspy >= 3", allow_module_level=True)

import dspy  # noqa: E402
from litellm.llms.custom_httpx.http_handler import HTTPHandler  # noqa: E402

from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.frameworks.dspy import DSPyAdapter  # noqa: E402

from .conftest import find_event, find_events, capture_framework_trace  # noqa: E402
from ..._recorded import load_recorded, mock_transport  # noqa: E402

# The real Week-3 excerpts the recorded run's retrieval returned. Kept verbatim so
# the replayed program does the same real retrieval work; the LM body is served
# from the fixture either way.
_NOTES = [
    {
        "note_id": "STAT101-W3-N2",
        "title": "Sample variance and the n-1 divisor",
        "text": (
            "The sample variance s-squared divides the sum of squared deviations by n-1, not "
            "by n. Dividing by n yields a biased estimator that systematically UNDERESTIMATES "
            "the population variance."
        ),
    },
    {
        "note_id": "STAT101-W3-N3",
        "title": "Degrees of freedom and Bessel's correction",
        "text": (
            "The n-1 divisor is known as Bessel's correction, and n-1 is the degrees of freedom of the sample variance."
        ),
    },
]

_QUESTION = (
    "Why do we divide by n-1 instead of n when we compute the sample variance, and what "
    "does 'degrees of freedom' have to do with it?"
)


class TutorAnswer(dspy.Signature):
    """Answer an enrolled student's question about the course using ONLY the
    retrieved course material. Cite the note ids you actually used."""

    course_material: str = dspy.InputField(desc="Excerpts retrieved from the course lecture notes.")
    question: str = dspy.InputField(desc="The student's question.")
    answer: str = dspy.OutputField(desc="A clear tutoring answer grounded strictly in the material.")
    citations: str = dspy.OutputField(desc="Comma-separated note ids that support the answer.")


class TutorRAG(dspy.Module):
    """The education sample's real program: retrieve course notes, then answer."""

    def __init__(self, notes: list) -> None:
        super().__init__()
        self._notes = notes
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
        return "\n\n".join("[%s] %s\n%s" % (n["note_id"], n["title"], n["text"]) for n in self._notes[:k])

    def forward(self, question: str) -> Any:
        material = self.search_course_material(query=question, k=3)
        return self.answer(course_material=material, question=question)


def _lm(fixture: Dict[str, Any]) -> Any:
    """A real ``dspy.LM`` whose litellm HTTP client serves the recorded body.

    ``HTTPHandler`` is litellm's documented ``client=`` seam for its base HTTP
    handler (the path ``ollama_chat`` takes), so litellm still does its REAL
    request transformation and its REAL response transformation against the
    recorded ollama body — only the socket is replaced.
    """
    transport, _ = mock_transport(fixture)
    http_client = HTTPHandler()
    http_client.client = httpx.Client(transport=transport)
    return dspy.LM(
        "ollama_chat/llama3:8b",
        api_base="http://localhost:11434",
        api_key="",
        client=http_client,
        cache=False,
    )


class TestDSPyRecorded:
    def test_tutor_program_over_recorded_ollama(self, mock_client):
        fixture = load_recorded("dspy", "default")
        uploaded = capture_framework_trace(mock_client)

        dspy.configure(lm=_lm(fixture))
        program = TutorRAG(_NOTES)

        adapter = DSPyAdapter(mock_client, capture_config=CaptureConfig.full())
        adapter.connect()
        try:
            prediction = program(question=_QUESTION)
        finally:
            adapter.disconnect()

        # dspy's REAL ChatAdapter parsed the recorded body's field blocks.
        assert "n-1" in prediction.answer
        assert "STAT101-W3-N2" in prediction.citations

        events = uploaded["events"]

        # --- model.invoke: the real body's real usage, and the real id split ---
        mi = find_event(events, "model.invoke")
        assert mi["payload"]["framework"] == "dspy"
        # _split_model_id really split litellm's "ollama_chat/llama3:8b": the BARE
        # model id is what the pricing table is keyed by, the prefix is the provider.
        assert mi["payload"]["model"] == "llama3:8b"
        assert mi["payload"]["model_name"] == "llama3:8b"
        assert mi["payload"]["provider"] == "ollama_chat"
        # ollama's real prompt_eval_count/eval_count off the recorded body, carried
        # through litellm's usage normalization onto LM.history.
        assert mi["payload"]["tokens_prompt"] == 620
        assert mi["payload"]["tokens_completion"] == 138
        assert mi["payload"]["tokens_total"] == 758
        # The LM event carries NO agent identity, and that is the honest outcome
        # here rather than an oversight: _emit_lm_call attributes to the INNERMOST
        # enclosing module frame, which in any real nested dspy program is dspy's
        # own ``Predict`` — a framework primitive with no producer-declared name,
        # so _honest_agent_name yields None and both keys are omitted together.
        # An omission is honest; stamping "Predict" here would fabricate an agent.
        # (Attribution to the nearest NAMED ancestor — TutorRAG, which genuinely
        # made this call — would be strictly better and is PINGed to the SDK owner;
        # this asserts what the adapter really does today.)
        assert "agent_name" not in mi["payload"]
        assert "agent_id" not in mi["payload"]
        # It is not orphaned, though: the span tree still parents it under the
        # module that made the call.
        assert mi["parent_span_id"]

        # --- cost.record: same real accounting; NO invented price for a local model ---
        cost = find_event(events, "cost.record")
        assert cost["payload"]["framework"] == "dspy"
        assert cost["payload"]["model"] == "llama3:8b"
        assert cost["payload"]["provider"] == "ollama_chat"
        assert cost["payload"]["tokens_prompt"] == 620
        assert cost["payload"]["tokens_completion"] == 138
        assert cost["payload"]["tokens_total"] == 758
        # llama3:8b carries no provider tariff and is absent from the pricing
        # table, so an honest record prices it at nothing rather than at 0.0.
        assert cost["payload"].get("cost_usd") is None

        # --- identity: the developer's class only, never dspy's primitives ---
        identity = find_event(events, "agent.identity")
        assert identity["payload"]["agent_name"] == "TutorRAG"
        named = {
            (e.get("payload") or {}).get("agent_name") for e in events if (e.get("payload") or {}).get("agent_name")
        }
        assert named == {"TutorRAG"}, (
            "a dspy framework primitive (ChainOfThought/Predict) or the synthesized "
            "_LayerLensTraced_* class leaked into the Agent column: %s" % sorted(named)
        )

        # --- the real nested module boundary: TutorRAG -> ChainOfThought -> Predict ---
        modules = {(e.get("payload") or {}).get("module_type") for e in find_events(events, "agent.input")}
        assert modules == {"TutorRAG", "ChainOfThought", "Predict"}

        # --- the real dspy.Tool retrieval crossed the callback bus ---
        tool = find_event(events, "tool.call")
        assert tool["payload"]["tool_name"] == "search_course_material"
        assert tool["payload"]["success"] is True
        assert tool["payload"]["agent_name"] == "TutorRAG"
        assert "STAT101-W3-N2" in tool["payload"]["output"]

        # --- the outer module's output carries the parsed real prediction ---
        outs = [
            e for e in find_events(events, "agent.output") if (e.get("payload") or {}).get("module_type") == "TutorRAG"
        ]
        assert len(outs) == 1
        assert "error_type" not in outs[0]["payload"]
        assert "n-1" in outs[0]["payload"]["prediction"]["answer"]

        # --- environment.config describes the real signature it really ran ---
        cfgs = {
            (e.get("payload") or {}).get("module_type"): e["payload"] for e in find_events(events, "environment.config")
        }
        assert cfgs["TutorRAG"]["predictor_count"] == 1
        assert cfgs["Predict"]["input_fields"] == ["course_material", "question"]
        # ``reasoning`` is NOT declared on TutorAnswer — dspy's real ChainOfThought
        # prepends it when it extends the signature, so seeing it here proves the
        # config was read off the real live predictor rather than off the source
        # Signature class.
        assert cfgs["Predict"]["output_fields"] == ["answer", "citations", "reasoning"]
        # ...and for the same reason the signature is dspy's dynamically-built
        # ``StringSignature`` (make_signature's default class name), not the
        # authored ``TutorAnswer``. The adapter reports what really ran.
        assert cfgs["Predict"]["signature"] == "StringSignature"
        # A program that was never compiled by an optimizer carries no demos.
        assert cfgs["Predict"]["demo_count"] == 0
