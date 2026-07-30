"""Recorded-real-response replay for the Haystack framework (LAY-3614).

Drives a REAL Haystack ``Pipeline`` whose ``OpenAIGenerator`` is backed by an
``httpx.MockTransport`` serving the captured OpenAI chat.completion response,
with the real ``HaystackAdapter`` attached. This exercises the full path — real
provider response shape -> real Haystack pipeline/component tracing spans -> real
adapter -> emitted events — which the unit doubles (hand-built tag dicts) never
combine with a real provider body.

The adapter instruments Haystack by swapping in its own
``haystack.tracing.tracer.actual_tracer``, so the natural run unit is a
``Pipeline.run()`` (a bare generator never opens a ``haystack.pipeline.run``
span, so nothing would flush). The pipeline's component span carries the real
generator output dict (``replies`` + a top-level ``meta`` list), and the adapter
reads the model id + usage straight off ``meta[0]``.

The MockTransport is injected through Haystack's documented, serialization-safe
``http_client_kwargs=`` seam: ``init_http_client`` forwards the dict to
``httpx.Client(**kwargs)``, and ``httpx.Client`` accepts ``transport=`` — so the
real OpenAI SDK client inside the generator does its real routing +
deserialization against the recorded body.

The strong tell that the real provider shape flowed through: ``model.invoke``
reports ``gpt-4o-mini-2024-07-18`` (the model echoed in the recorded *response*
body), not the ``gpt-4o-mini`` we *requested* — the adapter read it off the real
parsed generator ``meta``, not off our config.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

pytest.importorskip("haystack")  # skips in the base venv (not installed there)

from haystack import Pipeline  # noqa: E402
from haystack.utils import Secret
from haystack.components.generators.openai import OpenAIGenerator

import layerlens.instrument.adapters.frameworks.haystack as _mod  # noqa: E402
from layerlens.instrument.adapters.frameworks.haystack import HaystackAdapter

from .conftest import find_event, capture_framework_trace
from ..._recorded import load_recorded, mock_transport


@pytest.fixture(autouse=True)
def _arm_haystack_flag():
    # test_haystack.py runs in the same matrix pytest process; re-arm the module
    # flag so connect() sees the truthy value the real installed import sets,
    # independent of cross-file run order (and of any state the legacy file left).
    prev = _mod._HAS_HAYSTACK
    _mod._HAS_HAYSTACK = True
    yield
    _mod._HAS_HAYSTACK = prev


def _pipeline(fixture: Dict[str, Any]) -> Pipeline:
    transport, _ = mock_transport(fixture)
    # Haystack's OpenAIGenerator builds its OpenAI client via init_http_client,
    # which calls httpx.Client(**http_client_kwargs); httpx.Client accepts a
    # custom transport, so this is the public seam for the MockTransport.
    generator = OpenAIGenerator(
        api_key=Secret.from_token("test-key"),
        model="gpt-4o-mini",
        http_client_kwargs={"transport": transport},
    )
    pipe = Pipeline()
    pipe.add_component("llm", generator)
    return pipe


class TestHaystackRecorded:
    def test_pipeline_over_recorded_openai(self, mock_client):
        fixture = load_recorded("openai", "default")
        uploaded = capture_framework_trace(mock_client)

        adapter = HaystackAdapter(mock_client)
        adapter.connect()
        pipe = _pipeline(fixture)
        result = pipe.run({"llm": {"prompt": "Reply with exactly: pong"}})
        adapter.disconnect()

        assert result["llm"]["replies"][0] == "pong"

        events = uploaded["events"]

        # The real component span carries the real generator output meta parsed
        # off the recorded chat.completion body — model id is the *response*
        # model, not the requested one.
        mi = find_event(events, "model.invoke")
        assert mi["payload"]["model"] == "gpt-4o-mini-2024-07-18"
        assert mi["payload"]["component_type"] == "OpenAIGenerator"
        assert mi["payload"]["tokens_prompt"] == 12
        assert mi["payload"]["tokens_completion"] == 1
        assert mi["payload"]["tokens_total"] == 13

        # cost.record echoes the same real usage from the recorded response.
        cost = find_event(events, "cost.record")
        assert cost["payload"]["framework"] == "haystack"
        assert cost["payload"]["model"] == "gpt-4o-mini-2024-07-18"
        assert cost["payload"]["tokens_prompt"] == 12
        assert cost["payload"]["tokens_completion"] == 1
        assert cost["payload"]["tokens_total"] == 13

        # The pipeline lifecycle frames the run with its own input/output pair.
        out = find_event(events, "agent.output")
        assert out["payload"]["framework"] == "haystack"
