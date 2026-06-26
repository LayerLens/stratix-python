"""Recorded-real-response replay for the Ollama provider (LAY-3614).

Modern ``ollama.chat`` returns a ``ChatResponse`` *object* (a pydantic model),
not the plain dict the adapter's ``extract_output`` / ``extract_meta`` assumed.
This replay drives the adapter with the REAL captured ``ChatResponse`` and
asserts the emitted ``model.invoke`` carries the real output + token usage —
catching the shape drift the dict-only doubles never exercised.
"""

from __future__ import annotations

from types import SimpleNamespace

import ollama

from layerlens.instrument import trace
from layerlens.instrument.adapters.providers.ollama import OllamaProvider

from ...conftest import find_event
from ..._recorded import load_recorded


def _fake_client(response):
    # OllamaProvider.connect wraps target.chat; the wrapper returns whatever the
    # original returns, so a real ChatResponse flows into the parser.
    return SimpleNamespace(chat=lambda **kwargs: response)


class TestOllamaRecorded:
    def test_chat_object_real_shape(self, mock_client, capture_trace):
        fixture = load_recorded("ollama", "default")
        response = ollama.ChatResponse(**fixture["response"])
        client = _fake_client(response)
        provider = OllamaProvider()
        provider.connect(client)

        @trace(mock_client)
        def agent():
            r = client.chat(
                model="llama3:8b",
                messages=[{"role": "user", "content": "Reply with exactly: pong"}],
            )
            return r["message"]["content"]

        assert agent() == "pong"

        mi = find_event(capture_trace["events"], "model.invoke")
        assert mi["payload"]["name"] == "ollama.chat"
        assert mi["payload"]["response_model"] == "llama3:8b"
        assert mi["payload"]["finish_reason"] == "stop"
        # The real ChatResponse carries the assistant message + token counts —
        # the dict-only parser dropped all of these for the object form.
        assert mi["payload"]["output_message"] == {"role": "assistant", "content": "pong"}
        usage = mi["payload"]["usage"]
        assert usage["prompt_tokens"] == 15
        assert usage["completion_tokens"] == 2
        assert usage["total_tokens"] == 17
        assert mi["payload"]["duration_ms"] > 0

        provider.disconnect()

    def test_provenance(self):
        prov = load_recorded("ollama", "default")["provenance"]
        assert prov["provider"] == "ollama"
        assert prov["captured_at"] and prov["captured_at"] != "pending-creds"
