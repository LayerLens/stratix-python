"""Offline privacy + params + attestation floor for the aws_bedrock provider.

Closes the W1 census cells that were previously proven only in gated live lanes
(no AWS account / model access), so a regression fails in plain CI with no
credentials. Every test drives the REAL ``BedrockProvider`` over a real
``bedrock-runtime`` boto3 client whose network boundary is a ``botocore.stub``
Stubber (real botocore parameter validation + serialization, no network), and
returns realistic Amazon-Nova wire shapes — never a hand-rolled mock of the
adapter's own output.

* Redaction   — ``capture_content=False`` on a Nova ``invoke_model`` strips the
                prompt (``messages``) AND the completion (``output_message``)
                while usage + the honest model id remain, with a ``True``
                vacuity control that proves the assertion is not vacuous, plus a
                SENTINEL sweep over ``json.dumps(events)`` in BOTH directions.
* Params      — the request params (``modelId`` + ``inferenceConfig``, incl.
                ``maxTokens``/``temperature``) are captured HONESTLY (exact
                values) on a real ``converse`` call, and the raw message content
                is gated by ``capture_content`` (present when on, absent when
                off) — paired control on the SAME path.
* Attestation — the captured Nova trace's attestation chain verifies OFFLINE
                (mirrors the live harness ``_assert_attestation``): every
                envelope is reconstructed from the stored
                ``attestation.chain.events`` and ``verify_chain(...).valid``.
"""

from __future__ import annotations

import io
import json
from typing import Any, Dict, Tuple

import pytest

boto3 = pytest.importorskip("boto3")
from botocore.stub import ANY, Stubber  # noqa: E402
from botocore.response import StreamingBody  # noqa: E402

from layerlens.instrument import trace  # noqa: E402
from layerlens.instrument._capture_config import CaptureConfig  # noqa: E402
from layerlens.instrument.adapters.providers.bedrock import BedrockProvider  # noqa: E402

from ...conftest import find_event  # noqa: E402

SENTINEL = "LL-SENTINEL-7f3a9c2e"

_NOVA_ID = "amazon.nova-micro-v1:0"
_CLAUDE_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"


# ---------------------------------------------------------------------------
# Helpers — real bedrock-runtime client + real Nova wire shapes
# ---------------------------------------------------------------------------
def _make_client() -> Any:
    """Real bedrock-runtime client with static fake credentials (offline-safe)."""
    return boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
        aws_session_token="testing",
    )


def _nova_body_bytes(text: str, input_tokens: int = 10, output_tokens: int = 5) -> bytes:
    """Amazon Nova invoke_model RESPONSE body (Converse-shaped, NOT Titan)."""
    return json.dumps(
        {
            "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
            "usage": {
                "inputTokens": input_tokens,
                "outputTokens": output_tokens,
                "totalTokens": input_tokens + output_tokens,
            },
            "stopReason": "end_turn",
        }
    ).encode("utf-8")


def _nova_request_body(user_text: str, system_text: str = "You are terse.") -> str:
    """Amazon Nova invoke_model REQUEST body (schemaVersion messages-v1)."""
    return json.dumps(
        {
            "schemaVersion": "messages-v1",
            "system": [{"text": system_text}],
            "messages": [{"role": "user", "content": [{"text": user_text}]}],
            "inferenceConfig": {"maxTokens": 256},
        }
    )


def _invoke_model_stub_response(body: bytes) -> Dict[str, Any]:
    return {
        "body": StreamingBody(io.BytesIO(body), len(body)),
        "contentType": "application/json",
        "ResponseMetadata": {"RequestId": "11111111-2222-3333-4444-555555555555"},
    }


def _connect(client: Any) -> Tuple[BedrockProvider, Stubber]:
    provider = BedrockProvider()
    provider.connect(client)
    stubber = Stubber(client)
    stubber.activate()
    return provider, stubber


def _run_nova_invoke(mock_client, config, *, user_text: str, output_text: str) -> None:
    """Drive a real Nova invoke_model through the real adapter under *config*."""
    client = _make_client()
    provider, stubber = _connect(client)
    stubber.add_response(
        "invoke_model",
        _invoke_model_stub_response(_nova_body_bytes(text=output_text)),
        {"modelId": _NOVA_ID, "body": ANY, "accept": "application/json", "contentType": "application/json"},
    )

    @trace(mock_client, capture_config=config)
    def run():
        r = client.invoke_model(
            modelId=_NOVA_ID,
            body=_nova_request_body(user_text),
            accept="application/json",
            contentType="application/json",
        )
        return json.loads(r["body"].read())

    run()
    stubber.assert_no_pending_responses()
    provider.disconnect()


# ---------------------------------------------------------------------------
# Redaction floor (offline, credential-free)
# ---------------------------------------------------------------------------
class TestRedactionFloor:
    def test_content_stripped_when_capture_content_false(self, mock_client, capture_trace):
        _run_nova_invoke(
            mock_client,
            CaptureConfig(capture_content=False),
            user_text="Say hello",
            output_text="Hi there!",
        )

        mi = find_event(capture_trace["events"], "model.invoke")["payload"]
        # Content stripped: neither the prompt nor the completion survives.
        assert "messages" not in mi
        assert "output_message" not in mi
        # Redaction removes CONTENT, not metadata: usage + the honest model id remain.
        assert mi["model"] == _NOVA_ID
        assert mi["usage"]["prompt_tokens"] == 10
        assert mi["usage"]["completion_tokens"] == 5
        assert mi["usage"]["total_tokens"] == 15
        # Flat token keys (atlas extractor reads these) also survive redaction.
        assert mi["completion_tokens"] == 5

    def test_content_present_when_capture_content_true(self, mock_client, capture_trace):
        """Vacuity control: the SAME path DOES carry content when capture is on."""
        _run_nova_invoke(
            mock_client,
            CaptureConfig.full(),
            user_text="Say hello",
            output_text="Hi there!",
        )

        mi = find_event(capture_trace["events"], "model.invoke")["payload"]
        assert {"role": "user", "content": "Say hello"} in mi["messages"]
        assert mi["output_message"] == {"role": "assistant", "content": "Hi there!"}

    def test_sentinel_never_leaks_when_redacted(self, mock_client, capture_trace):
        _run_nova_invoke(
            mock_client,
            CaptureConfig(capture_content=False),
            user_text=f"Remember {SENTINEL}",
            output_text=f"Secret is {SENTINEL}",
        )
        blob = json.dumps(capture_trace["events"])
        assert SENTINEL not in blob, "content redaction leaked the SENTINEL into the stored trace"

    def test_sentinel_present_when_capture_on(self, mock_client, capture_trace):
        """Vacuity control for the sweep above."""
        _run_nova_invoke(
            mock_client,
            CaptureConfig.full(),
            user_text=f"Remember {SENTINEL}",
            output_text=f"Secret is {SENTINEL}",
        )
        assert SENTINEL in json.dumps(capture_trace["events"])


# ---------------------------------------------------------------------------
# Params / privacy sweep (converse — inferenceConfig is a top-level request param)
# ---------------------------------------------------------------------------
class TestParamsPrivacySweep:
    def _run_converse(self, mock_client, config) -> None:
        client = _make_client()
        provider, stubber = _connect(client)
        stubber.add_response(
            "converse",
            {
                "output": {"message": {"role": "assistant", "content": [{"text": "Paris."}]}},
                "stopReason": "end_turn",
                "usage": {"inputTokens": 9, "outputTokens": 7, "totalTokens": 16},
                "metrics": {"latencyMs": 142},
                "ResponseMetadata": {"RequestId": "66666666-7777-8888-9999-000000000000"},
            },
            {
                "modelId": _CLAUDE_ID,
                "messages": ANY,
                "inferenceConfig": {"maxTokens": 100, "temperature": 0.2},
            },
        )

        @trace(mock_client, capture_config=config)
        def run():
            return client.converse(
                modelId=_CLAUDE_ID,
                messages=[{"role": "user", "content": [{"text": "What is the capital of France?"}]}],
                inferenceConfig={"maxTokens": 100, "temperature": 0.2},
            )

        run()
        stubber.assert_no_pending_responses()
        provider.disconnect()

    def test_params_captured_honestly_and_content_present_when_on(self, mock_client, capture_trace):
        self._run_converse(mock_client, CaptureConfig.full())
        mi = find_event(capture_trace["events"], "model.invoke")["payload"]

        # Request params captured HONESTLY — the exact values the caller passed.
        assert mi["parameters"]["modelId"] == _CLAUDE_ID
        assert mi["parameters"]["inferenceConfig"] == {"maxTokens": 100, "temperature": 0.2}
        # Raw message content is present when capture is on.
        assert mi["messages"] == [{"role": "user", "content": "What is the capital of France?"}]

    def test_content_gated_off_but_model_id_honest(self, mock_client, capture_trace):
        """Paired control: same path, capture off — raw content is gated, but the
        honest model-id metadata still survives (redaction never blinds the
        model column)."""
        self._run_converse(mock_client, CaptureConfig(capture_content=False))
        mi = find_event(capture_trace["events"], "model.invoke")["payload"]

        # Raw message content gated by capture_content.
        assert "messages" not in mi
        assert "output_message" not in mi
        # The raw request temperature/inferenceConfig is content-adjacent config;
        # under deny-by-default it does not survive in parameters either.
        assert mi["parameters"] == {}
        # Honest model id (metadata) still surfaces at the top level.
        assert mi["model"] == _CLAUDE_ID
        # Usage still recorded so cost/observability are not blinded.
        assert mi["usage"]["total_tokens"] == 16


# ---------------------------------------------------------------------------
# Offline attestation-chain verification
# ---------------------------------------------------------------------------
class TestAttestationOffline:
    def test_attestation_chain_verifies_offline(self, mock_client, capture_trace):
        from layerlens.attestation._verify import verify_chain
        from layerlens.attestation._envelope import HashScope, AttestationEnvelope

        _run_nova_invoke(
            mock_client,
            CaptureConfig.full(),
            user_text="Say hello",
            output_text="Hi there!",
        )

        events = capture_trace["events"]
        raw = (capture_trace["attestation"].get("chain") or {}).get("events") or []
        envelopes = [
            AttestationEnvelope(
                hash=e["hash"],
                scope=HashScope(e["scope"]),
                previous_hash=e.get("previous_hash"),
            )
            for e in raw
        ]
        assert envelopes, "no attestation envelopes captured"
        assert len(envelopes) == len(events)
        result = verify_chain(envelopes)
        assert result.valid, f"attestation chain invalid: {result.error}"
