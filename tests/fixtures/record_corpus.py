"""Offline capture harness for the recorded-real-response corpus (LAY-3614).

⚠️  NOT a test and NEVER run in CI. This makes REAL provider/service calls using
the live creds in ``tests/e2e/live/.env`` and writes scrubbed fixtures under
``tests/fixtures/recorded/<adapter>/<scenario>.json``. It is gated on
``LAYERLENS_RECORD=1`` (same spirit as ``LAYERLENS_LIVE``) and dispatched per
adapter so spend stays controlled and one-at-a-time:

    set -a; . tests/e2e/live/.env; set +a
    LAYERLENS_RECORD=1 rye run python tests/fixtures/record_corpus.py openai anthropic ...

It records UPSTREAM of the parser — the raw provider transport response — so the
committed fixture is the thing we do not control. The replay tests in CI assert
the adapter's emitted events against these. azure_openai / google_vertex have no
creds and are *seeded* (see ``seed_*`` below), flagged ``captured_at:
pending-creds``.
"""

from __future__ import annotations

import os
import sys
import json
import base64
import datetime as _dt
from typing import Any, Dict, List, Callable, Optional
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tests.instrument._recorded import (  # noqa: E402
    RECORDED_ROOT,
    scrub,
)

# Headers worth keeping (scrubbed) for shape realism / debugging.
_KEEP_HEADERS = frozenset({"content-type", "x-request-id", "request-id", "anthropic-version", "openai-version"})


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def _provenance(
    provider: str, sdk_version: str, model: str, scenario: str, captured_at: Optional[str] = None
) -> Dict[str, Any]:
    return {
        "provider": provider,
        "sdk_version": sdk_version,
        "model": model,
        "scenario": scenario,
        "captured_at": captured_at or _now(),
    }


def _write(adapter: str, scenario: str, fixture: Dict[str, Any]) -> None:
    fixture = scrub(fixture)
    out = RECORDED_ROOT / adapter / f"{scenario}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(fixture, f, indent=2, sort_keys=False)
        f.write("\n")
    print(f"  wrote {out.relative_to(RECORDED_ROOT.parent.parent)}")


# ---------------------------------------------------------------------------
# Recording httpx transport — real request, captured response, re-materialized
# ---------------------------------------------------------------------------


class _RecordingTransport(httpx.BaseTransport):
    """Performs the real request, captures the response body, and returns a
    fresh response the SDK can still read (the original stream is consumed by
    the capture)."""

    def __init__(self) -> None:
        self._real = httpx.HTTPTransport()
        self.interactions: List[Dict[str, Any]] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        request.read()
        response = self._real.handle_request(request)
        raw = response.read()
        body_json: Any = None
        body_text: Optional[str] = None
        try:
            body_json = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            body_text = raw.decode("utf-8", errors="replace")
        headers = {k: v for k, v in response.headers.items() if k.lower() in _KEEP_HEADERS}
        self.interactions.append(
            {
                "request": {"method": request.method, "path": request.url.path},
                "response": {
                    "status_code": response.status_code,
                    "json": body_json,
                    "text": body_text,
                    "headers": headers,
                },
            }
        )
        # ``raw`` is already decompressed by ``response.read()``; hand the SDK a
        # response whose headers don't claim an encoding/length it no longer has.
        passthrough = httpx.Headers(
            [
                (k, v)
                for k, v in response.headers.items()
                if k.lower() not in ("content-encoding", "content-length", "transfer-encoding")
            ]
        )
        return httpx.Response(response.status_code, headers=passthrough, content=raw, request=request)


def _http_fixture(
    provider: str, sdk_version: str, model: str, scenario: str, interactions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    return {
        "provenance": _provenance(provider, sdk_version, model, scenario),
        "transport": "http",
        "interactions": interactions,
    }


# ---------------------------------------------------------------------------
# Provider captures
# ---------------------------------------------------------------------------


def capture_openai() -> None:
    import openai

    rec = _RecordingTransport()
    client = openai.OpenAI(http_client=httpx.Client(transport=rec))
    model = os.environ.get("LL_OPENAI_MODEL", "gpt-4o-mini")

    # default
    client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with exactly: pong"}],
        temperature=0,
        max_tokens=8,
    )
    _write("openai", "default", _http_fixture("openai", openai.__version__, model, "default", rec.interactions[-1:]))

    # tool_call
    rec2 = _RecordingTransport()
    client2 = openai.OpenAI(http_client=httpx.Client(transport=rec2))
    client2.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "What's the weather in Paris? Use the tool."}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        tool_choice="required",
        temperature=0,
        max_tokens=64,
    )
    _write(
        "openai", "tool_call", _http_fixture("openai", openai.__version__, model, "tool_call", rec2.interactions[-1:])
    )


def capture_anthropic() -> None:
    import anthropic

    rec = _RecordingTransport()
    client = anthropic.Anthropic(http_client=httpx.Client(transport=rec))
    model = os.environ.get("LL_ANTHROPIC_MODEL", "claude-3-5-haiku-20241022")

    client.messages.create(
        model=model,
        max_tokens=16,
        messages=[{"role": "user", "content": "Reply with exactly: pong"}],
    )
    _write(
        "anthropic",
        "default",
        _http_fixture("anthropic", anthropic.__version__, model, "default", rec.interactions[-1:]),
    )

    rec2 = _RecordingTransport()
    client2 = anthropic.Anthropic(http_client=httpx.Client(transport=rec2))
    client2.messages.create(
        model=model,
        max_tokens=256,
        tools=[
            {
                "name": "get_weather",
                "description": "Get the weather for a city",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ],
        tool_choice={"type": "any"},
        messages=[{"role": "user", "content": "What's the weather in Paris? Use the tool."}],
    )
    _write(
        "anthropic",
        "tool_call",
        _http_fixture("anthropic", anthropic.__version__, model, "tool_call", rec2.interactions[-1:]),
    )


def capture_bedrock() -> None:
    import boto3

    region = os.environ.get("AWS_REGION", "us-east-1")
    client = boto3.client("bedrock-runtime", region_name=region)
    model = os.environ.get("LL_BEDROCK_MODEL", "amazon.nova-micro-v1:0")

    # converse (the good path for Nova)
    converse = client.converse(
        modelId=model,
        messages=[{"role": "user", "content": [{"text": "Reply with exactly: pong"}]}],
        inferenceConfig={"maxTokens": 16, "temperature": 0},
    )
    converse.pop("ResponseMetadata", None)
    _write(
        "bedrock",
        "converse",
        {
            "provenance": _provenance("aws_bedrock", boto3.__version__, model, "converse"),
            "transport": "boto3",
            "service": "bedrock-runtime",
            "operation": "converse",
            "response": converse,
        },
    )

    # invoke_model (Nova messages-v1 body)
    body = json.dumps(
        {
            "messages": [{"role": "user", "content": [{"text": "Reply with exactly: pong"}]}],
            "inferenceConfig": {"maxTokens": 16, "temperature": 0},
        }
    )
    resp = client.invoke_model(modelId=model, body=body, contentType="application/json", accept="application/json")
    raw = resp["body"].read()
    _write(
        "bedrock",
        "invoke_model",
        {
            "provenance": _provenance("aws_bedrock", boto3.__version__, model, "invoke_model"),
            "transport": "boto3",
            "service": "bedrock-runtime",
            "operation": "invoke_model",
            "response": {"body_b64": base64.b64encode(raw).decode(), "content_type": "application/json"},
        },
    )


def capture_bedrock_agents() -> None:
    import boto3

    region = os.environ.get("AWS_REGION", "us-east-1")
    agent_id = os.environ["BEDROCK_AGENT_ID"]
    alias_id = os.environ["BEDROCK_AGENT_ALIAS_ID"]
    client = boto3.client("bedrock-agent-runtime", region_name=region)

    resp = client.invoke_agent(
        agentId=agent_id,
        agentAliasId=alias_id,
        sessionId="record-corpus-session",
        inputText="What is 2 + 2? Answer with just the number.",
        enableTrace=True,
    )
    events: List[Dict[str, Any]] = []
    for event in resp["completion"]:
        events.append(_jsonable(event))
    request_id = resp.get("ResponseMetadata", {}).get("RequestId", "")
    _write(
        "bedrock_agents",
        "default",
        {
            "provenance": _provenance("bedrock_agents", boto3.__version__, "amazon.nova-micro-v1:0", "default"),
            "transport": "eventstream",
            "service": "bedrock-agent-runtime",
            "operation": "InvokeAgent",
            "request_id": request_id,
            "events": events,
        },
    )


def _object_fixture(
    provider: str, sdk: str, sdk_version: str, model: str, scenario: str, response: Dict[str, Any]
) -> Dict[str, Any]:
    """For adapters whose parser consumes a response OBJECT (not an HTTP body):
    record the object's serialized form (``model_dump``) as the upstream; the
    replay test reconstructs the real SDK type from it and stubs the wrapped
    call."""
    return {
        "provenance": _provenance(provider, sdk_version, model, scenario),
        "transport": "object",
        "sdk": sdk,
        "response": response,
    }


def capture_ollama() -> None:
    import ollama

    host = os.environ.get("OLLAMA_HOST")
    model = os.environ.get("OLLAMA_MODEL", "llama3:8b")
    client = ollama.Client(host=host)
    resp = client.chat(
        model=model,
        messages=[{"role": "user", "content": "Reply with exactly: pong"}],
        options={"num_predict": 8, "temperature": 0},
    )
    # ollama.chat returns a ChatResponse pydantic object; record its dump.
    response = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
    _write(
        "ollama",
        "default",
        _object_fixture("ollama", "ollama", getattr(ollama, "__version__", "?"), model, "default", response),
    )


def capture_litellm() -> None:
    import litellm

    model = os.environ.get("LL_LITELLM_MODEL", "gpt-4o-mini")
    resp = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": "Reply with exactly: pong"}],
        temperature=0,
        max_tokens=8,
    )
    response = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
    _write(
        "litellm",
        "default",
        _object_fixture("litellm", "litellm", getattr(litellm, "__version__", "?"), model, "default", response),
    )


def _jsonable(obj: Any) -> Any:
    """Make a boto3 completion event JSON-serializable (bytes -> base64 dict,
    datetime -> isoformat)."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, bytes):
        return {"__bytes_b64__": base64.b64encode(obj).decode()}
    if isinstance(obj, _dt.datetime):
        return obj.isoformat()
    return obj


def capture_agentforce() -> None:
    from unittest.mock import Mock

    import layerlens.instrument.adapters.frameworks.agentforce as af

    rec = _RecordingTransport()
    real_httpx = af.httpx

    class _RecordingShim:
        def Client(self, **kwargs: Any) -> Any:
            kwargs.pop("transport", None)
            return real_httpx.Client(transport=rec, timeout=kwargs.get("timeout", 30.0))

        def __getattr__(self, name: str) -> Any:
            return getattr(real_httpx, name)

    af.httpx = _RecordingShim()
    try:
        adapter = af.AgentforceAdapter(Mock())
        adapter.connect(
            credentials={
                "client_id": os.environ["SF_CLIENT_ID"],
                "client_secret": os.environ["SF_CLIENT_SECRET"],
                "instance_url": os.environ["SF_INSTANCE_URL"],
            }
        )
        adapter.import_sessions(limit=2)
        adapter.disconnect()
    finally:
        af.httpx = real_httpx

    _write(
        "agentforce",
        "default",
        _http_fixture("agentforce", "salesforce-stdm", "ssot__AiAgent*__dlm", "default", rec.interactions),
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

CAPTURES: Dict[str, Callable[[], None]] = {
    "openai": capture_openai,
    "anthropic": capture_anthropic,
    "bedrock": capture_bedrock,
    "bedrock_agents": capture_bedrock_agents,
    "ollama": capture_ollama,
    "litellm": capture_litellm,
    "agentforce": capture_agentforce,
}


def main(argv: List[str]) -> int:
    if os.environ.get("LAYERLENS_RECORD") != "1":
        print("refusing to record: set LAYERLENS_RECORD=1 (offline capture, real creds, spend).", file=sys.stderr)
        return 2
    targets = argv or sorted(CAPTURES)
    unknown = [t for t in targets if t not in CAPTURES]
    if unknown:
        print(f"unknown capture targets: {unknown}; known: {sorted(CAPTURES)}", file=sys.stderr)
        return 2
    for name in targets:
        print(f"capturing {name} ...")
        CAPTURES[name]()
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
