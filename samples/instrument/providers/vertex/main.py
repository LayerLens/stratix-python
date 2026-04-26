"""Sample: instrument the real Vertex AI client with the LayerLens adapter.

Runs a single ``generate_content`` call through ``VertexAdapter`` with
an ``HttpEventSink`` pointed at atlas-app. Every event the adapter
emits (``model.invoke``, ``cost.record``, optional ``tool.call``) is
shipped to the platform's telemetry ingest endpoint.

Two run modes:

* **Live** (default when credentials are present): hits real Vertex AI
  and requires ``google-cloud-aiplatform`` plus a Service-Account JSON
  (``GOOGLE_APPLICATION_CREDENTIALS``) or Application Default
  Credentials.
* **Mock** (``LAYERLENS_VERTEX_SAMPLE_MODE=mock``): swaps in a stubbed
  response so the sample runs end-to-end with neither the SDK nor a
  GCP project. Useful for CI smoke tests and local trace inspection.

Required environment (live mode):

* ``GOOGLE_APPLICATION_CREDENTIALS`` — absolute path to your SA-JSON
  key file. ADC also works (set by ``gcloud auth application-default
  login``).
* ``GOOGLE_CLOUD_PROJECT`` — your GCP project id.
* ``GOOGLE_CLOUD_REGION`` — Vertex region (defaults to ``us-central1``).
* ``LAYERLENS_STRATIX_API_KEY`` — your LayerLens API key (optional;
  defaults to anonymous if unset).
* ``LAYERLENS_STRATIX_BASE_URL`` — atlas-app base URL (optional;
  defaults to ``https://api.layerlens.ai/api/v1``).

Run::

    pip install 'layerlens[providers-vertex]'
    python -m samples.instrument.providers.vertex.main           # live
    LAYERLENS_VERTEX_SAMPLE_MODE=mock \\
        python -m samples.instrument.providers.vertex.main       # mock
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import Any

from layerlens.instrument.adapters._base import CaptureConfig
from layerlens.instrument.adapters.providers.vertex import VertexAdapter

# ``layerlens.instrument.transport`` lands in atlas-app M1.B. Until then,
# we fall back to a no-op sink so the sample remains runnable end-to-end.
try:
    from layerlens.instrument.transport.sink_http import (
        HttpEventSink,  # type: ignore[import-not-found,unused-ignore]
    )
except ImportError:  # pragma: no cover - exercised only on pre-M1.B SDKs.

    class HttpEventSink:  # type: ignore[no-redef]
        """No-op sink used until ``transport.sink_http`` ships.

        Implements the :class:`EventSink` ABI: ``send`` is the only
        required call from the adapter's ``_emit_*`` helpers, and
        ``close`` is invoked by the sample's ``finally`` block.
        """

        def __init__(self, **_: Any) -> None:
            pass

        def send(
            self,
            event_type: str,
            payload: Any,
            timestamp_ns: int,
        ) -> None:
            del event_type, payload, timestamp_ns

        def close(self) -> None:
            pass


def _make_mock_client(model_name: str = "gemini-1.5-pro") -> Any:
    """Return an object that mimics ``GenerativeModel`` for offline runs."""

    def _stub_generate(*_args: Any, **_kwargs: Any) -> Any:
        part = SimpleNamespace(
            text="The sky is blue because shorter wavelengths scatter more.",
            function_call=None,
        )
        candidate = SimpleNamespace(
            content=SimpleNamespace(parts=[part]),
            finish_reason=SimpleNamespace(name="STOP"),
        )
        usage = SimpleNamespace(
            prompt_token_count=14,
            candidates_token_count=12,
            total_token_count=26,
            thoughts_token_count=None,
        )
        return SimpleNamespace(candidates=[candidate], usage_metadata=usage)

    client = SimpleNamespace(model_name=model_name)
    client.generate_content = _stub_generate
    return client


def _make_live_client(model_name: str) -> Any:
    """Return a real Vertex ``GenerativeModel`` instance."""
    try:
        import vertexai  # type: ignore[import-not-found,unused-ignore]
        from vertexai.generative_models import (  # type: ignore[import-not-found,unused-ignore]
            GenerativeModel,
        )
    except ImportError as exc:
        raise SystemExit(
            "google-cloud-aiplatform is not installed. Install with:\n"
            "    pip install 'layerlens[providers-vertex]'"
        ) from exc

    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")
    if not project:
        raise SystemExit("GOOGLE_CLOUD_PROJECT is not set; cannot run live sample.")
    vertexai.init(project=project, location=location)
    return GenerativeModel(model_name)


def main() -> int:
    mode = os.environ.get("LAYERLENS_VERTEX_SAMPLE_MODE", "auto").lower()
    if mode == "auto":
        # Auto: prefer mock when no credentials are visible, so plain
        # `python -m samples.instrument.providers.vertex.main` always
        # exits successfully on a fresh checkout.
        if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") and not os.environ.get(
            "GOOGLE_CLOUD_PROJECT"
        ):
            mode = "mock"
        else:
            mode = "live"

    model_name = os.environ.get("LAYERLENS_VERTEX_MODEL", "gemini-1.5-pro")

    if mode == "mock":
        print(f"[mock] Stubbing Vertex {model_name} response...")
        client = _make_mock_client(model_name=model_name)
    else:
        print(f"[live] Calling Vertex {model_name}...")
        client = _make_live_client(model_name)

    sink = HttpEventSink(
        adapter_name="vertex",
        path="/telemetry/spans",
        max_batch=10,
        flush_interval_s=1.0,
    )

    adapter = VertexAdapter(capture_config=CaptureConfig.standard())
    adapter.add_sink(sink)
    adapter.connect()
    adapter.connect_client(client)

    try:
        response = client.generate_content("Why is the sky blue?")
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            parts = getattr(candidates[0].content, "parts", None) or []
            text = "\n".join(getattr(p, "text", "") or "" for p in parts).strip()
            print(f"Response: {text}")
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            print(
                f"Tokens — prompt: {usage.prompt_token_count}, "
                f"candidates: {usage.candidates_token_count}, "
                f"total: {usage.total_token_count}"
            )
    finally:
        sink.close()
        adapter.disconnect()

    print("Telemetry shipped. Check the LayerLens dashboard adapter health page.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"Sample failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
