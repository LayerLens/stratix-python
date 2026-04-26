"""AWS Bedrock LLM Provider Adapter (M3 fan-out, ADP-057).

Wraps the four primary entry points on the ``bedrock-runtime`` boto3
client:

* ``invoke_model`` — native per-family request body (Anthropic Messages,
  Meta prompt-completion, Cohere generations, Amazon Titan, AI21,
  Mistral). Token / content / finish-reason extraction is dispatched on
  the family detected from ``modelId``.
* ``invoke_model_with_response_stream`` — same dispatch, but the
  response is a streaming envelope; only the request-side ``model.invoke``
  is emitted at wrap time (content extraction during stream iteration is
  deferred).
* ``converse`` — the unified Bedrock Converse envelope.
* ``converse_stream`` — streaming Converse.

Bedrock authentication is whatever boto3 is already configured for —
IAM role, static keys via ``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY``,
or SSO credentials — the adapter does not touch ``boto3.Session``.

Ported from ``ateam/stratix/sdk/python/adapters/llm_providers/bedrock_adapter.py``.
"""

from __future__ import annotations

import json
import time
import logging
from typing import Any, Dict, List, Optional

from layerlens.instrument.adapters._base.capture import CaptureConfig
from layerlens.instrument.adapters.providers.bedrock.body import RereadableBody
from layerlens.instrument.adapters.providers._base.pricing import BEDROCK_PRICING
from layerlens.instrument.adapters.providers._base.provider import LLMProviderAdapter
from layerlens.instrument.adapters.providers.bedrock.family import detect_provider_family
from layerlens.instrument.adapters.providers.bedrock.extract import (
    extract_invoke_usage,
    build_invoke_metadata,
    extract_invoke_output,
    extract_converse_usage,
    extract_converse_output,
    extract_invoke_messages,
)

logger = logging.getLogger(__name__)


class AWSBedrockAdapter(LLMProviderAdapter):
    """LayerLens adapter for AWS Bedrock (``bedrock-runtime``).

    Wraps ``invoke_model``, ``invoke_model_with_response_stream``,
    ``converse``, and ``converse_stream``. Parses ``modelId`` for
    provider-specific token, content, and finish-reason extraction.

    Example::

        import boto3
        from layerlens.instrument.adapters.providers.bedrock import AWSBedrockAdapter

        adapter = AWSBedrockAdapter()
        adapter.connect()

        client = boto3.client("bedrock-runtime", region_name="us-east-1")
        adapter.connect_client(client)

        client.converse(
            modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
            messages=[{"role": "user", "content": [{"text": "Hi"}]}],
        )
    """

    FRAMEWORK = "aws_bedrock"
    VERSION = "0.1.0"

    def __init__(
        self,
        layerlens: Any = None,
        capture_config: Optional[CaptureConfig] = None,
        *,
        stratix: Any = None,
    ) -> None:
        # ``stratix=`` kwarg is preserved for ateam call-sites — the
        # M1.B port renamed the public kwarg to ``layerlens=``. Either
        # form is accepted.
        target = layerlens if layerlens is not None else stratix
        super().__init__(stratix=target, capture_config=capture_config)

    # --- Connect / disconnect ---

    def connect_client(self, client: Any) -> Any:
        """Wrap a Bedrock runtime client with tracing.

        Each of the four supported methods is patched in place. The
        original is preserved on ``self._originals`` so :meth:`disconnect`
        can restore the client to its pre-instrumented state.
        """
        self._client = client

        if hasattr(client, "invoke_model"):
            original = client.invoke_model
            self._originals["invoke_model"] = original
            client.invoke_model = self._wrap_invoke_model(original)

        if hasattr(client, "converse"):
            original = client.converse
            self._originals["converse"] = original
            client.converse = self._wrap_converse(original)

        if hasattr(client, "invoke_model_with_response_stream"):
            original = client.invoke_model_with_response_stream
            self._originals["invoke_model_with_response_stream"] = original
            client.invoke_model_with_response_stream = self._wrap_invoke_stream(original)

        if hasattr(client, "converse_stream"):
            original = client.converse_stream
            self._originals["converse_stream"] = original
            client.converse_stream = self._wrap_converse_stream(original)

        return client

    def _restore_originals(self) -> None:
        if self._client is None:
            return
        for method_name, original in self._originals.items():
            try:
                setattr(self._client, method_name, original)
            except Exception:
                logger.warning("Could not restore %s", method_name)

    @staticmethod
    def _detect_framework_version() -> Optional[str]:
        try:
            import boto3  # type: ignore[import-untyped,unused-ignore]
        except ImportError:
            return None
        version = getattr(boto3, "__version__", None)
        return str(version) if version is not None else None

    # --- invoke_model ---

    def _wrap_invoke_model(self, original: Any) -> Any:
        adapter = self

        def traced_invoke(*args: Any, **kwargs: Any) -> Any:
            model_id = kwargs.get("modelId", "")
            start_ns = time.time_ns()

            input_messages = extract_invoke_messages(kwargs, model_id)

            try:
                response = original(*args, **kwargs)
            except Exception as exc:
                elapsed_ms = (time.time_ns() - start_ns) / 1_000_000
                try:
                    adapter._emit_model_invoke(
                        provider="aws_bedrock",
                        model=model_id,
                        latency_ms=elapsed_ms,
                        error=str(exc),
                        metadata={"method": "invoke_model"},
                        input_messages=input_messages,
                    )
                    adapter._emit_provider_error("aws_bedrock", str(exc), model=model_id)
                except Exception:
                    logger.warning("Error emitting Bedrock error event", exc_info=True)
                raise

            try:
                elapsed_ms = (time.time_ns() - start_ns) / 1_000_000
                body = response.get("body") if isinstance(response, dict) else None
                body_data: Dict[str, Any] = {}
                if body is not None and hasattr(body, "read"):
                    body_bytes = body.read()
                    try:
                        body_data = json.loads(body_bytes)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        body_data = {}
                    # Re-wrap so the caller can still read the body.
                    response["body"] = RereadableBody(body_bytes)

                family = detect_provider_family(model_id)
                usage = extract_invoke_usage(body_data, family)
                output_message = extract_invoke_output(body_data, family)
                invoke_metadata = build_invoke_metadata(body_data, family)

                adapter._emit_model_invoke(
                    provider="aws_bedrock",
                    model=model_id,
                    usage=usage,
                    latency_ms=elapsed_ms,
                    metadata=invoke_metadata,
                    input_messages=input_messages,
                    output_message=output_message,
                )
                adapter._emit_cost_record(
                    model=model_id,
                    usage=usage,
                    provider="aws_bedrock",
                    pricing_table=BEDROCK_PRICING,
                )
            except Exception:
                logger.warning("Error emitting Bedrock invoke events", exc_info=True)

            return response

        traced_invoke._layerlens_original = original  # type: ignore[attr-defined]
        return traced_invoke

    # --- converse ---

    def _wrap_converse(self, original: Any) -> Any:
        adapter = self

        def traced_converse(*args: Any, **kwargs: Any) -> Any:
            model_id = kwargs.get("modelId", "")
            start_ns = time.time_ns()

            input_messages = adapter._normalize_messages(kwargs.get("messages"))

            try:
                response = original(*args, **kwargs)
            except Exception as exc:
                elapsed_ms = (time.time_ns() - start_ns) / 1_000_000
                try:
                    adapter._emit_model_invoke(
                        provider="aws_bedrock",
                        model=model_id,
                        latency_ms=elapsed_ms,
                        error=str(exc),
                        metadata={"method": "converse"},
                        input_messages=input_messages,
                    )
                    adapter._emit_provider_error("aws_bedrock", str(exc), model=model_id)
                except Exception:
                    logger.warning("Error emitting Bedrock converse error", exc_info=True)
                raise

            try:
                elapsed_ms = (time.time_ns() - start_ns) / 1_000_000
                usage = extract_converse_usage(response)
                output_message = extract_converse_output(response)

                converse_metadata: Dict[str, Any] = {"method": "converse"}
                stop_reason = response.get("stopReason")
                if stop_reason is not None:
                    converse_metadata["finish_reason"] = stop_reason
                resp_meta = response.get("ResponseMetadata", {})
                request_id = resp_meta.get("RequestId") if isinstance(resp_meta, dict) else None
                if request_id is not None:
                    converse_metadata["response_id"] = request_id

                adapter._emit_model_invoke(
                    provider="aws_bedrock",
                    model=model_id,
                    usage=usage,
                    latency_ms=elapsed_ms,
                    metadata=converse_metadata,
                    input_messages=input_messages,
                    output_message=output_message,
                )
                adapter._emit_cost_record(
                    model=model_id,
                    usage=usage,
                    provider="aws_bedrock",
                    pricing_table=BEDROCK_PRICING,
                )
            except Exception:
                logger.warning("Error emitting Bedrock converse events", exc_info=True)

            return response

        traced_converse._layerlens_original = original  # type: ignore[attr-defined]
        return traced_converse

    # --- invoke_model_with_response_stream ---

    def _wrap_invoke_stream(self, original: Any) -> Any:
        """Wrap ``invoke_model_with_response_stream``.

        ``output_message`` is intentionally not extracted here because
        the response is a stream — content is not available until the
        caller fully consumes the iterator.
        """
        adapter = self

        def traced_invoke_stream(*args: Any, **kwargs: Any) -> Any:
            model_id = kwargs.get("modelId", "")
            start_ns = time.time_ns()

            input_messages = extract_invoke_messages(kwargs, model_id)

            try:
                response = original(*args, **kwargs)
            except Exception as exc:
                elapsed_ms = (time.time_ns() - start_ns) / 1_000_000
                try:
                    adapter._emit_model_invoke(
                        provider="aws_bedrock",
                        model=model_id,
                        latency_ms=elapsed_ms,
                        error=str(exc),
                        metadata={"method": "invoke_model_with_response_stream"},
                        input_messages=input_messages,
                    )
                except Exception:
                    logger.warning("Error emitting Bedrock stream error", exc_info=True)
                raise

            try:
                elapsed_ms = (time.time_ns() - start_ns) / 1_000_000
                adapter._emit_model_invoke(
                    provider="aws_bedrock",
                    model=model_id,
                    latency_ms=elapsed_ms,
                    metadata={
                        "method": "invoke_model_with_response_stream",
                        "streaming": True,
                    },
                    input_messages=input_messages,
                )
            except Exception:
                logger.warning("Error emitting Bedrock stream events", exc_info=True)

            return response

        traced_invoke_stream._layerlens_original = original  # type: ignore[attr-defined]
        return traced_invoke_stream

    # --- converse_stream ---

    def _wrap_converse_stream(self, original: Any) -> Any:
        """Wrap ``converse_stream``."""
        adapter = self

        def traced_converse_stream(*args: Any, **kwargs: Any) -> Any:
            model_id = kwargs.get("modelId", "")
            start_ns = time.time_ns()

            input_messages = adapter._normalize_messages(kwargs.get("messages"))

            try:
                response = original(*args, **kwargs)
            except Exception as exc:
                elapsed_ms = (time.time_ns() - start_ns) / 1_000_000
                try:
                    adapter._emit_model_invoke(
                        provider="aws_bedrock",
                        model=model_id,
                        latency_ms=elapsed_ms,
                        error=str(exc),
                        metadata={"method": "converse_stream"},
                        input_messages=input_messages,
                    )
                except Exception:
                    logger.warning(
                        "Error emitting Bedrock converse_stream error", exc_info=True
                    )
                raise

            try:
                elapsed_ms = (time.time_ns() - start_ns) / 1_000_000
                adapter._emit_model_invoke(
                    provider="aws_bedrock",
                    model=model_id,
                    latency_ms=elapsed_ms,
                    metadata={"method": "converse_stream", "streaming": True},
                    input_messages=input_messages,
                )
            except Exception:
                logger.warning(
                    "Error emitting Bedrock converse_stream events", exc_info=True
                )

            return response

        traced_converse_stream._layerlens_original = original  # type: ignore[attr-defined]
        return traced_converse_stream

    # --- Convenience exports of the per-family helpers -----------------
    # Preserved as static methods so existing ateam call-sites that did
    # ``AWSBedrockAdapter._extract_invoke_usage(body, "anthropic")``
    # continue to work after the package split.

    @staticmethod
    def _extract_invoke_messages(
        kwargs: Dict[str, Any],
        model_id: str,
    ) -> Optional[List[Dict[str, str]]]:
        return extract_invoke_messages(kwargs, model_id)

    @staticmethod
    def _extract_invoke_output(
        body_data: Dict[str, Any],
        family: str,
    ) -> Optional[Dict[str, str]]:
        return extract_invoke_output(body_data, family)

    @staticmethod
    def _extract_converse_output(response: Dict[str, Any]) -> Optional[Dict[str, str]]:
        return extract_converse_output(response)

    @staticmethod
    def _extract_invoke_usage(body_data: Dict[str, Any], family: str) -> Any:
        return extract_invoke_usage(body_data, family)

    @staticmethod
    def _extract_converse_usage(response: Dict[str, Any]) -> Any:
        return extract_converse_usage(response)
