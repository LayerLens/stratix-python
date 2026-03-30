from __future__ import annotations

from typing import Any

from .._base import AdapterInfo, BaseAdapter
from .openai import _extract_output, _extract_response_meta
from ._base_provider import fail_llm_span, create_llm_span, finish_llm_span

_CAPTURE_PARAMS = frozenset(
    {
        "model",
        "temperature",
        "max_tokens",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "response_format",
    }
)


class LiteLLMProvider(BaseAdapter):
    def __init__(self) -> None:
        self._original_completion: Any = None
        self._original_acompletion: Any = None
        self._connected = False

    def connect(self, target: Any = None, **kwargs: Any) -> Any:  # noqa: ARG002
        try:
            import litellm
        except ImportError as err:
            raise ImportError(
                "The 'litellm' package is required for LiteLLM instrumentation. Install it with: pip install litellm"
            ) from err

        if self._original_completion is None:
            self._original_completion = litellm.completion
            orig_sync = self._original_completion

            def patched_completion(*args: Any, **kwargs: Any) -> Any:
                span, token = create_llm_span("litellm.completion", kwargs, _CAPTURE_PARAMS)
                if span is None:
                    return orig_sync(*args, **kwargs)
                try:
                    response = orig_sync(*args, **kwargs)
                    finish_llm_span(span, token, response, _extract_output, _extract_response_meta)
                    return response
                except Exception as exc:
                    fail_llm_span(span, token, exc)
                    raise

            litellm.completion = patched_completion

        if self._original_acompletion is None:
            self._original_acompletion = litellm.acompletion
            orig_async = self._original_acompletion

            async def patched_acompletion(*args: Any, **kwargs: Any) -> Any:
                span, token = create_llm_span("litellm.acompletion", kwargs, _CAPTURE_PARAMS)
                if span is None:
                    return await orig_async(*args, **kwargs)
                try:
                    response = await orig_async(*args, **kwargs)
                    finish_llm_span(span, token, response, _extract_output, _extract_response_meta)
                    return response
                except Exception as exc:
                    fail_llm_span(span, token, exc)
                    raise

            litellm.acompletion = patched_acompletion

        self._connected = True
        return target

    def disconnect(self) -> None:
        try:
            import litellm
        except ImportError:
            self._connected = False
            return

        if self._original_completion is not None:
            litellm.completion = self._original_completion
            self._original_completion = None
        if self._original_acompletion is not None:
            litellm.acompletion = self._original_acompletion
            self._original_acompletion = None

        self._connected = False

    def adapter_info(self) -> AdapterInfo:
        return AdapterInfo(
            name="litellm",
            adapter_type="provider",
            connected=self._connected,
        )


# --- Convenience API ---


def instrument_litellm() -> LiteLLMProvider:
    from .._registry import get, register

    existing = get("litellm")
    if existing is not None:
        existing.disconnect()
    provider = LiteLLMProvider()
    provider.connect()
    register("litellm", provider)
    return provider


def uninstrument_litellm() -> None:
    from .._registry import unregister

    unregister("litellm")
