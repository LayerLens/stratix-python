from __future__ import annotations

from typing import Any

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

_original_completion = None
_original_acompletion = None


def instrument_litellm() -> None:
    try:
        import litellm
    except ImportError as err:
        raise ImportError(
            "The 'litellm' package is required for LiteLLM instrumentation. Install it with: pip install litellm"
        ) from err

    global _original_completion, _original_acompletion

    if _original_completion is None:
        _original_completion = litellm.completion
        orig_sync = _original_completion

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

    if _original_acompletion is None:
        _original_acompletion = litellm.acompletion
        orig_async = _original_acompletion

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


def uninstrument_litellm() -> None:
    global _original_completion, _original_acompletion
    try:
        import litellm
    except ImportError:
        return

    if _original_completion is not None:
        litellm.completion = _original_completion
        _original_completion = None
    if _original_acompletion is not None:
        litellm.acompletion = _original_acompletion
        _original_acompletion = None
