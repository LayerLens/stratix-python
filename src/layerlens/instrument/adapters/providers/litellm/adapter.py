"""LiteLLM provider adapter.

LiteLLM is a multi-provider router: a single ``litellm.completion()`` (or
``litellm.acompletion()``) call is dispatched to one of ~100 providers
(OpenAI, Anthropic, Bedrock, Vertex AI, Cohere, Ollama, Together, Groq,
HuggingFace, ...). Rather than monkey-patching every provider client,
this adapter installs a single :class:`LayerLensLiteLLMCallback` into
LiteLLM's callback registry and lets LiteLLM's own dispatch fire it for
both sync and async paths.

Ported from
``ateam/stratix/sdk/python/adapters/llm_providers/litellm_adapter.py``
(see PR description for the M3 fan-out plan).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from layerlens.instrument.adapters._base.adapter import AdapterStatus
from layerlens.instrument.adapters._base.capture import CaptureConfig
from layerlens.instrument.adapters.providers._base.provider import LLMProviderAdapter
from layerlens.instrument.adapters.providers.litellm.callback import LayerLensLiteLLMCallback

logger = logging.getLogger(__name__)


class LiteLLMAdapter(LLMProviderAdapter):
    """LayerLens adapter for the LiteLLM router.

    Uses LiteLLM's callback handler pattern instead of monkey-patching
    so the adapter does not interfere with LiteLLM's routing, fallback,
    or retry behaviour. Auto-detects the underlying provider from the
    model-string prefix (see
    :func:`layerlens.instrument.adapters.providers.litellm.routing.detect_provider`).

    Usage::

        import litellm
        from layerlens.instrument.adapters.providers.litellm import LiteLLMAdapter

        adapter = LiteLLMAdapter()
        adapter.connect()  # registers the callback

        # Sync — provider routed by the model string prefix.
        litellm.completion(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
        )

        # Async — same callback handles ``acompletion``.
        await litellm.acompletion(
            model="anthropic/claude-3-5-sonnet",
            messages=[{"role": "user", "content": "hi"}],
        )

        adapter.disconnect()
    """

    FRAMEWORK = "litellm"
    VERSION = "0.1.0"

    def __init__(
        self,
        stratix: Any = None,
        capture_config: Optional[CaptureConfig] = None,
    ) -> None:
        super().__init__(stratix=stratix, capture_config=capture_config)
        self._callback: Optional[LayerLensLiteLLMCallback] = None

    def connect(self) -> None:
        """Register the LayerLens callback with LiteLLM.

        Appends the callback instance to ``litellm.callbacks`` (and the
        ``success_callback`` / ``failure_callback`` lists for the proxy
        path). On ``ImportError`` the adapter still marks itself as
        ``connected`` but in :attr:`AdapterStatus.DEGRADED` so the
        registry can surface a clear "litellm not installed" diagnostic
        without crashing the host process.
        """
        self._callback = LayerLensLiteLLMCallback(self)
        try:
            import litellm  # type: ignore[import-not-found,import-untyped,unused-ignore]

            # ``litellm.callbacks`` is typed as ``list[Callable]`` upstream
            # but accepts handler instances by convention. Cast through
            # ``Any`` at the boundary to satisfy strict type checkers.
            callbacks: Any = getattr(litellm, "callbacks", None)
            if callbacks is None:
                callbacks = []
                litellm.callbacks = callbacks
            callbacks.append(self._callback)

            version = getattr(litellm, "__version__", None)
            self._framework_version = str(version) if version is not None else None
            self._connected = True
            self._status = AdapterStatus.HEALTHY
        except ImportError:
            logger.warning("LiteLLM not installed; adapter in degraded mode")
            self._connected = True
            self._status = AdapterStatus.DEGRADED

    def disconnect(self) -> None:
        """Remove the LayerLens callback from LiteLLM."""
        if self._callback:
            try:
                import litellm  # type: ignore[import-not-found,import-untyped,unused-ignore]

                callbacks: Any = getattr(litellm, "callbacks", None)
                if callbacks is not None and self._callback in callbacks:
                    callbacks.remove(self._callback)
            except ImportError:
                pass
            self._callback = None
        self._connected = False
        self._status = AdapterStatus.DISCONNECTED

    def connect_client(self, client: Any) -> Any:
        """LiteLLM uses a module-level callback registry — no client to wrap."""
        return client

    @staticmethod
    def _detect_framework_version() -> Optional[str]:
        """Return the installed ``litellm.__version__`` or ``None``."""
        try:
            import litellm  # type: ignore[import-not-found,import-untyped,unused-ignore]

            version = getattr(litellm, "__version__", None)
            return str(version) if version is not None else None
        except ImportError:
            return None


__all__ = ["LiteLLMAdapter"]
