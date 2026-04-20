from __future__ import annotations

from typing import Any, Dict
from urllib.parse import urlparse

from .openai import _CAPTURE_PARAMS, OpenAIProvider  # type: ignore[attr-defined]
from .pricing import AZURE_PRICING


class AzureOpenAIProvider(OpenAIProvider):
    """Azure OpenAI adapter.

    Reuses OpenAIProvider's extraction + monkey-patch targets (the Azure SDK
    uses the same ``chat.completions.create`` / ``responses.create`` / ``embeddings.create``
    surface) and layers on Azure-specific response metadata and pricing.
    """

    name = "azure_openai"
    capture_params = _CAPTURE_PARAMS
    pricing_table = AZURE_PRICING

    @staticmethod
    def extract_meta(response: Any) -> Dict[str, Any]:
        meta = OpenAIProvider.extract_meta(response)
        # Surface Azure-specific attributes when the SDK attaches them.
        for attr, key in (("api_version", "azure_api_version"), ("deployment", "azure_deployment")):
            val = getattr(response, attr, None)
            if val is not None:
                meta[key] = val
        return meta

    def connect(self, target: Any = None, **kwargs: Any) -> Any:
        result = super().connect(target, **kwargs)
        # Capture the client's base URL (stripped of query params) for trace metadata.
        endpoint = _scrubbed_endpoint(target)
        if endpoint is not None:
            self._endpoint = endpoint
        return result


def _scrubbed_endpoint(client: Any) -> str | None:
    """Return the client's endpoint without query string — never log api-keys."""
    url = (
        getattr(client, "base_url", None)
        or getattr(client, "_base_url", None)
        or getattr(client, "azure_endpoint", None)
    )
    if url is None:
        return None
    try:
        parsed = urlparse(str(url))
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")
    except Exception:
        return None


def instrument_azure_openai(client: Any) -> AzureOpenAIProvider:
    from .._registry import get, register

    existing = get("azure_openai")
    if existing is not None:
        existing.disconnect()
    provider = AzureOpenAIProvider()
    provider.connect(client)
    register("azure_openai", provider)
    return provider


def uninstrument_azure_openai() -> None:
    from .._registry import unregister

    unregister("azure_openai")
