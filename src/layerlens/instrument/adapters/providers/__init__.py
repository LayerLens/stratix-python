"""LLM provider adapters for the LayerLens Instrument layer.

Each provider adapter wraps a vendor SDK client to intercept API calls
and emit ``model.invoke``, ``cost.record``, ``tool.call``, and
``policy.violation`` events through the LayerLens telemetry pipeline.

Adapters available:

* ``openai_adapter`` — OpenAI Python SDK (``openai >= 1.30``)
* ``anthropic_adapter`` — Anthropic Python SDK (``anthropic >= 0.30``)
* ``azure_openai_adapter`` — Azure OpenAI (``openai >= 1.30``)
* ``bedrock_adapter`` — AWS Bedrock (``boto3``)
* ``google_vertex_adapter`` — Google Vertex AI (``google-cloud-aiplatform``)
* ``ollama_adapter`` — Ollama (``ollama``)
* ``litellm_adapter`` — LiteLLM proxy (``litellm``)
* ``cohere_adapter`` — Cohere (``cohere`` >= 5)
* ``mistral_adapter`` — Mistral AI (``mistralai`` >= 1)

Importing this package does NOT import any vendor SDK; modules are
loaded on demand via :class:`AdapterRegistry`.
"""

from __future__ import annotations
