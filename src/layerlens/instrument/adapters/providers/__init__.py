"""LLM provider adapters for the LayerLens Instrument layer.

Each provider adapter wraps a single LLM SDK (OpenAI, Anthropic, Bedrock,
Vertex, Cohere, Mistral, Ollama, LiteLLM, Azure OpenAI) so direct provider
calls outside any agent framework still emit ``model.invoke`` events.

Adapters are loaded on demand via :class:`AdapterRegistry`; importing this
package does NOT import any provider SDK.
"""

from __future__ import annotations
