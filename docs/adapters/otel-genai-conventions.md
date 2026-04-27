# OpenTelemetry GenAI Semantic Conventions — Adapter Coverage

> **Spec reference:** `docs/incubation-docs/adapter-framework/07-otel-genai-semantic-conventions.md`
> **Upstream:** https://opentelemetry.io/docs/specs/semconv/gen-ai/

Every LLM-call event emitted by every LayerLens adapter is automatically
stamped with the canonical OpenTelemetry GenAI semantic-convention
(`gen_ai.*`) attribute set in addition to LayerLens's own custom
attributes. This page documents the contract, the implementation, and
the per-adapter coverage matrix.

---

## 1. Contract

### 1.1 Additive

The stamping is **purely additive**. Every existing custom attribute
LayerLens has historically emitted (`provider`, `model`, `parameters`,
`prompt_tokens`, `completion_tokens`, `latency_ms`, `metadata`,
`framework`, …) continues to be emitted unchanged. The `gen_ai.*` keys
are added alongside them. Dashboards, queries, sinks, and downstream
consumers built against the legacy attribute set keep working
indefinitely.

### 1.2 Required attributes

For every `model.invoke` (and equivalent) event, the helper guarantees
the following keys are present:

| Key                          | Type     | Source                                         |
|------------------------------|----------|------------------------------------------------|
| `gen_ai.system`              | string   | adapter `FRAMEWORK` / explicit override        |
| `gen_ai.provider.name`       | string   | mirror of `gen_ai.system`                      |
| `gen_ai.operation.name`      | string   | `chat` (default), `embeddings`, `text_completion`, `generate_content` |

### 1.3 Conditional attributes

Populated when source data is available; silently omitted otherwise
(the helper never invents defaults):

| Key                                              | Source                                  |
|--------------------------------------------------|-----------------------------------------|
| `gen_ai.request.model`                           | request kwargs / payload `model`        |
| `gen_ai.request.max_tokens`                      | request kwargs / `parameters`           |
| `gen_ai.request.temperature`                     | request kwargs / `parameters`           |
| `gen_ai.request.top_p`                           | request kwargs / `parameters`           |
| `gen_ai.request.top_k`                           | request kwargs / `parameters`           |
| `gen_ai.request.frequency_penalty`               | request kwargs / `parameters`           |
| `gen_ai.request.presence_penalty`                | request kwargs / `parameters`           |
| `gen_ai.request.stop_sequences`                  | request kwargs (string → 1-element list) |
| `gen_ai.request.seed`                            | request kwargs                          |
| `gen_ai.request.choice.count`                    | request kwargs `n`                      |
| `gen_ai.request.encoding_formats`                | request kwargs `encoding_format`        |
| `gen_ai.response.id`                             | response object `.id`                   |
| `gen_ai.response.model`                          | response object `.model`                |
| `gen_ai.response.finish_reasons`                 | response object `.finish_reason` (string → 1-element list) |
| `gen_ai.usage.input_tokens`                      | payload `prompt_tokens` / `token_usage` |
| `gen_ai.usage.output_tokens`                     | payload `completion_tokens` / `token_usage` |
| `gen_ai.openai.response.system_fingerprint`      | response `.system_fingerprint`          |
| `gen_ai.openai.response.service_tier`            | response `.service_tier`                |
| `gen_ai.openai.request.service_tier`             | request kwargs `service_tier`           |
| `gen_ai.openai.request.response_format`          | request kwargs `response_format`        |
| `gen_ai.anthropic.cache_creation_input_tokens`   | response `.usage.cache_creation_input_tokens` |
| `gen_ai.anthropic.cache_read_input_tokens`       | response `.usage.cache_read_input_tokens` |
| `aws.bedrock.guardrail.id`                       | request kwargs `guardrailConfig.guardrailIdentifier` |
| `aws.bedrock.knowledge_base.id`                  | request kwargs `knowledgeBaseId`        |
| `aws.bedrock.agent.id`                           | request kwargs `agentId`                |

For tool-call events:

| Key                  | Source                |
|----------------------|-----------------------|
| `gen_ai.tool.name`   | `tool.name`           |
| `gen_ai.tool.call.id`| `tool.id`             |

### 1.4 Idempotent

Multiple calls to `stamp_genai_attributes` on the same payload write
the same key-value pairs and never multiply attributes — adapters MAY
re-stamp at any layer without risk.

### 1.5 Safe under partial data

Adapter emissions on the error path frequently lack a usage object,
response object, or even a model name. The helper writes only what is
present and never raises; the centralized hook additionally swallows
all stamping exceptions at DEBUG log level so the circuit-breaker path
keeps running.

---

## 2. Implementation

### 2.1 Helper module

`src/layerlens/instrument/adapters/_base/genai_semconv.py` provides:

```python
detect_gen_ai_system(name: str | None) -> str
stamp_genai_attributes(
    payload: dict,
    request_kwargs: Mapping | None = None,
    response_obj: Any = None,
    *,
    system: str | None = None,
    operation: str | None = None,
) -> dict
```

Plus module-level constants for every spec-defined attribute key.

### 2.2 Centralized hook

`BaseAdapter.emit_dict_event` calls `_stamp_gen_ai_attributes` whenever
the event type is one of:

| Event type          | Operation override (None ⇒ auto-detect) |
|---------------------|------------------------------------------|
| `model.invoke`      | `None` (auto: chat / embeddings)         |
| `embedding.create`  | `embeddings`                             |
| `model.request`     | `None`                                   |
| `model.response`    | `None`                                   |

This guarantees every framework adapter is covered by default — any
adapter going through the standard emission path receives `gen_ai.*`
stamping for free.

### 2.3 Provider-side explicit stamping

`LLMProviderAdapter._emit_model_invoke` additionally calls
`stamp_genai_attributes` directly with the concrete `provider` argument
(`"openai"`, `"anthropic"`, …). This gives the most accurate
`gen_ai.system` resolution because the provider name passed at the
call site is the canonical source of truth for that adapter's identity.

`LLMProviderAdapter._emit_tool_calls` populates `gen_ai.tool.name` and
`gen_ai.tool.call.id` directly into the tool-call payload.

---

## 3. Per-Adapter Coverage

### 3.1 LLM Providers (9)

All providers route through `LLMProviderAdapter._emit_model_invoke`,
which stamps both via the per-adapter explicit call AND via the
centralized hook. Double-stamping is safe (idempotent).

| Adapter             | FRAMEWORK             | gen_ai.system          | Notes                                              |
|---------------------|-----------------------|------------------------|----------------------------------------------------|
| OpenAIAdapter       | `openai`              | `openai`               | Captures system_fingerprint, service_tier, response_format |
| AzureOpenAIAdapter  | `azure_openai`        | `azure.openai`         | Same OpenAI-specific keys                          |
| AnthropicAdapter    | `anthropic`           | `anthropic`            | Captures cache creation/read tokens                |
| AWSBedrockAdapter   | `aws_bedrock`         | `aws.bedrock`          | Captures guardrail / knowledge-base / agent IDs    |
| GoogleVertexAdapter | `google_vertex`       | `gcp.vertex_ai`        |                                                    |
| CohereAdapter       | `cohere`              | `cohere`               |                                                    |
| MistralAdapter      | `mistral`             | `mistral_ai`           |                                                    |
| OllamaAdapter       | `ollama`              | `ollama`               |                                                    |
| LiteLLMAdapter      | `litellm`             | `litellm`              | Re-detects underlying provider per-call            |

### 3.2 Framework Adapters (16)

All framework adapters route through `BaseAdapter.emit_dict_event` and
are covered by the centralized hook. The `gen_ai.system` value is
resolved from the adapter's `FRAMEWORK` constant via
`detect_gen_ai_system`, with payload `provider` taking precedence when
present (e.g. langchain wrapping a known provider).

| Adapter                | FRAMEWORK                  | Default gen_ai.system | Stamping path                          |
|------------------------|----------------------------|-----------------------|----------------------------------------|
| AgnoLifecycleAdapter   | `agno`                     | `_OTHER`              | centralized hook                       |
| AutogenAdapter         | `autogen`                  | `_OTHER`              | centralized hook                       |
| BedrockAgentsAdapter   | `bedrock_agents`           | `aws.bedrock`         | centralized hook                       |
| CrewAIAdapter          | `crewai`                   | `_OTHER`              | centralized hook                       |
| GoogleADKAdapter       | `google_adk`               | `gcp.gemini`          | centralized hook                       |
| LangChainCallback      | `langchain`                | `_OTHER` (per-call)   | centralized hook (uses payload provider) |
| LangFuseAdapter        | `langfuse`                 | `_OTHER`              | centralized hook                       |
| LangGraphAdapter       | `langgraph`                | `_OTHER`              | centralized hook                       |
| LlamaIndexAdapter      | `llama_index`              | `_OTHER`              | centralized hook                       |
| MSAgentFrameworkAdapter| `ms_agent_framework`       | `_OTHER`              | centralized hook                       |
| OpenAIAgentsAdapter    | `openai_agents`            | `openai`              | centralized hook                       |
| PydanticAIAdapter      | `pydantic_ai`              | `_OTHER`              | centralized hook                       |
| SemanticKernelAdapter  | `semantic_kernel`          | `_OTHER`              | centralized hook                       |
| SmolagentsAdapter      | `smolagents`               | `_OTHER`              | centralized hook                       |
| StrandsAdapter         | `strands`                  | `_OTHER`              | centralized hook                       |
| AgentforceAdapter      | `salesforce_agentforce`    | `_OTHER`              | centralized hook                       |

### 3.3 Embedding Adapters

| Adapter                  | Event type         | Operation     | Stamping path     |
|--------------------------|--------------------|---------------|-------------------|
| EmbeddingAdapter         | `embedding.create` | `embeddings`  | centralized hook  |
| VectorStoreAdapter       | (vector ops)       | n/a           | n/a               |

`embedding.create` events are stamped with `gen_ai.operation.name = embeddings`
per spec §3.1 span naming convention.

---

## 4. Validation

* `tests/instrument/adapters/_base/test_genai_semconv.py` — 49 tests covering
  every constant, every helper code path, type contracts, and the
  centralized hook.
* `tests/instrument/adapters/test_genai_semconv_per_adapter.py` — 35 tests
  parameterized across all 9 providers and all 16 framework adapters,
  asserting `gen_ai.*` attributes appear on emitted `model.invoke` events.

Run:

```bash
uv run pytest tests/instrument/adapters/_base/test_genai_semconv.py \
              tests/instrument/adapters/test_genai_semconv_per_adapter.py
```

---

## 5. Migration & Compatibility

The OTel GenAI semantic conventions are upstream-marked **experimental**
(v0.29+). LayerLens pins a known-good attribute set; upstream renames
will be picked up by updating the constants in `genai_semconv.py` and
the spec test file `test_genai_semconv.py::TestGenAiAttributeNamesMatchSpec`,
both of which fail loudly if the spec drifts.

The legacy LayerLens attribute set (`provider`, `model`,
`prompt_tokens`, …) is **not deprecated** and continues to be emitted
indefinitely. There is no migration deadline; both sets coexist.

---

*Spec version pinned: 1.0.0 (matches `07-otel-genai-semantic-conventions.md`)*
