# Pydantic v1 / v2 Compatibility Matrix

Round-2 deliberation item 20. Each `layerlens` framework adapter
declares which Pydantic major versions it supports. Use this table
**before pinning Pydantic in your environment** — installing a v2-only
adapter under a v1-pinned runtime now raises a clear `RuntimeError` at
import time instead of producing a confusing `ImportError` deep inside
the framework SDK.

## Reading the matrix

| Value      | Meaning                                                           |
| ---------- | ----------------------------------------------------------------- |
| `v2_only`  | Adapter or its underlying framework requires Pydantic v2.         |
| `v1_only`  | Adapter or its underlying framework requires Pydantic v1.         |
| `v1_or_v2` | Adapter is version-agnostic — either Pydantic major works.        |

The declaration lives on the adapter class as a `requires_pydantic`
class attribute, is surfaced via `BaseAdapter.info().requires_pydantic`,
and is emitted in the adapter manifest consumed by the atlas-app
catalog UI.

## Framework adapters

| Adapter (`framework` key)  | Compat     | Justification                                                                                     |
| -------------------------- | ---------- | ------------------------------------------------------------------------------------------------- |
| `langchain`                | `v2_only`  | pyproject pin `langchain>=0.2,<0.4`; LangChain 0.2 migrated to Pydantic v2.                       |
| `langgraph`                | `v2_only`  | pyproject pin `langgraph>=0.2,<0.4`; depends on `langchain-core>=0.2` (Pydantic v2).              |
| `crewai`                   | `v2_only`  | pyproject pin `crewai>=0.30,<0.90`; CrewAI's pyproject pins `pydantic = "^2.4.2"`.                |
| `pydantic_ai`              | `v2_only`  | pydantic-ai is Pydantic v2 from day one (its pyproject requires `pydantic>=2.7`).                 |
| `langfuse`                 | `v2_only`  | Adapter's `frameworks/langfuse/config.py` line 13 imports `field_validator` (v2-only decorator).  |
| `autogen`                  | `v1_or_v2` | Adapter has no direct `pydantic` imports; pyautogen 0.2.x supports both majors.                   |
| `salesforce_agentforce`    | `v1_or_v2` | `frameworks/agentforce/models.py` uses only `BaseModel`/`Field` (identical surface in v1 and v2). |
| `semantic_kernel`          | `v1_or_v2` | Adapter has no direct `pydantic` imports; only filter callbacks + dict events.                    |
| `llama_index`              | `v1_or_v2` | Adapter has no direct `pydantic` imports; uses LlamaIndex Instrumentation Module dicts.           |
| `openai_agents`            | `v1_or_v2` | Adapter has no direct `pydantic` imports; reads SpanData structurally.                            |
| `agno`                     | `v1_or_v2` | Adapter has no direct `pydantic` imports; only wraps `Agent.run`/`Agent.arun`.                    |
| `bedrock_agents`           | `v1_or_v2` | Adapter has no direct `pydantic` imports; consumes Bedrock via boto3 (no Pydantic).               |
| `strands`                  | `v1_or_v2` | Adapter has no direct `pydantic` imports; agent-callback hooks emit dict events.                  |
| `smolagents`               | `v1_or_v2` | Only Pydantic touch is `layerlens._compat.pydantic.model_dump` (the v1/v2 shim).                  |
| `ms_agent_framework`       | `v1_or_v2` | Adapter has no direct `pydantic` imports.                                                         |
| `google_adk`               | `v1_or_v2` | Adapter has no direct `pydantic` imports; uses ADK's 6-callback hook system.                      |
| `embedding`                | `v1_or_v2` | Adapter has no direct `pydantic` imports; wraps client methods structurally.                      |

## Protocol adapters

All six protocol adapters (`a2a`, `agui`, `mcp_extensions`, `ap2`,
`a2ui`, `ucp`) are pydantic-agnostic — they speak protocol envelopes,
not Pydantic models — and inherit the `v1_or_v2` default.

## LLM provider adapters

All nine provider adapters (`openai`, `anthropic`, `azure_openai`,
`google_vertex`, `aws_bedrock`, `ollama`, `litellm`, `cohere`,
`mistral`) route any Pydantic access through
`layerlens._compat.pydantic` and are `v1_or_v2`. Note that the
underlying provider SDKs (`openai`, `anthropic`, etc.) themselves
require Pydantic v2 in current versions — but that constraint comes
from the provider SDK, not from the LayerLens adapter.

## Programmatic check

```python
from layerlens.instrument.adapters._base import (
    AdapterRegistry,
    PydanticCompat,
)

registry = AdapterRegistry()
for info in registry.list_available():
    if info.requires_pydantic is PydanticCompat.V2_ONLY:
        print(f"{info.framework}: requires Pydantic v2")
```

## Adding a new adapter

When porting a new framework adapter:

1. Set `requires_pydantic` on the adapter subclass explicitly. The
   linter test in `tests/instrument/adapters/test_pydantic_compat.py`
   refuses to merge an adapter that relies on the `BaseAdapter`
   default.
2. Document the rationale in the class docstring or as a comment
   beside the declaration. Cite the specific Pydantic-imports inside
   the adapter code or the framework's version pin — speculation is
   not accepted.
3. For `v2_only` adapters, also call `requires_pydantic(...)` at the
   top of the adapter package's `__init__.py`. This produces a clear
   `RuntimeError` at import time on incompatible runtimes instead of
   leaving the user to debug a deep stack trace in the framework SDK.
4. Update this document with the new row.
