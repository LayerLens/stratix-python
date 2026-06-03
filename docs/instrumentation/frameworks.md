# Agent Framework Instrumentation

Framework adapters plug into agent frameworks as callback handlers. Unlike provider adapters (which monkey-patch SDK methods), framework adapters receive events from the framework and build span trees from them.

## Supported Frameworks

| Framework | Adapter | Integration |
| --------- | ------- | ----------- |
| LangChain | `LangChainCallbackHandler` | Pass as a callback handler |
| LangGraph | `LangGraphCallbackHandler` | Pass as a callback handler |

## LangChain

### Installation

```bash
pip install layerlens[langchain]
```

### Usage

```python
from layerlens import Stratix
from layerlens.instrument.adapters.frameworks.langchain import LangChainCallbackHandler

client = Stratix()
handler = LangChainCallbackHandler(client)

# Pass the handler to any LangChain runnable
chain = prompt | llm | parser
result = chain.invoke(
    {"question": "What is RAG?"},
    config={"callbacks": [handler]},
)
```

The handler automatically captures:

| Event | Span Kind | Captured Data |
| ----- | --------- | ------------- |
| Chain start/end | `chain` | Chain name, input, output |
| LLM start/end | `llm` | Model name, prompts, response, token usage |
| Tool start/end | `tool` | Tool name, input query, output |
| Retriever start/end | `retriever` | Query, retrieved documents |

### How It Works

LangChain provides `run_id` (UUID) and `parent_run_id` for every callback event. The handler uses these to build a span tree:

1. `on_chain_start` — creates a root span (or child span if `parent_run_id` exists)
2. `on_llm_start` / `on_tool_start` / `on_retriever_start` — creates child spans
3. `on_*_end` — finishes the span with output data
4. `on_*_error` — finishes the span with `status="error"`
5. When the root chain ends — the full span tree is flushed as a trace

### Example: RAG Chain

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from layerlens import Stratix
from layerlens.instrument.adapters.frameworks.langchain import LangChainCallbackHandler

client = Stratix()
handler = LangChainCallbackHandler(client)

prompt = ChatPromptTemplate.from_template("Answer: {question}")
llm = ChatOpenAI(model="gpt-4o")
chain = prompt | llm | StrOutputParser()

result = chain.invoke(
    {"question": "What is retrieval-augmented generation?"},
    config={"callbacks": [handler]},
)
```

This produces a trace like:

```
RunnableSequence (kind=chain)
├── ChatPromptTemplate (kind=chain)
├── ChatOpenAI (kind=llm)
│   metadata: {model: "gpt-4o", usage: {total_tokens: 150}}
└── StrOutputParser (kind=chain)
```

### Error Handling

Chain and LLM errors are captured automatically:

```python
handler = LangChainCallbackHandler(client)

try:
    chain.invoke(input, config={"callbacks": [handler]})
except Exception:
    pass  # Trace still uploads with error spans
```

## LangGraph

The LangGraph adapter extends the LangChain handler with graph node awareness.

### Installation

```bash
pip install layerlens[langchain]
```

### Usage

```python
from layerlens import Stratix
from layerlens.instrument.adapters.frameworks.langgraph import LangGraphCallbackHandler

client = Stratix()
handler = LangGraphCallbackHandler(client)

# Use with a LangGraph compiled graph
result = graph.invoke(
    {"messages": [{"role": "user", "content": "Hello"}]},
    config={"callbacks": [handler]},
)
```

### Node Name Extraction

LangGraph attaches metadata to chain events that identifies which graph node is executing. The adapter extracts this to produce cleaner span names:

- Checks `metadata.langgraph_node` for the node name (highest priority)
- Falls back to the first plain tag (no colon), skipping internal `graph:step:*` tags
- Uses the chain name from `serialized` if neither is present

This means your traces show meaningful names like `agent`, `tools`, `retrieve` instead of generic `RunnableSequence` spans.

### Example Trace Output

```
StateGraph (kind=chain)
├── agent (kind=chain, node)
│   └── ChatOpenAI (kind=llm)
├── tools (kind=chain, node)
│   └── search (kind=tool)
└── agent (kind=chain, node)
    └── ChatOpenAI (kind=llm)
```

## Framework vs Provider Adapters

You can use both together. For example, use the LangChain callback handler for span tree structure, and a provider adapter to enrich LLM spans with token usage:

```python
from layerlens.instrument.adapters.providers.openai import instrument_openai
from layerlens.instrument.adapters.frameworks.langchain import LangChainCallbackHandler

# Both can be active simultaneously
instrument_openai(openai_client)
handler = LangChainCallbackHandler(client)

chain.invoke(input, config={"callbacks": [handler]})
```

Note: When using both, you may get duplicate LLM spans (one from the provider adapter, one from the framework callback). In most cases, using just the framework adapter is sufficient since it captures LLM events through callbacks.

## Next Steps

- [LLM Providers](providers.md) — Auto-instrument OpenAI, Anthropic, and LiteLLM
- [Quick Start](quickstart.md) — Manual instrumentation with `@trace` and `span()`
