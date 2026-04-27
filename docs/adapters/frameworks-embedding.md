# Embedding & vector store framework adapter

`layerlens.instrument.adapters.frameworks.embedding.EmbeddingAdapter` and
`VectorStoreAdapter` instrument embedding-creation calls and vector-store
operations across the common providers. They emit `embedding.create` and
`vector_store.query` events with dimension, batch size, and latency metadata.

## Install

```bash
pip install 'layerlens[embedding]'
```

The `embedding` extra has no required dependencies — bring your own provider
client (`openai`, `cohere`, `sentence-transformers`, `pinecone-client`,
`weaviate-client`, `chromadb`).

## Quick start (embeddings)

```python
from openai import OpenAI

from layerlens.instrument.adapters.frameworks.embedding import EmbeddingAdapter
from layerlens.instrument.transport.sink_http import HttpEventSink

sink = HttpEventSink(adapter_name="embedding")
adapter = EmbeddingAdapter()
adapter.add_sink(sink)
adapter.connect()

client = OpenAI()
adapter.wrap_openai(client)

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=["hello world"],
)
print(f"Dimensions: {len(response.data[0].embedding)}")

adapter.disconnect()
sink.close()
```

## Quick start (vector stores)

```python
from layerlens.instrument.adapters.frameworks.embedding import VectorStoreAdapter

vs_adapter = VectorStoreAdapter()
vs_adapter.connect()

# Pinecone:  vs_adapter.wrap_pinecone(my_index)
# Weaviate:  vs_adapter.wrap_weaviate(my_collection)
# Chroma:    vs_adapter.wrap_chroma(my_collection)
```

## What's wrapped

`EmbeddingAdapter`:

- `wrap_openai(client)` — patches `client.embeddings.create`.
- `wrap_cohere(client)` — patches `client.embed`.
- `wrap_sentence_transformer(model)` — patches `model.encode`.

`VectorStoreAdapter`:

- `wrap_pinecone(index)` — patches `index.query`.
- `wrap_weaviate(collection)` — patches `collection.query.near_vector` and
  `collection.query.bm25`.
- `wrap_chroma(collection)` — patches `collection.query`.

`disconnect()` restores all wrapped methods to their originals.

## Events emitted

| Event | Layer | When |
|---|---|---|
| `embedding.create` | L3 | Per embedding call. Payload: `provider`, `model`, `batch_size`, `dimensions`, `total_tokens`, `latency_ms`. |
| `vector_store.query` | L3 | Per vector-store query. Payload: `provider`, `top_k`, `result_count`, `latency_ms`, `index_name`. |

## Dimension tracking

The adapter inspects the response shape to record the actual returned
dimension count:

- OpenAI: `result.data[0].embedding` length.
- Cohere: `result.embeddings[0]` length.
- SentenceTransformer: `result.shape[1]` when the result is a numpy/torch tensor.

If a model is configured with `dimensions=N` truncation (OpenAI v3 family),
the recorded value is the post-truncation dimensionality.

## Capture config

```python
from layerlens.instrument.adapters._base import CaptureConfig

# Both events are L3, so the standard preset captures them.
adapter = EmbeddingAdapter(capture_config=CaptureConfig.standard())

# Production: drop content (the input text) but keep dimension/latency.
adapter = EmbeddingAdapter(
    capture_config=CaptureConfig(
        l3_model_metadata=True,
        capture_content=False,
    ),
)
```

## BYOK

The embedding adapter does not own provider keys — they belong to the
underlying client. For platform-managed BYOK see `docs/adapters/byok.md`.
