# Embedding instrumentation sample

End-to-end demo of `EmbeddingAdapter` — wraps an OpenAI client with
`EmbeddingAdapter.wrap_openai` and runs a single `embeddings.create`
call, emitting one `embedding.create` event per the module docstring.

## Prerequisites

```bash
pip install 'layerlens[embedding,providers-openai]'
```

> The `embedding` extra is intentionally empty — vector store hooks
> come from whichever store you actually use. This sample only
> exercises the OpenAI embedding path, so it needs the
> `providers-openai` extra for the `openai` SDK.

Required environment:

- `OPENAI_API_KEY` — your OpenAI API key. If unset the sample exits
  with code `2`.
- `LAYERLENS_STRATIX_API_KEY` — your LayerLens API key (optional).
- `LAYERLENS_STRATIX_BASE_URL` — atlas-app base URL (optional).

## Run

```bash
uv run python -m samples.instrument.embedding.main
```

## What this demonstrates

| Component | What it proves (source: `main.py`) |
|---|---|
| `EmbeddingAdapter(capture_config=CaptureConfig.standard())` | Standard capture profile. |
| `adapter.add_sink(HttpEventSink(...))` | Batched HTTP transport. |
| `adapter.connect()` / `adapter.disconnect()` | Full lifecycle. |
| `adapter.wrap_openai(client)` | Wraps the `OpenAI()` client's `embeddings.create` path; module docstring states each call emits one `embedding.create` event. |
| `client.embeddings.create(model="text-embedding-3-small", input=[..., ...])` | Two input strings → two vectors. The sample prints vector count and dimension. |
| Token usage print | When `response.usage` is non-`None`, total tokens are printed. |

## Expected output

When `OPENAI_API_KEY` is unset:

```text
OPENAI_API_KEY is not set; cannot run sample.
```

Exit code: `2`.

When the `openai` SDK is not installed:

```text
openai not installed. Install with:
    pip install 'layerlens[embedding,providers-openai]'
```

Exit code: `2`.

When the call succeeds:

```text
Embeddings: 2 vectors of dim 1536
Tokens: <n>
Telemetry shipped. Check the LayerLens dashboard adapter health page.
```

(The token line is conditional on `response.usage` being non-`None`.
The dim value is whatever `text-embedding-3-small` returns; 1536 is
the documented OpenAI dimension at the time of writing.)

## Multi-tenancy note

This sample does not pass `org_id` to `EmbeddingAdapter`. The
constructor does not yet accept `org_id` — production multi-tenant
wiring lands with the PR #118 adapter-side contract (currently
DRAFT). Once PR #118 merges, pass `org_id` to the adapter so every
emitted event carries it.
