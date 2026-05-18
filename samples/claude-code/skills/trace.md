---
name: trace
description: Upload, list, inspect, and manage traces in LayerLens
user_invocable: true
---

You are helping the user manage traces in the LayerLens platform using the Python SDK.

## SDK Reference

```python
from layerlens import Stratix
client = Stratix()

# Upload traces from a JSONL/JSON file (max 50 MB)
result = client.traces.upload("path/to/traces.jsonl")
# Returns CreateTracesResponse with .trace_ids list

# List traces with filtering and pagination
response = client.traces.get_many(
    page_size=20,           # default 100, max 500
    sort_by="created_at",   # sort field
    sort_order="desc",      # asc or desc
    source="filename.jsonl", # filter by source file
    status="error",         # filter by status
    search="query text",    # text search
)
# Returns TracesResponse with .traces list, .count, .total_count

# Get a single trace by ID
trace = client.traces.get("trace_id_here")
# Returns Trace with .id, .data, .filename, .created_at

# Delete a trace
deleted = client.traces.delete("trace_id_here")  # Returns bool

# Get available trace sources
sources = client.traces.get_sources()  # Returns List[str]
```

## Instructions

When the user asks to work with traces:
1. If they want to upload: use `client.traces.upload(file_path)`. The file must be JSON or JSONL format, max 50 MB.
2. If they want to list/search: use `client.traces.get_many()` with appropriate filters. Display trace IDs, creation dates, and sources.
3. If they want to inspect: use `client.traces.get(id)` and display the trace data including input, output, and metadata.
4. If they want to delete: confirm the trace ID with the user before calling `client.traces.delete(id)`.
5. If they want to see what sources exist: use `client.traces.get_sources()`.

Always show the trace IDs returned from uploads so the user can reference them later.

Sample trace data files are available in `samples/data/traces/` (simple_llm_trace.json, rag_pipeline_trace.json, multi_agent_trace.json, error_trace.json, batch_traces.jsonl).

See `samples/core/basic_trace.py` for a full upload example and `samples/core/trace_investigation.py` for an investigation workflow.
