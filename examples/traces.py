#!/usr/bin/env python3

import os

from layerlens import Stratix

# Construct sync client (API key from env or inline)
client = Stratix()

# --- Upload traces from a file
traces_file = os.path.join(os.path.dirname(__file__), "traces.jsonl")
result = client.traces.upload(traces_file)
print(f"Uploaded {len(result.trace_ids)} traces")

# --- List traces
response = client.traces.get_many()
print(f"Found {response.total_count} traces")
for trace in response.traces[:5]:
    print(f"  - {trace.id}: {trace.filename}")

# --- List traces with filters
filtered = client.traces.get_many(
    sort_by="created_at",
    sort_order="desc",
    page_size=10,
)
print(f"Filtered traces: {filtered.count}")

# --- Get a single trace (use the uploaded trace ID)
trace = client.traces.get(result.trace_ids[0])
print(f"Trace {trace.id}: {len(trace.data)} data keys")

# --- Get available sources
sources = client.traces.get_sources()
print(f"Sources: {sources}")

# --- Delete a trace
deleted = client.traces.delete(trace.id)
print(f"Deleted: {deleted}")
