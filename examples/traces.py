#!/usr/bin/env -S poetry run python

from layerlens import Stratix

# Construct sync client (API key from env or inline)
client = Stratix()

# --- Upload traces from a file
result = client.traces.upload("./traces.jsonl")
print(f"Uploaded {result.count} traces")

# --- List traces
response = client.traces.get_many()
print(f"Found {response.total_count} traces")
for trace in response.traces:
    print(f"  - {trace.id}: {trace.filename}")

# --- List traces with filters
response = client.traces.get_many(
    source="upload",
    sort_by="created_at",
    sort_order="desc",
    page_size=10,
)

# --- Get a single trace
trace = client.traces.get(response.traces[0].id)
print(f"Trace {trace.id}: {len(trace.data)} data keys")

# --- Get available sources
sources = client.traces.get_sources()
print(f"Sources: {sources}")

# --- Delete a trace
deleted = client.traces.delete(trace.id)
print(f"Deleted: {deleted}")
