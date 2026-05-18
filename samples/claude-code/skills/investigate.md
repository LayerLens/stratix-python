---
name: investigate
description: Investigate production traces for errors, latency issues, and quality problems in LayerLens
user_invocable: true
---

You are helping the user investigate production traces in the LayerLens platform using the Python SDK. This workflow combines trace listing, filtering, inspection, and evaluation to diagnose issues.

## SDK Reference

```python
from layerlens import Stratix
client = Stratix()

# Step 1: List recent traces
response = client.traces.get_many(
    page_size=50,              # fetch a good sample
    sort_by="created_at",      # most recent first
    sort_order="desc",
    source="production.jsonl", # optional: filter by source
    status="error",            # optional: filter for errors
    search="timeout",          # optional: text search
)
# Returns TracesResponse with .traces, .count, .total_count

# Step 2: Inspect a specific trace
trace = client.traces.get("trace_id")
# Returns Trace with .id, .data, .filename, .created_at
# trace.data is a dict with keys like: input, output, metadata, error

# Step 3: Get available sources to narrow the search
sources = client.traces.get_sources()  # Returns List[str]

# Step 4: Run a judge against suspicious traces
# First, find or create a judge
judges_response = client.judges.get_many()
judge = judges_response.judges[0]  # pick an existing judge

# Or create one for the investigation
judge = client.judges.create(
    name="Investigation Judge",
    evaluation_goal="Check for factual errors, hallucinations, and incomplete answers.",
    model_id=model_id,
)

# Estimate evaluation cost
estimate = client.trace_evaluations.estimate_cost(
    trace_ids=["trace_id_1", "trace_id_2"],
    judge_id=judge.id,
)

# Evaluate a trace
trace_eval = client.trace_evaluations.create(
    trace_id="trace_id",
    judge_id=judge.id,
)

# Fetch results
results = client.trace_evaluations.get_results(trace_eval.id)
# Each result has .score, .outcome

# Step 5: List evaluations for a trace to see past judgments
evals = client.trace_evaluations.get_many(
    trace_id="trace_id",
    sort_by="created_at",
    sort_order="desc",
)
```

## Investigation Workflow

When the user asks to investigate traces, follow this structured approach:

### 1. Scope the investigation
- Ask what they are looking for: errors, slow responses, quality issues, or a specific problem.
- Use `client.traces.get_sources()` to see available data sources if needed.

### 2. Fetch and filter traces
- Use `client.traces.get_many()` with appropriate filters:
  - `status="error"` for error traces
  - `search="keyword"` for text search across trace content
  - `sort_by="created_at"` and `sort_order="desc"` for most recent
- Display a summary: total count, count matching filters, and a preview of the first few traces.

### 3. Inspect suspicious traces
- Use `client.traces.get(id)` to pull full trace details.
- Examine the trace data dict for:
  - `input` -- what was sent to the model
  - `output` -- what the model returned
  - `metadata` -- model name, latency, tokens, error codes
  - `error` -- error messages if present
- Look for patterns: repeated errors, high latency in metadata, empty outputs.

### 4. Evaluate traces with a judge (optional)
- If the user wants automated quality assessment, create or select a judge.
- Estimate cost first with `estimate_cost()`.
- Run `trace_evaluations.create()` on the suspicious traces.
- Fetch and display results showing scores and outcomes.

### 5. Summarize findings
- Report: total traces examined, issues found, patterns identified.
- Recommend next steps (fix prompts, adjust model, escalate errors).

See `samples/core/trace_investigation.py` for a complete investigation workflow example.
