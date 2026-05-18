---
name: evaluate
description: Run trace evaluations with judges or model evaluations with benchmarks in LayerLens
user_invocable: true
---

You are helping the user run evaluations in the LayerLens platform using the Python SDK. There are two evaluation workflows: trace evaluations (judge a specific trace) and model evaluations (evaluate a model against a benchmark).

## SDK Reference: Trace Evaluations

```python
from layerlens import Stratix
client = Stratix()

# Estimate cost before running
estimate = client.trace_evaluations.estimate_cost(
    trace_ids=["trace_id_1", "trace_id_2"],  # list of trace IDs to evaluate
    judge_id="judge_id_here",                 # judge to use
)

# Create a trace evaluation (run a judge against a trace)
trace_eval = client.trace_evaluations.create(
    trace_id="trace_id_here",    # the trace to evaluate
    judge_id="judge_id_here",    # the judge to use
)
# Returns TraceEvaluation with .id, .status

# Get a trace evaluation by ID
trace_eval = client.trace_evaluations.get("trace_eval_id")

# List trace evaluations with filtering
response = client.trace_evaluations.get_many(
    judge_id="judge_id",       # filter by judge
    trace_id="trace_id",       # filter by trace
    outcome="pass",            # filter by outcome
    search="query",            # text search
    sort_by="created_at",      # sort field
    sort_order="desc",         # asc or desc
)
# Returns TraceEvaluationsResponse with .trace_evaluations, .count

# Get evaluation results
results = client.trace_evaluations.get_results("trace_eval_id")
# Returns TraceEvaluationResultsResponse (extends TraceEvaluationResult)
# Has .score, .passed, .reasoning, .steps, .latency_ms, .total_cost directly
```

## SDK Reference: Model Evaluations

```python
# Fetch available models and benchmarks
models = client.models.get()           # project models
benchmarks = client.benchmarks.get()   # project benchmarks

# Look up by key
model = client.models.get_by_key("gpt-4o")
benchmark = client.benchmarks.get_by_key("mmlu")

# Create an evaluation (model + benchmark)
evaluation = client.evaluations.create(
    model=model,           # Model object
    benchmark=benchmark,   # Benchmark object
)
# Returns Evaluation with .id, .status

# Wait for completion (blocking with polling)
evaluation = client.evaluations.wait_for_completion(
    evaluation,                # Evaluation object
    interval_seconds=15,       # poll interval (default 15)
    timeout_seconds=600,       # max wait time (default 600)
)
# Raises TimeoutError if not completed in time

# Check status manually
evaluation = client.evaluations.get(evaluation)      # by Evaluation object
evaluation = client.evaluations.get_by_id("eval_id")  # by ID string

# List evaluations
response = client.evaluations.get_many(
    sort_by="submittedAt",     # sort field
    order="desc",              # asc or desc
    model_ids=["id1"],         # filter by model IDs
    benchmark_ids=["id1"],     # filter by benchmark IDs
    status="completed",        # filter by status
)
# Returns EvaluationsResponse with .evaluations list

# Fetch results
results = client.results.get(evaluation=evaluation, page=1, page_size=20)
results = client.results.get_all(evaluation=evaluation)  # all pages
# Returns ResultsResponse with .results list
```

## Instructions

When the user asks to evaluate:

### Trace Evaluation (judge a specific trace)
1. Ensure the user has a trace ID and a judge ID. If not, help them find or create these first.
2. Optionally estimate cost with `client.trace_evaluations.estimate_cost()`.
3. Run `client.trace_evaluations.create(trace_id=..., judge_id=...)`.
4. Wait briefly, then fetch results with `client.trace_evaluations.get_results(id)`.
5. Display the score and outcome for each result.

### Model Evaluation (model vs benchmark)
1. Help the user select a model and benchmark using `client.models.get()` and `client.benchmarks.get()`, or look up by key.
2. Create the evaluation with `client.evaluations.create(model=model, benchmark=benchmark)`.
3. Wait for completion with `client.evaluations.wait_for_completion()`.
4. Fetch and display results with `client.results.get(evaluation=evaluation)`.

See `samples/core/trace_evaluation.py` for the trace evaluation workflow and `samples/core/run_evaluation.py` for the model evaluation workflow.
