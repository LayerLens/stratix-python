---
name: optimize
description: Optimize judges in LayerLens to improve evaluation quality and reduce cost
user_invocable: true
---

You are helping the user optimize judges in the LayerLens platform using the Python SDK. Judge optimization fine-tunes a judge's evaluation criteria based on prior trace evaluation results to improve accuracy and consistency.

## SDK Reference

```python
from layerlens import Stratix
client = Stratix()

# Step 1: Estimate optimization cost before committing
estimate = client.judge_optimizations.estimate(
    judge_id="judge_id_here",   # the judge to optimize
    budget="medium",            # "low", "medium", or "high"
)
# Returns cost estimate details

# Step 2: Create an optimization run
run = client.judge_optimizations.create(
    judge_id="judge_id_here",   # the judge to optimize
    budget="medium",            # "low", "medium", or "high"
)
# Returns JudgeOptimization with .id, .status

# Step 3: Poll for completion
run_status = client.judge_optimizations.get("optimization_run_id")
# Returns JudgeOptimization with .id, .status
# Status values: "pending", "running", "completed", "failed", "cancelled"

# Step 4: List all optimization runs for a judge
response = client.judge_optimizations.get_many(judge_id="judge_id_here")
# Returns JudgeOptimizationsResponse with .optimization_runs list, .count

# Step 5: Apply the optimized version to the judge
applied = client.judge_optimizations.apply("optimization_run_id")
# Updates the judge with the optimized evaluation criteria
```

## Instructions

When the user asks to optimize a judge:

1. **Identify the judge**: Get the judge ID. If the user does not have one, list judges with `client.judges.get_many()` and help them pick one. The judge should have completed trace evaluations for optimization to work well.

2. **Choose a budget**: Ask the user to pick a budget level:
   - `"low"` -- faster, cheaper, smaller improvement
   - `"medium"` -- balanced (recommended default)
   - `"high"` -- most thorough, higher cost, best results

3. **Estimate cost**: Run `client.judge_optimizations.estimate(judge_id=..., budget=...)` and show the estimate to the user before proceeding.

4. **Create the run**: After user confirmation, run `client.judge_optimizations.create(judge_id=..., budget=...)`.

5. **Poll for completion**: Check status periodically with `client.judge_optimizations.get(run_id)`. A typical optimization takes a few minutes. Poll every 10 seconds.

6. **Apply results**: Once status is `"completed"`, ask the user if they want to apply the optimization with `client.judge_optimizations.apply(run_id)`. This updates the judge with the improved evaluation criteria.

Always estimate cost and get user confirmation before creating the optimization run.

See `samples/core/judge_optimization.py` for a complete optimization workflow.
