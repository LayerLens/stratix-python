---
name: judge
description: Create, configure, and manage judges in LayerLens
user_invocable: true
---

You are helping the user manage judges in the LayerLens platform using the Python SDK. Judges are LLM-powered evaluators that score traces against a defined evaluation goal.

## SDK Reference

```python
from layerlens import Stratix
client = Stratix()

# List available models to use as the judge's backing LLM
models = client.models.get(type="public")
# Each model has .id, .name, .key

# Create a judge
judge = client.judges.create(
    name="My Judge",                    # display name
    evaluation_goal="Evaluate AI responses for accuracy, completeness, and safety.",  # what the judge checks
    model_id="model_id_here",           # backing LLM model ID
)
# Returns Judge with .id, .name, .version

# Get a single judge by ID
judge = client.judges.get("judge_id_here")
# Returns Judge with .id, .name, .evaluation_goal, .version, .run_count

# List all judges with pagination
response = client.judges.get_many(page_size=20)
# Returns JudgesResponse with .judges list, .total_count

# Update a judge
updated = client.judges.update(
    "judge_id_here",
    name="Updated Name",                             # optional
    evaluation_goal="Updated evaluation criteria.",   # optional
    model_id="new_model_id",                          # optional
)
# Returns updated Judge

# Delete a judge
deleted = client.judges.delete("judge_id_here")
# Returns deleted Judge with .id
```

## Instructions

When the user asks to work with judges:
1. If they want to create a judge: first fetch available models with `client.models.get(type="public")` so they can pick a backing LLM. Then call `client.judges.create()` with name, evaluation_goal, and model_id. The `evaluation_goal` is a natural language description of what the judge should evaluate.
2. If they want to list judges: use `client.judges.get_many()` and display name, ID, version, and run count.
3. If they want to inspect a judge: use `client.judges.get(id)` and show all fields.
4. If they want to update: use `client.judges.update(id, ...)` with only the fields to change.
5. If they want to delete: confirm the judge ID with the user before calling `client.judges.delete(id)`.

When crafting the `evaluation_goal`, help the user write a clear, specific description of what the judge should evaluate (e.g., factual accuracy, safety compliance, tone, completeness).

See `samples/core/create_judge.py` for a full CRUD example and `samples/core/judge_creation_and_test.py` for creating and testing a judge end-to-end.
