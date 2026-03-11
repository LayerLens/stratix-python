#!/usr/bin/env python3

import time

from layerlens import Stratix

# Construct sync client (API key from env or inline)
client = Stratix()

# --- Fetch a model to use as the judge's LLM
models = client.models.get(type="public", name="gpt-4o")
if not models:
    print("No models found, exiting")
    exit(1)
model = models[0]
print(f"Using model: {model.name} ({model.id})")

# --- Create a judge
judge = client.judges.create(
    name=f"Code Quality Judge {int(time.time())}",
    evaluation_goal="Evaluate the quality of code output including correctness, readability, and style",
    model_id=model.id,
)
print(f"Created judge {judge.id}: {judge.name}")

# --- Get a judge by ID
judge = client.judges.get(judge.id)
print(f"Judge: {judge.name}, version: {judge.version}")

# --- List all judges
response = client.judges.get_many()
print(f"Found {response.total_count} judges")
for j in response.judges:
    print(f"  - {j.name} (v{j.version}, {j.run_count} runs)")

# --- Update a judge
updated = client.judges.update(
    judge.id,
    name="Updated Code Quality Judge",
    evaluation_goal="Evaluate code output for correctness, readability, style, and security",
)
print(f"Updated judge {updated.id}")

# --- Delete a judge
deleted = client.judges.delete(judge.id)
print(f"Deleted judge {deleted.id}")
