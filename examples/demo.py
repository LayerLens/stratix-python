#!/usr/bin/env -S poetry run python

from atlas import Atlas

# gets API key from environment variable:
# - LAYERLENS_ATLAS_API_KEY
client = Atlas()

# Evaluations
evaluation = client.evaluations.create(model="random_model_id", benchmark="random_benchmark_id")

# Results
if evaluation is not None:
    results = client.results.get(evaluation_id=evaluation.id)
