#!/usr/bin/env -S poetry run python

from atlas import Atlas

# gets API key, organization ID and project ID from environment variables:
# - LAYERLENS_ATLAS_API_KEY
# - LAYERLENS_ATLAS_ORG_ID
# - LAYERLENS_ATLAS_PROJECT_ID
client = Atlas()

# Evaluations
evaluation = client.evaluations.create(model="random", benchmark="random")

# Results
if evaluation is not None:
    results = client.results.get(evaluation_id=evaluation.id)
