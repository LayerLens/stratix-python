#!/usr/bin/env python3

from layerlens import Stratix

# Construct sync client (API key from env or inline)
client = Stratix()

# --- Models
models = client.models.get(type="public", name="gpt-4o")

if not models:
    print("gpt-4o not found, exiting")

model = models[0]

# --- Benchmarks
benchmarks = client.benchmarks.get(type="public", name="simpleQA")

if not benchmarks:
    print("SimpleQA benchmark not found, exiting")

benchmark = benchmarks[0]

# --- Create evaluation
evaluation = client.evaluations.create(
    model=model,
    benchmark=benchmark,
)
