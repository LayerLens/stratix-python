#!/usr/bin/env python
"""
Model & Benchmark Management -- LayerLens Python SDK Sample
===========================================================

Demonstrates model and benchmark management operations:

  1. List all models (public + custom) with filtering.
  2. Look up a model by key.
  3. Create a custom model.
  4. List all benchmarks (public + custom).
  5. Look up a benchmark by key.
  6. Add/remove models and benchmarks from the project.

This sample covers SDK model and benchmark management capabilities.

Prerequisites
-------------
* ``pip install layerlens --index-url https://sdk.layerlens.ai/package``
* Set ``LAYERLENS_STRATIX_API_KEY`` environment variable

Usage
-----
::

    export LAYERLENS_STRATIX_API_KEY=your-api-key
    python model_benchmark_management.py
"""

from __future__ import annotations

import logging
import sys

from layerlens import Stratix

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("layerlens.samples.model_benchmark_management")


def main() -> None:
    try:
        client = Stratix()
    except Exception as exc:
        logger.error("Failed to initialize client: %s", exc)
        sys.exit(1)

    logger.info("Connected to LayerLens (org=%s, project=%s)", client.organization_id, client.project_id)

    # --- Models ---
    logger.info("=" * 60)
    logger.info("Models")
    logger.info("=" * 60)

    # List all models
    all_models = client.models.get()
    if all_models:
        logger.info("Total models: %d", len(all_models))
    else:
        logger.info("No models in project")

    # List only public models
    public_models = client.models.get(type="public")
    if public_models:
        logger.info("Public models: %d", len(public_models))
        for m in public_models[:5]:
            logger.info("  - %s (key=%s)", m.name, m.key)

    # List only custom models
    custom_models = client.models.get(type="custom")
    if custom_models:
        logger.info("Custom models: %d", len(custom_models))
        for m in custom_models[:5]:
            logger.info("  - %s (key=%s)", m.name, m.key)

    # Look up by key
    if public_models:
        key = public_models[0].key
        model = client.models.get_by_key(key)
        if model:
            logger.info("Looked up model by key '%s': %s (id=%s)", key, model.name, model.id)

    # --- Benchmarks ---
    logger.info("=" * 60)
    logger.info("Benchmarks")
    logger.info("=" * 60)

    # List all benchmarks
    all_benchmarks = client.benchmarks.get()
    if all_benchmarks:
        logger.info("Total benchmarks: %d", len(all_benchmarks))
    else:
        logger.info("No benchmarks in project")

    # List only public benchmarks
    public_benchmarks = client.benchmarks.get(type="public")
    if public_benchmarks:
        logger.info("Public benchmarks: %d", len(public_benchmarks))
        for b in public_benchmarks[:5]:
            logger.info("  - %s (key=%s)", b.name, b.key)

    # Look up by key
    if public_benchmarks:
        key = public_benchmarks[0].key
        benchmark = client.benchmarks.get_by_key(key)
        if benchmark:
            logger.info("Looked up benchmark by key '%s': %s (id=%s)", key, benchmark.name, benchmark.id)

    # --- Public catalog (no auth required) ---
    logger.info("=" * 60)
    logger.info("Public Catalog")
    logger.info("=" * 60)

    try:
        pub_models = client.public.models.get()
        if pub_models and pub_models.models:
            logger.info("Public catalog models: %d", len(pub_models.models))
            for m in pub_models.models[:3]:
                logger.info("  - %s", getattr(m, "name", str(m)))
    except Exception as exc:
        logger.info("Public catalog not available: %s", exc)

    try:
        pub_benchmarks = client.public.benchmarks.get()
        if pub_benchmarks and pub_benchmarks.datasets:
            logger.info("Public catalog benchmarks: %d", len(pub_benchmarks.datasets))
            for b in pub_benchmarks.datasets[:3]:
                logger.info("  - %s", getattr(b, "name", str(b)))
    except Exception as exc:
        logger.info("Public catalog not available: %s", exc)

    # --- Additional: Add/remove models from the project ---
    logger.info("=" * 60)
    logger.info("Add/Remove Models")
    logger.info("=" * 60)

    try:
        # Add a public model to the project by ID
        success = client.models.add("model-id")
        logger.info("Add model: %s", "success" if success else "failed")

        # Remove a model from the project by ID
        success = client.models.remove("model-id")
        logger.info("Remove model: %s", "success" if success else "failed")
    except Exception as exc:
        logger.info("models.add/remove not available: %s", exc)

    # --- Additional: Add/remove benchmarks from the project ---
    logger.info("=" * 60)
    logger.info("Add/Remove Benchmarks")
    logger.info("=" * 60)

    try:
        # Add a public benchmark to the project by ID
        success = client.benchmarks.add("benchmark-id")
        logger.info("Add benchmark: %s", "success" if success else "failed")

        # Remove a benchmark from the project by ID
        success = client.benchmarks.remove("benchmark-id")
        logger.info("Remove benchmark: %s", "success" if success else "failed")
    except Exception as exc:
        logger.info("benchmarks.add/remove not available: %s", exc)

    # --- Additional: Filter models by company and region ---
    logger.info("=" * 60)
    logger.info("Model Filters: companies and regions")
    logger.info("=" * 60)

    try:
        # Filter models by company names
        filtered_models = client.models.get(companies=["openai", "anthropic"])
        if filtered_models:
            logger.info("Models from openai/anthropic: %d", len(filtered_models))
        else:
            logger.info("No models found for those companies")
    except Exception as exc:
        logger.info("models.get(companies=) not available: %s", exc)

    try:
        # Filter models by region
        regional_models = client.models.get(regions=["usa"])
        if regional_models:
            logger.info("Models in region 'usa': %d", len(regional_models))
        else:
            logger.info("No models found for that region")
    except Exception as exc:
        logger.info("models.get(regions=) not available: %s", exc)

    # --- Additional: Filter benchmarks by name ---
    logger.info("=" * 60)
    logger.info("Benchmark Filter: by name")
    logger.info("=" * 60)

    try:
        # Filter benchmarks by name
        mmlu = client.benchmarks.get(name="mmlu")
        if mmlu:
            logger.info("Found %d benchmark(s) matching 'mmlu'", len(mmlu))
            for b in mmlu:
                logger.info("  - %s (id=%s)", b.name, b.id)
        else:
            logger.info("No benchmarks matching 'mmlu'")
    except Exception as exc:
        logger.info("benchmarks.get(name=) not available: %s", exc)

    logger.info("Sample complete.")


if __name__ == "__main__":
    main()
