"""Shared helpers for LayerLens SDK samples.

Provides utility functions used across multiple samples to keep
individual sample files focused on demonstrating SDK features.
"""

from __future__ import annotations

import os
import json
import time
import logging
import tempfile
from typing import Any, List, Optional

from layerlens import Stratix

logger = logging.getLogger(__name__)


def upload_trace_dict(
    client: Stratix,
    *,
    input_text: str,
    output_text: str,
    metadata: Optional[dict[str, Any]] = None,
) -> Any:
    """Upload a single trace from in-memory data.

    Writes the trace to a temporary JSONL file and uploads via the SDK's
    ``client.traces.upload()`` method.

    Args:
        client: An initialized :class:`Stratix` client.
        input_text: The input/prompt text for the trace.
        output_text: The output/response text for the trace.
        metadata: Optional metadata dict attached to the trace.

    Returns:
        A :class:`CreateTracesResponse` with ``trace_ids``.
    """
    trace_data: dict[str, Any] = {
        "input": [{"role": "user", "content": input_text}],
        "output": output_text,
    }
    if metadata:
        trace_data["metadata"] = metadata

    fd, path = tempfile.mkstemp(suffix=".jsonl")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(trace_data) + "\n")
        result = client.traces.upload(path)
    finally:
        if os.path.exists(path):
            os.unlink(path)

    return result


def get_default_model_id(client: Stratix) -> str:
    """Get a model ID suitable for judge creation.

    Checks project models first, then falls back to the public catalog.
    Caches the result for the lifetime of the process.

    Args:
        client: An initialized :class:`Stratix` client.

    Returns:
        A model ID string suitable for passing to ``judges.create(model_id=...)``.

    Raises:
        RuntimeError: If no models are available in the project or public catalog.
    """
    # Check cache
    cached = getattr(get_default_model_id, "_cached_id", None)
    if cached:
        return cached

    # Use public models (required for judge creation)
    try:
        public_resp = client.public.models.get()
        if public_resp and hasattr(public_resp, "models") and public_resp.models:
            get_default_model_id._cached_id = public_resp.models[0].id  # type: ignore[attr-defined]
            return public_resp.models[0].id
    except Exception:
        pass

    # Fall back to project models
    try:
        models = client.models.get()
        if models:
            get_default_model_id._cached_id = models[0].id  # type: ignore[attr-defined]
            return models[0].id
    except Exception:
        pass

    raise RuntimeError("No models available. Add a model to your project or check API connectivity.")


def create_judge(
    client: Stratix,
    *,
    name: str,
    evaluation_goal: str,
    model_id: Optional[str] = None,
) -> Any:
    """Create a judge, automatically resolving model_id if not provided.

    Args:
        client: An initialized :class:`Stratix` client.
        name: Judge display name.
        evaluation_goal: What the judge evaluates (min 10 characters).
        model_id: Explicit model ID. If ``None``, resolves via :func:`get_default_model_id`.

    Returns:
        A :class:`Judge` object.
    """
    if model_id is None:
        model_id = get_default_model_id(client)
    try:
        return client.judges.create(name=name, evaluation_goal=evaluation_goal, model_id=model_id)
    except Exception as exc:
        # Handle 409 Conflict (judge name already exists) by finding and returning the existing judge
        if "already exists" in str(exc) or "409" in str(exc):
            logger.info("Judge '%s' already exists, reusing.", name)
            resp = client.judges.get_many()
            if resp and resp.judges:
                for j in resp.judges:
                    if j.name == name:
                        return j
        raise


def poll_evaluation_results(
    client: Stratix,
    evaluation_id: str,
    *,
    max_attempts: int = 60,
    initial_delay: float = 2.0,
    max_delay: float = 10.0,
    backoff_factor: float = 1.3,
) -> Optional[List[Any]]:
    """Poll for trace evaluation results with exponential backoff.

    Trace evaluations are **asynchronous**. When ``trace_evaluations.create()``
    returns, the evaluation has been accepted but execution has not yet started.
    The actual LLM judge execution takes a variable amount of time (typically
    5-60 seconds depending on model and trace complexity). During this window:

    - ``get_results()`` may raise a 404 ``NotFoundError`` (results row not
      yet written to the database).
    - ``get_results()`` may return an empty ``results=[]`` list (row exists
      but execution is still in progress).

    Both cases are normal and expected. This helper retries with exponential
    backoff until a non-empty result list appears or the attempt budget is
    exhausted.

    Args:
        client: An initialized :class:`Stratix` client.
        evaluation_id: The trace evaluation ID to poll.
        max_attempts: Maximum number of poll attempts (default 60, ~3-4 min total).
        initial_delay: Initial delay in seconds between polls.
        max_delay: Maximum delay cap in seconds.
        backoff_factor: Multiplier applied to delay each iteration.

    Returns:
        A list of :class:`TraceEvaluationResult` objects, or ``None``
        if results were not available within the polling window.
    """
    delay = initial_delay
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.trace_evaluations.get_results(evaluation_id)
            if resp and resp.score is not None:
                return [resp]
            # None or missing score -- evaluation accepted but execution still in progress
        except Exception:
            # 404 NotFoundError is expected while the results row hasn't been
            # created yet. Other transient errors (429, 502) are also retryable.
            pass

        if attempt < max_attempts:
            time.sleep(delay)
            delay = min(delay * backoff_factor, max_delay)

    return None
