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


def recorded_trace_path(*parts: str) -> str:
    """Resolve a shipped recorded-trace fixture under ``samples/data/traces/``.

    The industry/cowork samples upload **recorded real traces** — genuine,
    fully instrumented traces captured once (by ``data/_generate_fixtures.py``)
    from real model runs and shipped in the repo. Passing the path parts here
    keeps the samples portable regardless of the working directory.

    Example::

        path = recorded_trace_path("industry", "financial_fraud.jsonl")
    """
    return os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "traces", *parts)
    )


def upload_recorded_trace(client: Stratix, fixture_path: str) -> List[str]:
    """Upload a recorded real-trace fixture and return the created trace IDs.

    A fixture is a JSONL file where each line is a complete, honestly
    attested trace captured from a real instrumented agent run (see
    ``samples/data/_generate_fixtures.py``). Because the trace carries real
    ``agent.identity``/``model.invoke``/``agent.output`` events, the LayerLens
    UI renders the Agent, Framework, and Status columns from genuine data — no
    fabrication. The returned IDs are in file order, so callers can zip them
    back onto the source scenarios to run evaluations.

    Args:
        client: An initialized :class:`Stratix` client.
        fixture_path: Absolute path to the ``.jsonl`` fixture (see
            :func:`recorded_trace_path`).

    Returns:
        The created trace IDs in file order (empty list if the upload was
        rejected without raising).
    """
    with open(fixture_path) as f:
        traces = [json.loads(line) for line in f if line.strip()]
    if not traces:
        return []

    # Upload the traces together as a JSON array so the backend creates one
    # trace per record (a JSONL file whose lines each start with "{" is read as
    # a single JSON object, yielding only the first record). The array is
    # created in-order, so the returned IDs line up with the fixture records.
    fd, path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w") as tmp:
            json.dump(traces, tmp, default=str)
        result = client.traces.upload(path)
    finally:
        if os.path.exists(path):
            os.unlink(path)

    if result and getattr(result, "trace_ids", None):
        return list(result.trace_ids)
    return []


def trace_call(
    client: Stratix,
    *,
    agent_name: str,
    run_fn: Any,
    input_value: Any = None,
) -> tuple[Any, Optional[str]]:
    """Trace a real, instrumented model call and upload the resulting trace.

    Use this in the integration samples to demonstrate **live tracing**: first
    ``instrument_openai(...)`` / ``instrument_anthropic(...)`` to wire the
    provider, then call ``trace_call`` with a ``run_fn`` that makes the real
    API call. The provider's instrumented call emits genuine
    ``model.invoke``/``cost.record`` events into the trace, so the uploaded
    trace carries the real framework, model, token counts, and status — nothing
    is fabricated.

    The trace is uploaded synchronously (via :meth:`traces.upload`) so the
    created trace ID is returned to the caller for running evaluations.

    Args:
        client: An initialized :class:`Stratix` client.
        agent_name: Honest name for the agent being traced (fills the Agent
            column). It is re-verified server-side.
        run_fn: A zero-argument callable that makes the instrumented API call
            and returns its result (e.g. the completion text).
        input_value: Optional value recorded as the trace's input.

    Returns:
        ``(result, trace_id)`` where ``result`` is ``run_fn()``'s return value
        and ``trace_id`` is the created trace ID (``None`` if the upload was
        rejected).
    """
    import uuid

    from layerlens.instrument import TraceCollector
    from layerlens.instrument._capture_config import CaptureConfig
    from layerlens.instrument._context import _current_collector, _push_span, _pop_span

    collector = TraceCollector(client, CaptureConfig.full())
    root_span_id = uuid.uuid4().hex[:16]
    col_token = _current_collector.set(collector)
    span_snapshot = _push_span(root_span_id, agent_name)
    try:
        collector.emit(
            "agent.input",
            {"name": agent_name, "input": input_value},
            span_id=root_span_id,
            span_name=agent_name,
        )
        result = run_fn()
        collector.emit(
            "agent.output",
            {"name": agent_name, "output": result, "status": "ok"},
            span_id=root_span_id,
            span_name=agent_name,
        )
        # Declare the agent identity so the Agent column renders (re-verified
        # server-side; a generic/model name would be rejected).
        collector.emit(
            "agent.identity",
            {"agent_name": agent_name},
            span_id=root_span_id,
            span_name=agent_name,
        )
    finally:
        _pop_span(span_snapshot)
        _current_collector.reset(col_token)

    payload = collector.to_replay_dict()
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(payload, default=str) + "\n")
        upload_result = client.traces.upload(path)
    finally:
        if os.path.exists(path):
            os.unlink(path)

    trace_id = None
    if upload_result and getattr(upload_result, "trace_ids", None):
        trace_id = upload_result.trace_ids[0]
    return result, trace_id


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

    raise RuntimeError(
        "No models available. Add a model to your project or check API connectivity."
    )


def create_judge(
    client: Stratix,
    *,
    name: str,
    evaluation_goal: str,
    model_id: Optional[str] = None,
    namespace: Optional[str] = None,
) -> Any:
    """Create a judge, automatically resolving model_id if not provided.

    Args:
        client: An initialized :class:`Stratix` client.
        name: Judge display name.
        evaluation_goal: What the judge evaluates (min 10 characters).
        model_id: Explicit model ID. If ``None``, resolves via :func:`get_default_model_id`.
        namespace: Optional per-sample namespace appended to the display name
            (``"<name> (<namespace>)"``). Judges are matched/reused by name, so
            two samples that both want e.g. ``"Relevance Judge"`` would otherwise
            silently cross-wire if a run is interrupted before cleanup. Passing
            the sample's own namespace keeps every sample's judges distinct.

    Returns:
        A :class:`Judge` object.
    """
    if namespace:
        name = f"{name} ({namespace})"
    if model_id is None:
        model_id = get_default_model_id(client)
    try:
        return client.judges.create(
            name=name, evaluation_goal=evaluation_goal, model_id=model_id
        )
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
