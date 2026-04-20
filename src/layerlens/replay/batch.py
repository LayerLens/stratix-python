"""Concurrent batch replay with aggregated reporting."""

from __future__ import annotations

import time
import uuid
from typing import Dict, List, Callable, Iterable, Optional
from concurrent.futures import Future, TimeoutError as FuturesTimeoutError, ThreadPoolExecutor

from pydantic import Field, BaseModel

from .models import (
    ReplayResult,
    ReplayStatus,
    ReplayRequest,
    BatchReplayFilter,
)
from .controller import ReplayController
from ..models.trace import Trace


class BatchReplayRequest(BaseModel):
    filter: BatchReplayFilter = Field(default_factory=BatchReplayFilter)
    model_override: Optional[str] = None
    config_overrides: Dict[str, object] = Field(default_factory=dict)
    prompt_overrides: Dict[str, object] = Field(default_factory=dict)
    concurrency: int = Field(default=5, ge=1, le=50)
    timeout_per_trace_ms: float = 60_000.0


class BatchReplaySummary(BaseModel):
    total_traces: int = 0
    completed: int = 0
    failed: int = 0
    timed_out: int = 0
    output_change_rate: float = 0.0
    avg_output_similarity: float = 1.0
    avg_cost_diff_usd: Optional[float] = None
    total_cost_original_usd: Optional[float] = None
    total_cost_replay_usd: Optional[float] = None
    avg_latency_diff_ms: Optional[float] = None
    duration_ms: float = 0.0


class BatchReplayResult(BaseModel):
    batch_id: str
    summary: BatchReplaySummary = Field(default_factory=BatchReplaySummary)
    results: List[ReplayResult] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)


class BatchReplayer:
    """Apply a single override profile across many traces in parallel."""

    def __init__(self, controller: ReplayController) -> None:
        self._controller = controller

    def run(
        self,
        traces: Iterable[Trace],
        request: BatchReplayRequest,
        *,
        cost_lookup: Optional[Callable[[Trace], float]] = None,
    ) -> BatchReplayResult:
        batch_id = f"batch_{uuid.uuid4().hex[:16]}"
        start = time.time()
        traces = list(traces)
        results: List[ReplayResult] = []
        errors: List[str] = []

        with ThreadPoolExecutor(max_workers=request.concurrency) as pool:
            futures: Dict[Future[ReplayResult], Trace] = {}
            for trace in traces:
                req = ReplayRequest(
                    trace_id=trace.id,
                    model_override=request.model_override,
                    config_overrides=dict(request.config_overrides),
                    prompt_overrides=dict(request.prompt_overrides),
                )
                cost_original = cost_lookup(trace) if cost_lookup else None
                futures[
                    pool.submit(
                        self._controller.run,
                        trace,
                        req,
                        cost_original=cost_original,
                        cost_replay_fn=cost_lookup,
                    )
                ] = trace

            timeout_s = max(request.timeout_per_trace_ms / 1000, 0.001)
            for fut, trace in futures.items():
                try:
                    results.append(fut.result(timeout=timeout_s))
                except FuturesTimeoutError:
                    results.append(
                        ReplayResult(
                            original_trace_id=trace.id,
                            replay_trace_id=f"replay_{uuid.uuid4().hex[:16]}",
                            status=ReplayStatus.TIMEOUT,
                            duration_ms=request.timeout_per_trace_ms,
                            error="timeout",
                        )
                    )
                except Exception as exc:
                    errors.append(f"{trace.id}: {exc}")

        summary = _summarize(results)
        summary.total_traces = len(traces)
        summary.duration_ms = (time.time() - start) * 1000
        return BatchReplayResult(batch_id=batch_id, summary=summary, results=results, errors=errors)


def _summarize(results: List[ReplayResult]) -> BatchReplaySummary:
    summary = BatchReplaySummary()
    if not results:
        return summary
    sims: List[float] = []
    cost_diffs: List[float] = []
    latency_diffs: List[float] = []
    changed = 0
    for r in results:
        if r.status == ReplayStatus.COMPLETED:
            summary.completed += 1
            sims.append(r.diff.output_similarity)
            if r.diff.output_changed:
                changed += 1
            if r.diff.cost_diff_usd is not None:
                cost_diffs.append(r.diff.cost_diff_usd)
            if r.diff.latency_diff_ms is not None:
                latency_diffs.append(r.diff.latency_diff_ms)
        elif r.status == ReplayStatus.FAILED:
            summary.failed += 1
        elif r.status == ReplayStatus.TIMEOUT:
            summary.timed_out += 1
    if sims:
        summary.avg_output_similarity = sum(sims) / len(sims)
        summary.output_change_rate = changed / len(sims)
    if cost_diffs:
        summary.avg_cost_diff_usd = sum(cost_diffs) / len(cost_diffs)
    if latency_diffs:
        summary.avg_latency_diff_ms = sum(latency_diffs) / len(latency_diffs)
    return summary
