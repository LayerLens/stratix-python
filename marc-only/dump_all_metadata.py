#!/usr/bin/env python
"""Dump every piece of metadata available via the LayerLens API.

Pulls all resources: models, benchmarks, evaluations, judges, traces,
trace evaluations, judge optimizations, results, public catalog, and
comparisons. Writes everything to a timestamped JSON file.

Usage:
    export LAYERLENS_STRATIX_API_KEY=your-api-key
    python marc-only/dump_all_metadata.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

from layerlens import Stratix


def _to_dict(obj: Any) -> Any:
    """Recursively convert SDK model objects to dicts."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_to_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__"):
        return {k: _to_dict(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
    if hasattr(obj, "value"):  # enums
        return obj.value
    return str(obj)


def _safe_call(label: str, fn, *args, max_retries: int = 5, **kwargs) -> Any:
    """Call an SDK method with retry on rate limits."""
    delay = 2.0
    for attempt in range(max_retries):
        try:
            result = fn(*args, **kwargs)
            return result
        except Exception as exc:
            err_str = str(exc)
            if "429" in err_str or "Limit" in err_str or "Rate" in err_str:
                wait = delay * (1.5 ** attempt)
                print(f"  [{label}] Rate limited, waiting {wait:.0f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
                continue
            print(f"  [{label}] ERROR: {type(exc).__name__}: {exc}")
            return None
    print(f"  [{label}] FAILED after {max_retries} retries")
    return None


def _paginate_all(label: str, fn, page_size: int = 100, **kwargs) -> list[Any]:
    """Paginate through all pages of a list endpoint."""
    all_items = []
    page = 1
    while True:
        resp = _safe_call(f"{label} page={page}", fn, page=page, page_size=page_size, **kwargs)
        if resp is None:
            break

        # Try common response patterns
        items = None
        for attr in ["traces", "evaluations", "judges", "results", "optimization_runs",
                      "trace_evaluations", "models", "benchmarks", "datasets"]:
            if hasattr(resp, attr):
                items = getattr(resp, attr)
                break

        if items is None:
            if isinstance(resp, list):
                items = resp
            else:
                # Single-page response
                all_items.append(resp)
                break

        if not items:
            break

        all_items.extend(items)
        print(f"  [{label}] page {page}: {len(items)} items (total so far: {len(all_items)})")

        # Check if there are more pages
        total = None
        for attr in ["total_count", "total"]:
            if hasattr(resp, attr):
                total = getattr(resp, attr)
                break
        if total is not None and len(all_items) >= total:
            break

        count = getattr(resp, "count", len(items))
        if count < page_size:
            break

        page += 1
        time.sleep(0.3)  # Rate limit courtesy

    return all_items


def main() -> None:
    # Retry client init in case of transient rate limits from prior runs
    client = None
    for attempt in range(5):
        try:
            client = Stratix()
            break
        except Exception as exc:
            if "429" in str(exc) or "Limit" in str(exc):
                wait = 5 * (attempt + 1)
                print(f"Rate limited on init, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"ERROR: {exc}")
                sys.exit(1)
    if client is None:
        print("ERROR: Could not initialize client after retries")
        sys.exit(1)

    org_id = client.organization_id
    project_id = client.project_id
    print(f"Connected: org={org_id}, project={project_id}")
    print(f"Base URL: {client.base_url}\n")

    dump: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "organization_id": org_id,
        "project_id": project_id,
        "base_url": str(client.base_url),
    }

    # --- 1. Project Models ---
    print("1. Project Models")
    custom_models = _safe_call("models.get(custom)", client.models.get, type="custom")
    public_models = _safe_call("models.get(public)", client.models.get, type="public")
    all_models = _safe_call("models.get(all)", client.models.get)
    dump["models"] = {
        "custom": _to_dict(custom_models),
        "public": _to_dict(public_models),
        "all": _to_dict(all_models),
    }
    print(f"  Custom: {len(custom_models) if custom_models else 0}")
    print(f"  Public: {len(public_models) if public_models else 0}")
    print(f"  All:    {len(all_models) if all_models else 0}")

    # --- 2. Project Benchmarks ---
    print("\n2. Project Benchmarks")
    custom_benchmarks = _safe_call("benchmarks.get(custom)", client.benchmarks.get, type="custom")
    public_benchmarks = _safe_call("benchmarks.get(public)", client.benchmarks.get, type="public")
    all_benchmarks = _safe_call("benchmarks.get(all)", client.benchmarks.get)
    dump["benchmarks"] = {
        "custom": _to_dict(custom_benchmarks),
        "public": _to_dict(public_benchmarks),
        "all": _to_dict(all_benchmarks),
    }
    print(f"  Custom: {len(custom_benchmarks) if custom_benchmarks else 0}")
    print(f"  Public: {len(public_benchmarks) if public_benchmarks else 0}")
    print(f"  All:    {len(all_benchmarks) if all_benchmarks else 0}")

    # --- 3. Evaluations ---
    print("\n3. Evaluations")
    evals_resp = _safe_call("evaluations.get_many", client.evaluations.get_many, page=1, page_size=100)
    dump["evaluations"] = _to_dict(evals_resp)
    if evals_resp and hasattr(evals_resp, "evaluations"):
        print(f"  Count: {len(evals_resp.evaluations)}")
        print(f"  Total: {getattr(evals_resp, 'total_count', 'N/A')}")
    else:
        print(f"  Response: {type(evals_resp)}")

    # --- 4. Judges ---
    print("\n4. Judges")
    judges = _paginate_all("judges", client.judges.get_many, page_size=100)
    dump["judges"] = _to_dict(judges)
    print(f"  Total: {len(judges)}")

    # --- 5. Traces ---
    print("\n5. Traces")
    traces = _paginate_all("traces", client.traces.get_many, page_size=100)
    dump["traces"] = _to_dict(traces)
    print(f"  Total: {len(traces)}")

    # --- 6. Trace Sources ---
    print("\n6. Trace Sources")
    sources = _safe_call("traces.get_sources", client.traces.get_sources)
    dump["trace_sources"] = _to_dict(sources)
    print(f"  Sources: {sources}")

    # --- 7. Trace Evaluations ---
    print("\n7. Trace Evaluations")
    trace_evals = _paginate_all("trace_evaluations", client.trace_evaluations.get_many, page_size=100)
    dump["trace_evaluations"] = _to_dict(trace_evals)
    print(f"  Total: {len(trace_evals)}")

    # --- 8. Judge Optimizations ---
    print("\n8. Judge Optimizations")
    opt_runs = _paginate_all("judge_optimizations", client.judge_optimizations.get_many, page_size=100)
    dump["judge_optimizations"] = _to_dict(opt_runs)
    print(f"  Total: {len(opt_runs)}")

    # --- 9. Public Catalog: Models ---
    print("\n9. Public Catalog: Models")
    pub_models = _safe_call("public.models.get", client.public.models.get)
    if pub_models and hasattr(pub_models, "models"):
        dump["public_models"] = {
            "count": len(pub_models.models),
            "total": getattr(pub_models, "total", None),
            "models": _to_dict(pub_models.models),
            "filters": _to_dict(getattr(pub_models, "filters", None)),
        }
        print(f"  Models: {len(pub_models.models)}")
    else:
        dump["public_models"] = _to_dict(pub_models)
        print(f"  Response: {type(pub_models)}")

    # --- 10. Public Catalog: Benchmarks ---
    print("\n10. Public Catalog: Benchmarks")
    pub_benchmarks = _safe_call("public.benchmarks.get", client.public.benchmarks.get)
    if pub_benchmarks and hasattr(pub_benchmarks, "datasets"):
        dump["public_benchmarks"] = {
            "count": len(pub_benchmarks.datasets),
            "total": getattr(pub_benchmarks, "total", None),
            "datasets": _to_dict(pub_benchmarks.datasets),
            "filters": _to_dict(getattr(pub_benchmarks, "filters", None)),
        }
        print(f"  Benchmarks: {len(pub_benchmarks.datasets)}")
    elif pub_benchmarks and hasattr(pub_benchmarks, "benchmarks"):
        dump["public_benchmarks"] = {
            "count": len(pub_benchmarks.benchmarks),
            "benchmarks": _to_dict(pub_benchmarks.benchmarks),
        }
        print(f"  Benchmarks: {len(pub_benchmarks.benchmarks)}")
    else:
        dump["public_benchmarks"] = _to_dict(pub_benchmarks)
        print(f"  Response: {type(pub_benchmarks)}")

    # --- 11. Public Catalog: Evaluations ---
    print("\n11. Public Catalog: Evaluations")
    pub_evals = _safe_call("public.evaluations.get_many", client.public.evaluations.get_many)
    dump["public_evaluations"] = _to_dict(pub_evals)
    if pub_evals and hasattr(pub_evals, "evaluations"):
        print(f"  Evaluations: {len(pub_evals.evaluations)}")
    else:
        print(f"  Response: {type(pub_evals)}")

    # --- 12. Individual trace details (first 5) ---
    print("\n12. Trace Details (first 5)")
    trace_details = []
    for trace in traces[:5]:
        tid = trace.id if hasattr(trace, "id") else str(trace)
        detail = _safe_call(f"traces.get({tid[:12]})", client.traces.get, tid)
        if detail:
            trace_details.append(_to_dict(detail))
            print(f"  {tid}: keys={list(detail.data.keys()) if hasattr(detail, 'data') and detail.data else 'N/A'}")
        time.sleep(0.3)
    dump["trace_details"] = trace_details

    # --- 13. Individual trace evaluation results (first 5) ---
    print("\n13. Trace Evaluation Results (first 5)")
    te_results = []
    for te in trace_evals[:5]:
        te_id = te.id if hasattr(te, "id") else str(te)
        results = _safe_call(f"trace_evaluations.get_results({te_id[:12]})", client.trace_evaluations.get_results, te_id)
        if results:
            te_results.append({"trace_evaluation_id": te_id, "results": _to_dict(results)})
            result_count = len(results.results) if hasattr(results, "results") and results.results else 0
            print(f"  {te_id}: {result_count} result(s)")
        time.sleep(0.3)
    dump["trace_evaluation_results"] = te_results

    # --- Write output ---
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(os.path.dirname(__file__), f"metadata_dump_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(dump, f, indent=2, default=str)

    print(f"\nDump written to: {out_path}")
    print(f"Total keys: {len(dump)}")
    for key, val in dump.items():
        if isinstance(val, dict):
            print(f"  {key}: {len(val)} entries")
        elif isinstance(val, list):
            print(f"  {key}: {len(val)} items")
        else:
            print(f"  {key}: {val}")


if __name__ == "__main__":
    main()
