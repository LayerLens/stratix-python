#!/usr/bin/env python
"""Dump the public catalog: models, benchmarks, and evaluations.

Uses only the public client endpoints which are less rate-limited
than project-scoped endpoints.

Usage:
    export LAYERLENS_STRATIX_API_KEY=your-api-key
    python marc-only/dump_public_catalog.py
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
    if hasattr(obj, "value"):
        return obj.value
    return str(obj)


def main() -> None:
    client = Stratix()
    print(f"Connected: org={client.organization_id}, project={client.project_id}\n")

    dump: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "organization_id": client.organization_id,
        "project_id": client.project_id,
    }

    # --- Public Models ---
    print("1. Public Models")
    resp = client.public.models.get()
    if resp and hasattr(resp, "models") and resp.models:
        dump["public_models"] = {
            "count": len(resp.models),
            "total": getattr(resp, "total", None),
            "filters": _to_dict(getattr(resp, "filters", None)),
            "models": _to_dict(resp.models),
        }
        print(f"   {len(resp.models)} models")
        for m in resp.models[:10]:
            print(f"   - {m.id}: {m.name} ({getattr(m, 'company', '?')})")
        if len(resp.models) > 10:
            print(f"   ... and {len(resp.models) - 10} more")
    else:
        dump["public_models"] = None
        print("   (none)")

    time.sleep(1)

    # --- Public Benchmarks ---
    print("\n2. Public Benchmarks")
    resp = client.public.benchmarks.get()
    datasets_attr = None
    for attr in ["datasets", "benchmarks"]:
        if resp and hasattr(resp, attr) and getattr(resp, attr):
            datasets_attr = attr
            break
    if datasets_attr:
        items = getattr(resp, datasets_attr)
        dump["public_benchmarks"] = {
            "count": len(items),
            "total": getattr(resp, "total", None),
            "filters": _to_dict(getattr(resp, "filters", None)),
            "benchmarks": _to_dict(items),
        }
        print(f"   {len(items)} benchmarks")
        for b in items[:10]:
            print(f"   - {b.id}: {getattr(b, 'name', '?')}")
        if len(items) > 10:
            print(f"   ... and {len(items) - 10} more")
    else:
        dump["public_benchmarks"] = None
        print("   (none)")

    time.sleep(1)

    # --- Public Evaluations ---
    print("\n3. Public Evaluations")
    resp = client.public.evaluations.get_many()
    if resp and hasattr(resp, "evaluations") and resp.evaluations:
        dump["public_evaluations"] = {
            "count": len(resp.evaluations),
            "total_count": getattr(resp, "total_count", None),
            "evaluations": _to_dict(resp.evaluations),
        }
        print(f"   {len(resp.evaluations)} evaluations")
        for e in resp.evaluations[:10]:
            print(f"   - {e.id}: status={getattr(e, 'status', '?')}")
        if len(resp.evaluations) > 10:
            print(f"   ... and {len(resp.evaluations) - 10} more")
    else:
        dump["public_evaluations"] = _to_dict(resp)
        print(f"   Response: {type(resp)}")

    # --- Write output ---
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(os.path.dirname(__file__), f"public_catalog_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(dump, f, indent=2, default=str)

    print(f"\nWritten to: {out_path}")


if __name__ == "__main__":
    main()
