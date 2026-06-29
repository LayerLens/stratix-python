"""Live L1 + L2 + linkage verification, one test per protocol adapter (LLM-free).

Opt-in: see ``conftest.py``. Built-in protocols (agui/a2ui/ap2/ucp) always run;
mcp/a2a skip unless their optional package is installed (Python 3.10+).

    LAYERLENS_LIVE=1 LAYERLENS_STRATIX_BASE_URL=http://localhost:8080/api/v1 \\
    LAYERLENS_STRATIX_API_KEY=... LAYERLENS_LIVE_INTEGRATION_ID=<id> \\
    ./scripts/test tests/e2e/live -k "protocol"
"""

from __future__ import annotations

import pytest

from ._framework_harness import run_framework_case
from ._protocol_registry import PROTOCOLS


def _variants(case) -> tuple:
    """Run the ``redaction`` variant too for content-bearing protocols — the
    end-to-end proof that payment/commerce/delegation content never reaches the
    platform under ``capture_content=False`` (L1-L4 / LAY-3578)."""
    out = ["default"]
    if case.supports_redaction:
        out.append("redaction")
    return tuple(out)


_CASES = [(c, v) for c in PROTOCOLS for v in _variants(c)]
_IDS = [f"{c.id}-{v}" for c, v in _CASES]


@pytest.mark.live
@pytest.mark.parametrize("case, variant", _CASES, ids=_IDS)
def test_protocol_live(case, variant, stratix_live_client, record_result) -> None:
    pytest.importorskip(case.import_name, reason=f"{case.id}: '{case.import_name}' not installed")
    row = run_framework_case(stratix_live_client, case, variant)
    record_result(row)
