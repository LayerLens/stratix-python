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


@pytest.mark.live
@pytest.mark.parametrize("case", PROTOCOLS, ids=[c.id for c in PROTOCOLS])
def test_protocol_live(case, stratix_live_client, record_result) -> None:
    pytest.importorskip(case.import_name, reason=f"{case.id}: '{case.import_name}' not installed")
    row = run_framework_case(stratix_live_client, case, "default")
    record_result(row)
