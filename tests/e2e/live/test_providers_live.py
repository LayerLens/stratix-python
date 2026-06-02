"""Live L1+L2 verification, one test per (provider, variant).

Opt-in: see conftest.py. Each test additionally skips unless that provider's SDK is
installed and its credentials are present. On success it appends a row to the L3
report (written at session end).

    LAYERLENS_LIVE=1 \\
    LAYERLENS_STRATIX_BASE_URL=<staging> LAYERLENS_STRATIX_API_KEY=... \\
    ANTHROPIC_API_KEY=... \\
    ./scripts/test tests/e2e/live -k anthropic
"""

from __future__ import annotations

import pytest

from ._harness import run_case
from ._registry import PROVIDERS, missing_credentials

_CASES = [(case, variant) for case in PROVIDERS for variant in case.variants]
_IDS = [f"{case.id}-{variant}" for case, variant in _CASES]


@pytest.mark.live
@pytest.mark.parametrize("case, variant", _CASES, ids=_IDS)
def test_provider_live(case, variant, stratix_live_client, record_result) -> None:
    pytest.importorskip(case.import_name, reason=f"{case.id}: '{case.import_name}' not installed")
    reason = missing_credentials(case)
    if reason:
        pytest.skip(f"{case.id}: {reason}")

    row = run_case(stratix_live_client, case, variant)
    record_result(row)
