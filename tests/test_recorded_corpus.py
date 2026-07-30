"""Guards for the recorded-real-response corpus (LAY-3614).

Two invariants every committed fixture under ``tests/fixtures/recorded/`` must
hold, independent of any adapter replay test:

* **No secret leaks** — capture-time scrubbing must have removed every API key,
  bearer/JWT token, AWS access key, and real account id. A fixture is committed
  to the repo, so a leak here is a real disclosure.
* **Provenance stamped** — each fixture records ``{provider, sdk_version, model,
  scenario, captured_at}`` so staleness is visible later (the corpus is a
  snapshot, not a freshness check — see ``tests/instrument/_recorded.py``).
"""

from __future__ import annotations

import json

import pytest

from tests.instrument._recorded import (
    PROVENANCE_KEYS,
    find_secret_leaks,
    iter_recorded_files,
)

_VALID_TRANSPORTS = {"http", "boto3", "eventstream", "object"}

_FIXTURES = iter_recorded_files()


def test_corpus_is_non_empty() -> None:
    """Guards against a vacuous pass: the secret-scan below is meaningless on an
    empty corpus, so prove fixtures actually exist."""
    assert _FIXTURES, "no recorded fixtures found under tests/fixtures/recorded/"


@pytest.mark.parametrize("path", _FIXTURES, ids=[str(p.relative_to(p.parents[2])) for p in _FIXTURES])
def test_no_secret_leaks(path) -> None:
    leaks = find_secret_leaks(path.read_text())
    assert not leaks, f"secret pattern leaked into committed fixture {path}: {leaks}"


@pytest.mark.parametrize("path", _FIXTURES, ids=[str(p.relative_to(p.parents[2])) for p in _FIXTURES])
def test_provenance_and_transport(path) -> None:
    fixture = json.loads(path.read_text())
    prov = fixture.get("provenance", {})
    missing = [k for k in PROVENANCE_KEYS if k not in prov]
    assert not missing, f"{path} missing provenance keys: {missing}"
    assert fixture.get("transport") in _VALID_TRANSPORTS, f"{path} has unknown transport {fixture.get('transport')!r}"
