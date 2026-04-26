# Protocol adapter certification suite

`layerlens.instrument.adapters.protocols.certification.ProtocolCertificationSuite`
runs GA-readiness checks against protocol adapter classes. It validates
that an adapter class complies with the `BaseProtocolAdapter` contract
required for General Availability release.

This is a developer/CI tool — there's no runnable end-user sample. The
suite is invoked from your certification CI to gate adapter releases.

## Install

The certification suite is part of the base SDK install — no extra is
required:

```bash
pip install layerlens
```

## Quick start

```python
from layerlens.instrument.adapters.protocols.a2a import A2AAdapter
from layerlens.instrument.adapters.protocols.certification import (
    ProtocolCertificationSuite,
)

suite = ProtocolCertificationSuite()
result = suite.certify(A2AAdapter)
assert result.passed, result.summary()

# Or certify all GA protocol adapters at once:
results = suite.certify_all()
assert all(r.passed for r in results)
```

## What gets checked

The suite runs the following categories of checks against every
candidate adapter class:

| Category | What's verified |
|---|---|
| Inheritance | Class extends both `BaseAdapter` and `BaseProtocolAdapter`. |
| Required class attributes | `FRAMEWORK`, `PROTOCOL`, `PROTOCOL_VERSION`, `VERSION` are non-empty strings. |
| Required methods | `connect`, `disconnect`, `health_check`, `get_adapter_info`, `serialize_for_replay`, `probe_health` all defined and not abstract. |
| Lifecycle correctness | `connect()` then `disconnect()` succeeds without exception, leaves the adapter in `DISCONNECTED` state. |
| Error handling | Adapter does not raise on construction with default args; `health_check()` returns an `AdapterHealth` even before `connect()`. |
| Return-type contracts | `get_adapter_info()` returns `AdapterInfo`, `probe_health()` returns `dict`, `serialize_for_replay()` returns `ReplayableTrace`. |

Each check produces a `CheckResult` with `passed: bool`, `message: str`,
and `severity: "error" | "warning"`. Warnings do not fail the
certification, errors do.

## Result types

```python
@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str
    severity: str  # "error" | "warning"

@dataclass
class CertificationResult:
    passed: bool
    adapter_name: str
    protocol_version: str
    checks: list[dict[str, Any]]   # serialised CheckResult entries

    def summary(self) -> str: ...
```

## Integrating into CI

A typical CI step:

```python
# tests/instrument/test_protocol_certification.py
from layerlens.instrument.adapters.protocols.certification import (
    ProtocolCertificationSuite,
)


def test_all_protocol_adapters_pass_ga_certification() -> None:
    suite = ProtocolCertificationSuite()
    results = suite.certify_all()

    failures = [r for r in results if not r.passed]
    assert not failures, "\n".join(r.summary() for r in failures)
```

`certify_all()` covers the three current GA protocol adapters:

- `A2AAdapter`
- `AGUIAdapter`
- `MCPExtensionsAdapter`

For commerce-protocol adapters (AP2, A2UI, UCP) certify each one
explicitly with `suite.certify(MyAdapterClass)` — they share the same
`BaseProtocolAdapter` contract.

## Adding a new check

To add a new check to the suite, append a private `_check_*` method that
returns a `CheckResult` (or `list[CheckResult]`) and call it from
`certify()`. Keep the contract narrow: each check should test one
invariant and produce a clear failure message naming the adapter class.

## What this suite does NOT verify

- Per-protocol semantic correctness (does A2A actually emit the right
  events for the protocol?). That belongs in the per-adapter unit and
  live tests under `tests/instrument/adapters/protocols/`.
- Performance under load. The suite runs a single `connect()`/`disconnect()`
  pair — load-testing is out of scope.
- Backward compatibility. Use the schema-compatibility test in
  `tests/instrument/test_event_schema_compat.py` for that.
