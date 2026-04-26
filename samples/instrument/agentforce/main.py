"""Sample: smoke-test the LayerLens AgentForce adapter wiring.

This sample is **smoke-only**. A real ``import_sessions`` call requires:

1. A Salesforce Connected App with the JWT Bearer flow configured.
2. A private key (PEM) authorized for the Connected App.
3. A user with read access to the AIAgentSession DMOs.

Most CI environments don't have those, so this sample only exercises the
adapter's local wiring (instantiate, attach sink, demonstrate the
configuration flow) and exits cleanly. If the required ``SALESFORCE_*``
env vars are present, it will attempt one ``connect()`` call to verify auth.

Required environment for the smoke run:

* (none — the sample exits cleanly without any env vars)

Optional environment for the auth check:

* ``SALESFORCE_CLIENT_ID`` — Connected App consumer key.
* ``SALESFORCE_USERNAME`` — Salesforce user the JWT is issued for.
* ``SALESFORCE_PRIVATE_KEY`` — PEM-encoded private key.
* ``SALESFORCE_INSTANCE_URL`` — your org's My Domain URL
  (e.g. ``https://example.my.salesforce.com``).
* ``LAYERLENS_STRATIX_API_KEY`` — your LayerLens API key (optional).
* ``LAYERLENS_STRATIX_BASE_URL`` — atlas-app base URL (optional).

Run::

    pip install 'layerlens[agentforce]'
    python -m samples.instrument.agentforce.main
"""

from __future__ import annotations

import os
import sys

from layerlens.instrument.adapters._base import CaptureConfig
from layerlens.instrument.transport.sink_http import HttpEventSink
from layerlens.instrument.adapters.frameworks.agentforce import (
    AgentForceAdapter,
    SalesforceAuthError,
    SalesforceCredentials,
)


def _have_salesforce_env() -> bool:
    return all(
        os.environ.get(name)
        for name in (
            "SALESFORCE_CLIENT_ID",
            "SALESFORCE_USERNAME",
            "SALESFORCE_PRIVATE_KEY",
        )
    )


def main() -> int:
    sink = HttpEventSink(
        adapter_name="salesforce_agentforce",
        path="/telemetry/spans",
        max_batch=10,
        flush_interval_s=1.0,
    )

    if not _have_salesforce_env():
        print(
            "SALESFORCE_* env vars are not set; running smoke check only.",
            file=sys.stderr,
        )
        # Smoke check: verify the adapter can be constructed and shut down
        # without performing any network I/O.
        adapter = AgentForceAdapter(capture_config=CaptureConfig.standard())
        adapter.add_sink(sink)
        info = adapter.get_adapter_info()
        print(f"Adapter: {info.name} v{info.version} (framework={info.framework})")
        sink.close()
        return 0

    credentials = SalesforceCredentials(
        client_id=os.environ["SALESFORCE_CLIENT_ID"],
        username=os.environ["SALESFORCE_USERNAME"],
        private_key=os.environ["SALESFORCE_PRIVATE_KEY"],
        instance_url=os.environ.get(
            "SALESFORCE_INSTANCE_URL", "https://login.salesforce.com"
        ),
    )

    adapter = AgentForceAdapter(
        credentials=credentials,
        capture_config=CaptureConfig.standard(),
    )
    adapter.add_sink(sink)

    try:
        adapter.connect()
        print("AgentForce adapter authenticated against Salesforce.")
        # An import call would look like:
        # result = adapter.import_sessions(start_date="2026-04-01", limit=10)
        # print(f"Imported {result.events_generated} events.")
    except SalesforceAuthError as exc:
        print(f"Salesforce auth failed: {exc}", file=sys.stderr)
        return 1
    finally:
        sink.close()
        adapter.disconnect()

    print("Telemetry shipped (smoke). Check the LayerLens dashboard.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
