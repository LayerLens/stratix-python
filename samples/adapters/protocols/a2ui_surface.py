"""Sample: A2UI commerce surface events (with PII-safe hashing)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from adapters._shared import capture_events  # type: ignore[import-not-found]

from layerlens.instrument.adapters.protocols.a2ui import A2UIProtocolAdapter


def main() -> None:
    adapter = A2UIProtocolAdapter()
    with capture_events("a2ui"):
        adapter.record_surface_created(surface_id="cart-1", surface_type="cart", item_count=3)
        adapter.record_user_action(
            surface_id="cart-1",
            action_type="add_to_cart",
            context={"sku": "ABC-123", "user_email": "alice@example.com"},
        )


if __name__ == "__main__":
    main()
