"""Live end-to-end test for the custom-model lifecycle.

Exercises the customer's exact workflow against a real LayerLens API:
create_custom → update_custom (repoint api_url) → delete_custom → verify gone.

Skipped unless ``LAYERLENS_STRATIX_API_KEY`` is set. Run with::

    pytest tests/test_models_custom_live.py -m live
"""

from __future__ import annotations

import os
import time
import uuid

import pytest

from layerlens import Stratix


@pytest.mark.live
def test_custom_model_lifecycle_live() -> None:
    if not os.environ.get("LAYERLENS_STRATIX_API_KEY"):
        pytest.skip("LAYERLENS_STRATIX_API_KEY not set")

    client = Stratix()

    # Use a unique key per run so the test can re-run cleanly.
    suffix = uuid.uuid4().hex[:8]
    name = f"sdk-live-custom-{suffix}"
    key = f"sdk-live/custom-{suffix}"

    created = client.models.create_custom(
        name=name,
        key=key,
        description="ephemeral live-test custom model",
        api_url="https://tunnel-1.example.com/v1",
        api_key="sk-live-test",
        max_tokens=2048,
    )
    assert created is not None, "create_custom returned None"
    model_id = created.model_id
    assert model_id

    try:
        # Repointing api_url is the customer's primary workflow (cloudflared
        # tunnels whose URL changes between sessions).
        updated = client.models.update_custom(
            model_id,
            api_url="https://tunnel-2.example.com/v1",
        )
        assert updated, "update_custom returned False"

        # Allow a brief moment for the backend to persist (S3 yaml regen +
        # Mongo write) — defensive, not strictly required.
        time.sleep(0.5)

        # Tear it down completely.
        deleted = client.models.delete_custom(model_id)
        assert deleted, "delete_custom returned False"

        remaining = client.models.get(type="custom") or []
        assert all(m.id != model_id for m in remaining), f"deleted custom model {model_id} still visible in models.get"
    except Exception:
        # Best-effort teardown on any assertion / API failure mid-test so a
        # broken run doesn't leak project-scoped resources.
        try:
            client.models.delete_custom(model_id)
        except Exception:  # noqa: BLE001
            pass
        raise
