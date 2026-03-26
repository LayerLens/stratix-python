"""Tests for automatic signing key fetch in @trace decorator."""

from __future__ import annotations

import json
import base64
import asyncio
import threading
from unittest.mock import Mock, AsyncMock

import pytest

from layerlens.instrument import trace, clear_signing_key_cache
from layerlens.instrument._recorder import _signing_key_cache, _resolve_signing_key


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear signing key cache before and after each test."""
    _signing_key_cache.clear()
    yield
    _signing_key_cache.clear()


def _make_client(*, signing_key_response=None, create_key_response=None):
    """Create a mock client with optional signing_keys.get_active() response.

    If create_key_response is provided, signing_keys.create() returns it.
    Otherwise create() returns None (simulating backend failure or no-op).
    """
    client = Mock()
    client.traces = Mock()
    client.traces.upload = Mock()
    client.signing_keys = Mock()
    if signing_key_response is not None:
        client.signing_keys.get_active = Mock(return_value=signing_key_response)
    else:
        client.signing_keys.get_active = Mock(return_value=None)
    if create_key_response is not None:
        client.signing_keys.create = Mock(return_value=create_key_response)
    else:
        client.signing_keys.create = Mock(return_value=None)
    return client


def _capture_upload(client):
    """Set up trace capture on the mock client. Returns dict that gets populated."""
    uploaded = {}

    def _capture(path):
        with open(path) as f:
            uploaded["trace"] = json.load(f)

    client.traces.upload.side_effect = _capture
    return uploaded


class TestAutoFetchSigningKey:
    def test_auto_fetches_and_signs(self):
        """When no signing_service passed, auto-fetch from client and sign."""
        secret = b"test-auto-key-32-bytes-long!!!!!"
        client = _make_client(
            signing_key_response={
                "key_id": "sk_auto_123",
                "name": "auto-key",
                "secret": base64.b64encode(secret).decode(),
            }
        )
        uploaded = _capture_upload(client)

        @trace(client)
        def my_agent():
            return "hello"

        my_agent()

        client.signing_keys.get_active.assert_called_once()
        payload = uploaded["trace"][0]
        assert "attestation" in payload
        att = payload["attestation"]
        # Chain events should have signatures
        events = att["chain"]["events"]
        assert len(events) > 0
        for event in events:
            assert "signature" in event, "Event should be signed"
            assert event["signing_key_id"] == "sk_auto_123"

    def test_auto_creates_key_when_none_exists(self):
        """When org has no active key, SDK auto-creates one and signs."""
        secret = b"auto-created-key-32-bytes!!!!!!!"
        client = _make_client(
            signing_key_response=None,
            create_key_response={
                "key_id": "sk_auto_created",
                "name": "default",
                "secret": base64.b64encode(secret).decode(),
            },
        )
        uploaded = _capture_upload(client)

        @trace(client)
        def my_agent():
            return "hello"

        my_agent()

        client.signing_keys.get_active.assert_called_once()
        client.signing_keys.create.assert_called_once()
        payload = uploaded["trace"][0]
        assert "attestation" in payload
        events = payload["attestation"]["chain"]["events"]
        assert len(events) > 0
        for event in events:
            assert "signature" in event, "Event should be signed with auto-created key"
            assert event["signing_key_id"] == "sk_auto_created"

    def test_no_signing_key_and_create_fails_produces_unsigned(self):
        """When org has no active key AND create fails, traces are unsigned."""
        client = _make_client(signing_key_response=None, create_key_response=None)
        uploaded = _capture_upload(client)

        @trace(client)
        def my_agent():
            return "hello"

        my_agent()

        client.signing_keys.get_active.assert_called_once()
        client.signing_keys.create.assert_called_once()
        payload = uploaded["trace"][0]
        assert "attestation" in payload
        events = payload["attestation"]["chain"]["events"]
        assert len(events) > 0
        for event in events:
            assert "signature" not in event, "Event should NOT be signed"

    def test_caches_across_traces(self):
        """Signing key is fetched once and reused across multiple @trace calls."""
        secret = b"cached-key-32-bytes-long!!!!!!!!"
        client = _make_client(
            signing_key_response={
                "key_id": "sk_cached",
                "name": "cached",
                "secret": base64.b64encode(secret).decode(),
            }
        )
        all_uploads: list = []

        def _capture_all(path):
            with open(path) as f:
                all_uploads.append(json.load(f))

        client.traces.upload.side_effect = _capture_all

        @trace(client)
        def agent_a():
            return "a"

        @trace(client)
        def agent_b():
            return "b"

        agent_a()
        agent_b()

        # Only one API call despite two traces
        client.signing_keys.get_active.assert_called_once()
        # Both traces should be signed
        assert len(all_uploads) == 2
        for upload in all_uploads:
            events = upload[0]["attestation"]["chain"]["events"]
            assert "signature" in events[0]
            assert events[0]["signing_key_id"] == "sk_cached"

    def test_clear_cache_forces_refetch(self):
        """clear_signing_key_cache() causes next trace to refetch."""
        secret = b"key-before-rotation!!!!!!!!!!!!!!"
        client = _make_client(
            signing_key_response={
                "key_id": "sk_old",
                "name": "old",
                "secret": base64.b64encode(secret).decode(),
            }
        )
        _capture_upload(client)

        @trace(client)
        def my_agent():
            return "hello"

        my_agent()
        assert client.signing_keys.get_active.call_count == 1

        # Simulate key rotation
        clear_signing_key_cache(client)
        new_secret = b"key-after-rotation!!!!!!!!!!!!!!!"
        client.signing_keys.get_active.return_value = {
            "key_id": "sk_new",
            "name": "new",
            "secret": base64.b64encode(new_secret).decode(),
        }

        my_agent()
        assert client.signing_keys.get_active.call_count == 2

    def test_explicit_signing_key_skips_autofetch(self):
        """Passing signing_service= explicitly bypasses auto-fetch entirely."""
        client = _make_client(
            signing_key_response={
                "key_id": "sk_should_not_fetch",
                "name": "nope",
                "secret": base64.b64encode(b"nope").decode(),
            }
        )
        uploaded = _capture_upload(client)

        @trace(client, signing_service=("explicit-key", b"explicit-secret"))
        def my_agent():
            return "hello"

        my_agent()

        # Auto-fetch should NOT be called
        client.signing_keys.get_active.assert_not_called()
        # But traces should still be signed with the explicit key
        payload = uploaded["trace"][0]
        events = payload["attestation"]["chain"]["events"]
        assert events[0]["signing_key_id"] == "explicit-key"

    def test_explicit_none_disables_signing(self):
        """Passing signing_service=None explicitly disables signing (no auto-fetch)."""
        client = _make_client(
            signing_key_response={
                "key_id": "sk_should_not_fetch",
                "name": "nope",
                "secret": base64.b64encode(b"nope").decode(),
            }
        )
        uploaded = _capture_upload(client)

        @trace(client, signing_service=None)
        def my_agent():
            return "hello"

        my_agent()

        # Auto-fetch should NOT be called
        client.signing_keys.get_active.assert_not_called()
        # Traces should be unsigned
        payload = uploaded["trace"][0]
        events = payload["attestation"]["chain"]["events"]
        for event in events:
            assert "signature" not in event

    def test_fetch_failure_degrades_to_unsigned(self):
        """If get_active() throws, traces are uploaded unsigned (not broken)."""
        client = _make_client()
        client.signing_keys.get_active = Mock(side_effect=RuntimeError("network error"))
        uploaded = _capture_upload(client)

        @trace(client)
        def my_agent():
            return "hello"

        my_agent()

        payload = uploaded["trace"][0]
        assert "attestation" in payload
        events = payload["attestation"]["chain"]["events"]
        for event in events:
            assert "signature" not in event

    def test_client_without_signing_keys_attr(self):
        """Clients that don't have signing_keys (e.g. old SDK) degrade gracefully."""
        client = Mock(spec=["traces"])
        client.traces = Mock()
        client.traces.upload = Mock()
        uploaded = {}

        def _capture(path):
            with open(path) as f:
                uploaded["trace"] = json.load(f)

        client.traces.upload.side_effect = _capture

        @trace(client)
        def my_agent():
            return "hello"

        my_agent()

        payload = uploaded["trace"][0]
        assert "attestation" in payload
        # Should be unsigned, no crash
        events = payload["attestation"]["chain"]["events"]
        for event in events:
            assert "signature" not in event

    def test_malformed_response_missing_key_id(self):
        """get_active() returns dict with secret but no key_id — falls back to create()."""
        client = _make_client(
            signing_key_response={
                "name": "broken-key",
                "secret": base64.b64encode(b"some-secret").decode(),
                # Missing "key_id"
            },
            create_key_response=None,  # create also fails
        )
        uploaded = _capture_upload(client)

        @trace(client)
        def my_agent():
            return "hello"

        my_agent()

        client.signing_keys.create.assert_called_once()
        payload = uploaded["trace"][0]
        events = payload["attestation"]["chain"]["events"]
        for event in events:
            assert "signature" not in event, "Should be unsigned when key_id is missing and create fails"

    def test_malformed_response_missing_secret(self):
        """get_active() returns dict with key_id but no secret — falls back to create()."""
        client = _make_client(
            signing_key_response={
                "key_id": "sk_123",
                "name": "broken-key",
                # Missing "secret"
            },
            create_key_response=None,  # create also fails
        )
        uploaded = _capture_upload(client)

        @trace(client)
        def my_agent():
            return "hello"

        my_agent()

        payload = uploaded["trace"][0]
        events = payload["attestation"]["chain"]["events"]
        for event in events:
            assert "signature" not in event, "Should be unsigned when secret is missing"

    def test_concurrent_cache_access(self):
        """Multiple threads resolving the same client only fetch once."""
        secret = b"concurrent-key-32-bytes-long!!!!!"
        client = _make_client(
            signing_key_response={
                "key_id": "sk_concurrent",
                "name": "concurrent",
                "secret": base64.b64encode(secret).decode(),
            }
        )

        results = []
        errors = []

        def resolve():
            try:
                result = _resolve_signing_key(client)
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=resolve) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(results) == 10
        # All threads got the same result
        for r in results:
            assert r is not None
            assert r[0] == "sk_concurrent"


class TestAsyncSigningFlush:
    def test_async_trace_signs_correctly(self):
        """Async @trace path produces signed attestation."""
        secret = b"async-test-key-32-bytes-long!!!!!"
        client = _make_client(
            signing_key_response={
                "key_id": "sk_async",
                "name": "async-key",
                "secret": base64.b64encode(secret).decode(),
            }
        )
        uploaded = {}

        async def _async_capture(path):
            with open(path) as f:
                uploaded["trace"] = json.load(f)

        client.traces.upload = AsyncMock(side_effect=_async_capture)

        @trace(client)
        async def my_async_agent():
            return "hello async"

        asyncio.run(my_async_agent())

        payload = uploaded["trace"][0]
        assert "attestation" in payload
        events = payload["attestation"]["chain"]["events"]
        assert len(events) > 0
        for event in events:
            assert "signature" in event
            assert event["signing_key_id"] == "sk_async"
