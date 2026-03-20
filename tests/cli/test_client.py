from __future__ import annotations

from unittest.mock import Mock

import pytest

from layerlens.cli._client import _is_uuid, resolve_model, resolve_benchmark


class TestIsUuid:
    """Test UUID detection."""

    def test_valid_uuid(self):
        assert _is_uuid("550e8400-e29b-41d4-a716-446655440000") is True

    def test_valid_uuid_uppercase(self):
        assert _is_uuid("550E8400-E29B-41D4-A716-446655440000") is True

    def test_short_id(self):
        assert _is_uuid("abc123") is False

    def test_mongo_id(self):
        assert _is_uuid("69805c582a3c129a75d168b8") is False

    def test_empty(self):
        assert _is_uuid("") is False

    def test_model_key(self):
        assert _is_uuid("openai/gpt-4o") is False


class TestResolveModel:
    """Test model resolution by ID, key, or name."""

    @pytest.fixture
    def client(self):
        c = Mock()
        c.models = Mock()
        return c

    def test_resolve_by_uuid(self, client):
        """UUID-like identifier tries get_by_id first."""
        model = Mock(id="550e8400-e29b-41d4-a716-446655440000")
        client.models.get_by_id.return_value = model

        result = resolve_model(client, "550e8400-e29b-41d4-a716-446655440000")

        assert result is model
        client.models.get_by_id.assert_called_once()

    def test_resolve_by_key(self, client):
        """Non-UUID identifier tries get_by_key."""
        model = Mock(id="m-1")
        client.models.get_by_key.return_value = model

        result = resolve_model(client, "openai/gpt-4o")

        assert result is model
        client.models.get_by_key.assert_called_once_with("openai/gpt-4o")

    def test_resolve_by_name(self, client):
        """Falls back to name search."""
        model = Mock(id="m-1")
        client.models.get_by_key.return_value = None
        client.models.get.return_value = [model]

        result = resolve_model(client, "GPT-4")

        assert result is model
        client.models.get.assert_called_once_with(name="GPT-4")

    def test_resolve_not_found(self, client):
        """Returns None when model not found."""
        client.models.get_by_key.return_value = None
        client.models.get.return_value = None

        result = resolve_model(client, "nonexistent")

        assert result is None

    def test_resolve_uuid_fallback_to_key(self, client):
        """UUID that fails get_by_id falls back to get_by_key."""
        client.models.get_by_id.return_value = None
        model = Mock(id="m-1")
        client.models.get_by_key.return_value = model

        result = resolve_model(client, "550e8400-e29b-41d4-a716-446655440000")

        assert result is model


class TestResolveBenchmark:
    """Test benchmark resolution by ID, key, or name."""

    @pytest.fixture
    def client(self):
        c = Mock()
        c.benchmarks = Mock()
        return c

    def test_resolve_by_key(self, client):
        bm = Mock(id="b-1")
        client.benchmarks.get_by_key.return_value = bm

        result = resolve_benchmark(client, "arc-agi-2")

        assert result is bm

    def test_resolve_not_found(self, client):
        client.benchmarks.get_by_key.return_value = None
        client.benchmarks.get.return_value = None

        result = resolve_benchmark(client, "nonexistent")

        assert result is None
