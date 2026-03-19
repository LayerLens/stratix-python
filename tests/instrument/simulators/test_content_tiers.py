"""Tests for Tier 1 (Seed), Tier 3 (LLM), and ContentCache."""

import json
import tempfile
import os

import pytest

from layerlens.instrument.simulators.content.cache import ContentCache
from layerlens.instrument.simulators.content.llm_provider import LLMContentProvider
from layerlens.instrument.simulators.content.seed_provider import SeedContentProvider


class TestContentCache:
    def setup_method(self):
        self._tmpdir = tempfile.mkdtemp()
        self.cache = ContentCache(cache_dir=self._tmpdir)

    def test_set_and_get(self):
        self.cache.set("hello world", key="test_key")
        result = self.cache.get(key="test_key")
        assert result == "hello world"

    def test_get_missing(self):
        assert self.cache.get(key="nonexistent") is None

    def test_dict_values(self):
        data = {"name": "test", "items": [1, 2, 3]}
        self.cache.set(data, scenario="sales", topic="pricing")
        result = self.cache.get(scenario="sales", topic="pricing")
        assert result == data

    def test_cache_dir_created(self):
        assert self.cache.cache_dir.exists()

    def test_clear(self):
        self.cache.set("a", key="1")
        self.cache.set("b", key="2")
        count = self.cache.clear()
        assert count == 2
        assert self.cache.get(key="1") is None
        assert self.cache.size == 0

    def test_size(self):
        assert self.cache.size == 0
        self.cache.set("a", key="1")
        assert self.cache.size == 1
        self.cache.set("b", key="2")
        assert self.cache.size == 2

    def test_deterministic_keys(self):
        cache1 = ContentCache(cache_dir=self._tmpdir)
        cache2 = ContentCache(cache_dir=self._tmpdir)
        cache1.set("test_value", scenario="sales", topic="pricing")
        result = cache2.get(scenario="sales", topic="pricing")
        assert result == "test_value"

    def test_memory_cache(self):
        self.cache.set("fast", key="memory_test")
        # Delete disk file to test memory-only path
        for f in self.cache.cache_dir.glob("*.json"):
            f.unlink()
        result = self.cache.get(key="memory_test")
        assert result == "fast"


class TestSeedContentProvider:
    def test_nonexistent_path(self):
        provider = SeedContentProvider(
            seed_data_path="/nonexistent/path",
            seed=42,
        )
        # Should not crash, just return fallback content
        msg = provider.get_user_message("customer_service", "test_topic")
        assert isinstance(msg, str) and len(msg) > 0

    def test_loaded_scenarios_empty(self):
        provider = SeedContentProvider(
            seed_data_path="/nonexistent/path",
            seed=42,
        )
        assert provider.loaded_scenarios == []
        assert provider.trace_count == 0

    def test_with_mock_seed_data(self):
        tmpdir = tempfile.mkdtemp()
        # Create mock Langfuse trace data
        scenario_dir = os.path.join(tmpdir, "scenario_customer_service", "langfuse")
        os.makedirs(scenario_dir)

        trace_data = [
            {
                "id": "trace_001",
                "name": "test",
                "metadata": {"topic": "Shipping_Delay"},
                "tags": ["customer_service", "Shipping_Delay"],
                "observations": [
                    {
                        "type": "GENERATION",
                        "input": [
                            {"role": "system", "content": "You are a helpful agent."},
                            {"role": "user", "content": "Where is my order?"},
                        ],
                        "output": {"role": "assistant", "content": "Let me check that for you."},
                    }
                ],
            }
        ]
        with open(os.path.join(scenario_dir, "traces.json"), "w") as f:
            json.dump(trace_data, f)

        provider = SeedContentProvider(seed_data_path=tmpdir, seed=42)
        assert "customer_service" in provider.loaded_scenarios
        assert provider.trace_count >= 1

        msg = provider.get_user_message("customer_service", "Shipping_Delay")
        assert isinstance(msg, str) and len(msg) > 0

        resp = provider.get_agent_response("customer_service", "Shipping_Delay")
        assert isinstance(resp, str) and len(resp) > 0

        prompt = provider.get_system_prompt("customer_service", "Test_Agent")
        assert isinstance(prompt, str) and len(prompt) > 0

    def test_fallback_content(self):
        provider = SeedContentProvider(
            seed_data_path="/nonexistent",
            seed=42,
        )
        msg = provider.get_user_message("sales", "Pricing_Inquiry")
        assert "pricing inquiry" in msg.lower()

        resp = provider.get_agent_response("sales", "Pricing_Inquiry")
        assert isinstance(resp, str)


class TestLLMContentProvider:
    def test_fallback_to_template(self):
        """Without API key, should fall back to template provider."""
        provider = LLMContentProvider(
            model="gpt-4o-mini",
            cache_enabled=False,
            api_key=None,
            seed=42,
        )
        msg = provider.get_user_message("customer_service", "Shipping_Delay")
        assert isinstance(msg, str) and len(msg) > 10

    def test_get_agent_response_fallback(self):
        provider = LLMContentProvider(cache_enabled=False, seed=42)
        resp = provider.get_agent_response("customer_service", "Shipping_Delay")
        assert isinstance(resp, str) and len(resp) > 10

    def test_get_system_prompt_fallback(self):
        provider = LLMContentProvider(cache_enabled=False, seed=42)
        prompt = provider.get_system_prompt("customer_service", "Test_Agent")
        assert isinstance(prompt, str) and len(prompt) > 0

    def test_get_tool_input(self):
        provider = LLMContentProvider(cache_enabled=False, seed=42)
        tool_input = provider.get_tool_input("Get_Order_Details", "Shipping_Delay")
        assert isinstance(tool_input, dict)

    def test_get_topics(self):
        provider = LLMContentProvider(cache_enabled=False, seed=42)
        topics = provider.get_topics("customer_service")
        assert len(topics) == 5

    def test_get_agent_names(self):
        provider = LLMContentProvider(cache_enabled=False, seed=42)
        names = provider.get_agent_names("customer_service")
        assert len(names) >= 1

    def test_cache_integration(self):
        tmpdir = tempfile.mkdtemp()
        provider = LLMContentProvider(
            cache_enabled=True,
            cache_path=tmpdir,
            seed=42,
        )
        # First call falls back to template
        msg1 = provider.get_user_message("sales", "Trial_Extension")
        assert isinstance(msg1, str)
