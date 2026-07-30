"""Contract-drift fixes for the provider emit path (2026-07-06 audit).

S11/F2: providers emit flat prompt_tokens/completion_tokens/total_tokens on
model.invoke *beside* the nested `usage` block, so the atlas extractor (which
reads flat top-level token keys, never `usage.*`) fills the tokens_total column.

S19/F12: providers stamp payload.framework = <integration name> so the
framework column reflects the integration (litellm -> 'litellm',
azure_openai -> 'azure_openai') rather than the routed/underlying provider.
"""

from __future__ import annotations

from unittest.mock import Mock

from layerlens.instrument import trace
from layerlens.instrument._capture_config import CaptureConfig
from layerlens.instrument.adapters.providers.openai import OpenAIProvider
from layerlens.instrument.adapters.providers.anthropic import AnthropicProvider

from .conftest import make_openai_response, make_anthropic_response
from ...conftest import find_event

# ---------------------------------------------------------------------------
# S11/F2 — flat token keys beside nested usage
# ---------------------------------------------------------------------------


class TestFlatTokensS11:
    def test_openai_emits_flat_tokens_beside_usage(self, mock_client, capture_trace):
        client = Mock()
        client.chat.completions.create = Mock(return_value=make_openai_response())

        provider = OpenAIProvider()
        provider.connect(client)

        @trace(mock_client, capture_config=CaptureConfig.full())
        def run():
            client.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": "Hi"}])

        run()
        mi = find_event(capture_trace["events"], "model.invoke")["payload"]

        # Flat keys (what the atlas extractor reads) present at top level.
        assert mi["prompt_tokens"] == 10
        assert mi["completion_tokens"] == 5
        assert mi["total_tokens"] == 15
        # Nested usage stays byte-identical (no regression).
        assert mi["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    def test_anthropic_sums_total_when_provider_omits_it(self, mock_client, capture_trace):
        client = Mock()
        client.messages.create = Mock(return_value=make_anthropic_response(input_tokens=20, output_tokens=10))

        provider = AnthropicProvider()
        provider.connect(client)

        @trace(mock_client, capture_config=CaptureConfig.full())
        def run():
            client.messages.create(model="claude-3-opus-20240229", messages=[{"role": "user", "content": "Hi"}])

        run()
        mi = find_event(capture_trace["events"], "model.invoke")["payload"]

        # Anthropic's usage has no total_tokens; flat total is the honest sum.
        assert mi["prompt_tokens"] == 20
        assert mi["completion_tokens"] == 10
        assert mi["total_tokens"] == 30
        # Nested usage still carries no fabricated total_tokens.
        assert "total_tokens" not in mi["usage"]

    def test_no_flat_tokens_when_no_usage(self, mock_client, capture_trace):
        from .conftest import make_openai_response_no_usage

        client = Mock()
        client.chat.completions.create = Mock(return_value=make_openai_response_no_usage())

        provider = OpenAIProvider()
        provider.connect(client)

        @trace(mock_client, capture_config=CaptureConfig.full())
        def run():
            client.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": "Hi"}])

        run()
        mi = find_event(capture_trace["events"], "model.invoke")["payload"]

        # Honest blank: no usage declared -> no flat token keys fabricated.
        assert "prompt_tokens" not in mi
        assert "completion_tokens" not in mi
        assert "total_tokens" not in mi


# ---------------------------------------------------------------------------
# S19/F12 — framework stamp = integration name (not routed/underlying provider)
# ---------------------------------------------------------------------------


class TestFrameworkStampS19:
    def test_openai_stamps_framework_on_all_events(self, mock_client, capture_trace):
        client = Mock()
        client.chat.completions.create = Mock(return_value=make_openai_response())

        provider = OpenAIProvider()
        provider.connect(client)

        @trace(mock_client, capture_config=CaptureConfig.full())
        def run():
            client.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": "Hi"}])

        run()
        events = capture_trace["events"]
        assert find_event(events, "model.invoke")["payload"]["framework"] == "openai"
        cost = find_event(events, "cost.record")["payload"]
        assert cost["framework"] == "openai"
        # cost.record.provider (the honest underlying provider) is unchanged.
        assert cost["provider"] == "openai"

    def test_anthropic_stamps_framework(self, mock_client, capture_trace):
        client = Mock()
        client.messages.create = Mock(return_value=make_anthropic_response())

        provider = AnthropicProvider()
        provider.connect(client)

        @trace(mock_client, capture_config=CaptureConfig.full())
        def run():
            client.messages.create(model="claude-3-opus-20240229", messages=[{"role": "user", "content": "Hi"}])

        run()
        assert find_event(capture_trace["events"], "model.invoke")["payload"]["framework"] == "anthropic"
