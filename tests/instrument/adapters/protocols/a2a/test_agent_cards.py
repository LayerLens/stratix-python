"""
Tests for A2A Agent Card discovery and event emission.
"""

import pytest

from layerlens.instrument.adapters.protocols.a2a.agent_card import (
    parse_agent_card,
    discover_agent_card,
)


class TestAgentCardParsing:
    def test_parse_from_dict(self):
        card_data = {
            "name": "TestAgent",
            "description": "A test agent",
            "url": "http://agent.test",
            "protocolVersion": "0.2.1",
            "capabilities": {"streaming": True},
            "skills": [{"id": "s1", "name": "Search"}],
            "authentication": {"scheme": "bearer"},
        }
        result = parse_agent_card(card_data)
        assert result["name"] == "TestAgent"
        assert result["protocolVersion"] == "0.2.1"
        assert result["authScheme"] == "bearer"
        assert len(result["skills"]) == 1

    def test_parse_from_json_string(self):
        import json
        card_data = {"name": "JSONAgent", "url": "http://agent.test", "version": "1.0.0"}
        result = parse_agent_card(json.dumps(card_data))
        assert result["name"] == "JSONAgent"

    def test_parse_invalid_json(self):
        with pytest.raises(ValueError):
            parse_agent_card("{invalid json")

    def test_parse_defaults(self):
        result = parse_agent_card({})
        assert result["name"] == "unknown"
        assert result["protocolVersion"] == "unknown"


class TestAgentCardRegistration:
    def test_register_agent_card(self, a2a_adapter, mock_stratix):
        card_data = {
            "name": "DiscoveredAgent",
            "description": "Discovered via /.well-known/agent.json",
            "url": "http://discovered-agent.test",
            "protocolVersion": "0.2.1",
            "capabilities": {"streaming": True},
            "skills": [
                {
                    "id": "search",
                    "name": "Web Search",
                    "description": "Search the web",
                    "tags": ["search"],
                    "examples": ["Search for X"],
                }
            ],
            "authentication": {"scheme": "bearer"},
        }
        a2a_adapter.register_agent_card(card_data, source="discovery")
        assert len(mock_stratix.events) == 1
        event = mock_stratix.events[0][0]
        assert event.event_type == "protocol.agent_card"
        assert event.card.name == "DiscoveredAgent"
        assert len(event.card.skills) == 1
        assert event.card.source == "discovery"

    def test_register_card_updates_internal_cache(self, a2a_adapter):
        card_data = {"name": "CachedAgent", "url": "http://cached.test"}
        a2a_adapter.register_agent_card(card_data)
        assert "CachedAgent" in a2a_adapter._agent_cards
