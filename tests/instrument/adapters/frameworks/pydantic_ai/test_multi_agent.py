"""Test Pydantic AI adapter multi-agent tracing.

Ported as-is from ``ateam/tests/adapters/pydantic_ai/test_multi_agent.py``.

Translation rules applied:
* ``stratix.sdk.python.adapters.pydantic_ai.lifecycle`` →
  ``layerlens.instrument.adapters.frameworks.pydantic_ai.lifecycle``
* ``stratix.sdk.python.adapters.base`` →
  ``layerlens.instrument.adapters._base``
* ``stratix.sdk.python.adapters.capture.CaptureConfig`` →
  ``layerlens.instrument.adapters._base.CaptureConfig``
* ``stratix.sdk.python.adapters.replay_models.ReplayableTrace`` →
  ``layerlens.instrument.adapters._base.ReplayableTrace``
* ``stratix.sdk.python.adapters.registry._ADAPTER_MODULES`` →
  ``layerlens.instrument.adapters._base.registry._ADAPTER_MODULES``
* The wrapper marker attribute renamed by the source from
  ``_stratix_original`` to ``_layerlens_original``.

Multi-tenancy: per the transitional "stratix attribute" pattern (see
migration doc §2.3 step 2 — keystone PR #118 still DRAFT), the
``MockStratix`` / ``EventCollector`` test stub gets an ``org_id``
attribute. The post-merge sweep PR will rebase to canonical kwarg once
#118 lands.
"""

class TestPydanticAIAdapterMultiAgent:
    def test_handoff_emits_agent_handoff(self, adapter, mock_stratix):
        adapter.on_handoff(
            from_agent="agent_a",
            to_agent="agent_b",
            context="delegation context",
        )
        events = mock_stratix.get_events("agent.handoff")
        assert len(events) == 1
        assert events[0]["payload"]["from_agent"] == "agent_a"
        assert events[0]["payload"]["to_agent"] == "agent_b"

    def test_multiple_handoffs(self, adapter, mock_stratix):
        adapter.on_handoff(from_agent="a", to_agent="b")
        adapter.on_handoff(from_agent="b", to_agent="c")
        events = mock_stratix.get_events("agent.handoff")
        assert len(events) == 2

    def test_agent_config_emitted_once(self, adapter, mock_stratix):
        adapter.emit_dict_event(
            "environment.config",
            {
                "framework": "pydantic_ai",
                "agent_name": "test_agent",
            },
        )
        adapter.emit_dict_event(
            "environment.config",
            {
                "framework": "pydantic_ai",
                "agent_name": "test_agent",
            },
        )
        # Both emit since dedup is in _emit_agent_config, not emit_dict_event
        events = mock_stratix.get_events("environment.config")
        assert len(events) >= 1
