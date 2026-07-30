"""A16 deny-by-default redaction of model.invoke.parameters (LAY-3643,
F-L12-002). Under capture_content=False the collector backstop must keep ONLY a
safe metric allowlist, so an unknown content-bearing param can't leak."""

from __future__ import annotations

import pytest

from layerlens.instrument._capture_config import CaptureConfig


@pytest.mark.invariant
class TestParamDenyByDefault:
    def test_unknown_param_is_stripped(self):
        cfg = CaptureConfig(capture_content=False)
        out = cfg.redact_payload(
            "model.invoke",
            {"parameters": {"temperature": 0.5, "my_custom_field": "leaked prompt content"}},
        )
        assert out["parameters"]["temperature"] == 0.5
        assert "my_custom_field" not in out["parameters"]  # deny-by-default: unknown key dropped

    def test_nested_content_in_generation_config_stripped(self):
        cfg = CaptureConfig(capture_content=False)
        out = cfg.redact_payload(
            "model.invoke",
            {
                "parameters": {
                    "generation_config": {"temperature": 0.2, "response_schema": {"d": "secret field descriptions"}}
                }
            },
        )
        gc = out["parameters"]["generation_config"]
        assert gc["temperature"] == 0.2
        assert "response_schema" not in gc  # content sub-key dropped, metric survives

    def test_known_safe_metrics_survive(self):
        cfg = CaptureConfig(capture_content=False)
        params = {
            "model": "m",
            "temperature": 0.5,
            "max_tokens": 100,
            "top_p": 0.9,
            "stream": True,
            "service_tier": "default",
        }
        out = cfg.redact_payload("model.invoke", {"parameters": dict(params)})
        assert out["parameters"] == params

    def test_capture_content_true_keeps_everything(self):
        cfg = CaptureConfig(capture_content=True)
        params = {"temperature": 0.5, "my_custom_field": "x", "tools": [{"name": "t"}]}
        out = cfg.redact_payload("model.invoke", {"parameters": dict(params)})
        assert out["parameters"] == params

    def test_derived_summaries_survive_while_raw_content_stripped(self):
        # The deny-by-default allowlist keeps adapter-DERIVED safe summaries
        # (cardinality/identity/category by shape) while dropping the raw content
        # they were derived from — so observability isn't blinded.
        cfg = CaptureConfig(capture_content=False)
        out = cfg.redact_payload(
            "model.invoke",
            {
                "parameters": {
                    "messages_count": 3,
                    "message_roles": {"user": 2, "assistant": 1},
                    "tools_count": 2,
                    "tool_names": ["a", "b"],
                    "tool_choice_type": "tool",
                    "tool_choice_name": "a",
                    "metadata_user_id": "u1",
                    "has_system": True,
                    "system_length": 42,
                    "messages": [{"role": "user", "content": "secret prompt"}],
                    "system": "secret system prompt",
                    "tools": [{"name": "a", "input_schema": {"x": 1}}],
                    "metadata": {"user_id": "u1", "leaked": "secret"},
                }
            },
        )
        p = out["parameters"]
        for k in (
            "messages_count",
            "message_roles",
            "tools_count",
            "tool_names",
            "tool_choice_type",
            "tool_choice_name",
            "metadata_user_id",
            "has_system",
            "system_length",
        ):
            assert k in p, f"derived summary {k} should survive"
        for k in ("messages", "system", "tools", "metadata"):
            assert k not in p, f"raw content {k} should be stripped"
