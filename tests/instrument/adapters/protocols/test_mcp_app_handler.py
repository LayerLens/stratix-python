from __future__ import annotations

from layerlens.instrument.adapters.protocols.mcp.mcp_app_handler import (
    COMPONENT_TYPES,
    INTERACTION_RESULTS,
    hash_result,
    hash_parameters,
    build_invocation_payload,
    normalize_component_type,
    build_interaction_payload,
    normalize_interaction_result,
)


class TestHashing:
    def test_hash_stable_across_key_order(self):
        a = hash_parameters({"a": 1, "b": 2})
        b = hash_parameters({"b": 2, "a": 1})
        assert a == b
        assert a.startswith("sha256:")

    def test_none_parameters_hash_like_empty_dict(self):
        assert hash_parameters(None) == hash_parameters({})

    def test_hash_result_none_returns_none(self):
        assert hash_result(None) is None

    def test_hash_result_handles_non_json_natively(self):
        # `default=str` lets datetime-like objects through without raising.
        class Weird:
            def __repr__(self):
                return "W"

        h = hash_result({"x": Weird()})
        assert h is not None and h.startswith("sha256:")


class TestNormalizers:
    def test_component_type_lowercased(self):
        assert normalize_component_type("Form") == "form"

    def test_unknown_component_becomes_custom(self):
        assert normalize_component_type("slider") == "custom"

    def test_every_known_component_preserved(self):
        for known in COMPONENT_TYPES:
            assert normalize_component_type(known) == known

    def test_interaction_result_defaults_to_submitted(self):
        assert normalize_interaction_result("weird") == "submitted"

    def test_empty_interaction_defaults_to_submitted(self):
        assert normalize_interaction_result("") == "submitted"

    def test_every_known_interaction_preserved(self):
        for known in INTERACTION_RESULTS:
            assert normalize_interaction_result(known) == known


class TestInvocationPayload:
    def test_builds_expected_fields(self):
        payload = build_invocation_payload(
            app_id="app-1",
            component_type="form",
            parameters={"email": "x@y"},
            server_name="svr",
        )
        assert payload["app_id"] == "app-1"
        assert payload["component_type"] == "form"
        assert payload["server_name"] == "svr"
        assert payload["parameters_hash"].startswith("sha256:")


class TestInteractionPayload:
    def test_includes_result_hash_and_latency(self):
        payload = build_interaction_payload(
            app_id="app-1",
            interaction_result="submitted",
            result={"answer": "yes"},
            latency_ms=12.3,
        )
        assert payload["interaction_result"] == "submitted"
        assert payload["result_hash"].startswith("sha256:")
        assert payload["latency_ms"] == 12.3

    def test_omits_optional_fields_when_absent(self):
        payload = build_interaction_payload(app_id="app-1", interaction_result="cancelled")
        assert "result_hash" not in payload
        assert "latency_ms" not in payload
