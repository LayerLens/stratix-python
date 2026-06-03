"""Pricing-table tests covering LAY-3330 ACs.

The acceptance criteria are:

* Cost calculated dynamically from a pricing table (not inline rates).
* Default pricing covers all current OpenAI models (GPT-4o, mini, turbo,
  GPT-4, GPT-3.5, o1, o1-mini, o3, o3-mini).
* User can override pricing via the ``pricing_table`` argument.
* Fuzzy model-name matching: dated model IDs (``gpt-4o-2024-08-06``,
  ``claude-3-5-sonnet-20241022``) resolve to base-model pricing.
* Unknown models return ``None`` cost gracefully (no error).
"""

from __future__ import annotations

import os
import json
from pathlib import Path

import pytest

from layerlens.instrument.adapters.providers.pricing import (
    PRICING,
    PRICING_OVERRIDE_ENV,
    CostRecord,
    PricingTable,
    calculate_cost,
    reset_pricing_cache,
)
from layerlens.instrument.adapters.providers.token_usage import NormalizedTokenUsage


def _usage(prompt: int = 100, completion: int = 50, cached: int | None = None) -> NormalizedTokenUsage:
    return NormalizedTokenUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
        cached_tokens=cached,
    )


class TestDefaultCoverage:
    @pytest.mark.parametrize(
        "model",
        [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
            "o1",
            "o1-mini",
            "o3",
            "o3-mini",
        ],
    )
    def test_default_pricing_covers_current_openai_models(self, model: str) -> None:
        # LAY-3330 AC: Default pricing covers all current OpenAI models.
        cost = calculate_cost(model, _usage())
        assert cost is not None, f"{model} missing from default PRICING table"
        assert cost > 0


class TestFuzzyMatching:
    def test_openai_dated_iso_suffix_resolves_to_base(self) -> None:
        # gpt-4o-2024-08-06 should match gpt-4o's pricing.
        dated = calculate_cost("gpt-4o-2024-08-06", _usage())
        base = calculate_cost("gpt-4o", _usage())
        assert dated is not None
        assert dated == base

    def test_openai_mini_dated_resolves_to_base(self) -> None:
        dated = calculate_cost("gpt-4o-mini-2024-07-18", _usage())
        base = calculate_cost("gpt-4o-mini", _usage())
        assert dated is not None
        assert dated == base

    def test_anthropic_short_date_suffix_resolves_to_base(self) -> None:
        # claude-3-5-sonnet-20241022 hits the exact entry; ensure another
        # short-date variant (e.g. an unknown dated build) also resolves.
        cost = calculate_cost("claude-3-5-sonnet-20990101", _usage())
        base = calculate_cost("claude-3-5-sonnet", _usage())
        assert cost is not None
        assert cost == base

    def test_longest_prefix_disambiguates_gpt_4o_from_gpt_4(self) -> None:
        # ``gpt-4o-2099-99-99-foo`` regex-strips to ``gpt-4o-2099-99-99-foo``
        # unchanged (the suffix isn't a valid date). The longest-prefix
        # fallback must pick ``gpt-4o`` over ``gpt-4``.
        cost = calculate_cost("gpt-4o-experimental-build", _usage())
        gpt_4o = calculate_cost("gpt-4o", _usage())
        gpt_4 = calculate_cost("gpt-4", _usage())
        assert cost is not None
        assert cost == gpt_4o
        assert cost != gpt_4


class TestUnknownModelsGracefully:
    def test_completely_unknown_model_returns_none(self) -> None:
        assert calculate_cost("totally-fake-model-9000", _usage()) is None

    def test_empty_model_returns_none(self) -> None:
        assert calculate_cost("", _usage()) is None


class TestUserOverrides:
    def test_caller_supplied_pricing_table_takes_precedence(self) -> None:
        # The caller can pass an entirely custom pricing table — no code changes
        # in the library needed (LAY-3327 + LAY-3330 ACs).
        custom = {"my-private-model": {"input": 1.0, "output": 2.0}}
        cost = calculate_cost(
            "my-private-model",
            _usage(prompt=1000, completion=500),
            pricing_table=custom,
        )
        # 1000 * 1.0/1000 + 500 * 2.0/1000 = 1.0 + 1.0 = 2.0
        assert cost == pytest.approx(2.0)

    def test_custom_table_isolates_from_defaults(self) -> None:
        # A custom table that doesn't include a model that exists in PRICING
        # must NOT silently fall through to PRICING — it's an explicit override.
        custom = {"my-private-model": {"input": 1.0, "output": 2.0}}
        assert calculate_cost("gpt-4o", _usage(), pricing_table=custom) is None


class TestCachedTokens:
    def test_cached_tokens_discounted(self) -> None:
        # OpenAI cached tokens are billed at 50% of input rate (per
        # _cached_token_discount). Anthropic gets 90% off; Google 75% off.
        without = calculate_cost("gpt-4o", _usage(prompt=1000, completion=0))
        with_cache = calculate_cost("gpt-4o", _usage(prompt=1000, completion=0, cached=500))
        assert with_cache is not None and without is not None
        # Half of the 1000 prompt tokens are cached at 50% off, so cost drops.
        assert with_cache < without

    def test_anthropic_cached_tokens_steeper_discount(self) -> None:
        # 90% off for Claude; ensure the function applies the right discount.
        cost = calculate_cost(
            "claude-3-5-sonnet",
            _usage(prompt=1000, completion=0, cached=1000),
        )
        # All-cached: 1000 * 0.003/1000 * 0.10 = 0.0003
        assert cost == pytest.approx(0.0003)


class TestPricingTableIsPubliclyAccessible:
    def test_pricing_dict_is_importable(self) -> None:
        # The story explicitly asks for "dynamic pricing table" — the table
        # itself must be a public attribute callers can introspect.
        assert isinstance(PRICING, dict)
        assert "gpt-4o" in PRICING
        assert "claude-3-5-sonnet" in PRICING


class TestEnvOverride:
    """LAY-3327 AC: pricing 'can be updated without code changes'.

    Setting ``LAYERLENS_PRICING_TABLE`` to a JSON file path applies overrides
    that take precedence over both the bundled defaults and any caller-supplied
    ``pricing_table``.
    """

    @pytest.fixture(autouse=True)
    def _isolate_env(self, monkeypatch: pytest.MonkeyPatch):
        # The override loader caches; reset before and after each test so the
        # tests are independent.
        monkeypatch.delenv(PRICING_OVERRIDE_ENV, raising=False)
        reset_pricing_cache()
        yield
        reset_pricing_cache()

    def test_env_override_changes_pricing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        override_path = tmp_path / "pricing.json"
        override_path.write_text(json.dumps({"gpt-4o": {"input": 0.999, "output": 0.999}}))
        monkeypatch.setenv(PRICING_OVERRIDE_ENV, str(override_path))
        reset_pricing_cache()

        cost = calculate_cost("gpt-4o", _usage(prompt=1000, completion=1000))
        # 1000 * 0.999/1000 + 1000 * 0.999/1000 = 1.998
        assert cost == pytest.approx(1.998)

    def test_env_override_wins_over_caller_supplied_table(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        override_path = tmp_path / "pricing.json"
        override_path.write_text(json.dumps({"gpt-4o": {"input": 0.5, "output": 0.5}}))
        monkeypatch.setenv(PRICING_OVERRIDE_ENV, str(override_path))
        reset_pricing_cache()

        caller_table = {"gpt-4o": {"input": 100.0, "output": 100.0}}
        cost = calculate_cost("gpt-4o", _usage(prompt=1000, completion=0), pricing_table=caller_table)
        # env value (0.5) used, not caller table (100.0).
        # 1000 * 0.5/1000 = 0.5
        assert cost == pytest.approx(0.5)

    def test_env_override_supports_fuzzy_matching(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Override of ``gpt-4o`` should also apply to dated variants via the
        # same fuzzy resolution.
        override_path = tmp_path / "pricing.json"
        override_path.write_text(json.dumps({"gpt-4o": {"input": 0.001, "output": 0.001}}))
        monkeypatch.setenv(PRICING_OVERRIDE_ENV, str(override_path))
        reset_pricing_cache()

        cost = calculate_cost("gpt-4o-2024-08-06", _usage(prompt=1000, completion=0))
        assert cost == pytest.approx(0.001)

    def test_missing_env_var_uses_defaults(self) -> None:
        # When LAYERLENS_PRICING_TABLE isn't set, defaults work normally.
        assert PRICING_OVERRIDE_ENV not in os.environ
        cost = calculate_cost("gpt-4o", _usage(prompt=1000, completion=0))
        assert cost is not None
        assert cost > 0

    def test_unreadable_override_file_falls_back_gracefully(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Pointing at a missing file must not crash the request path.
        monkeypatch.setenv(PRICING_OVERRIDE_ENV, str(tmp_path / "does-not-exist.json"))
        reset_pricing_cache()
        cost = calculate_cost("gpt-4o", _usage())
        # Falls back to defaults.
        assert cost is not None

    def test_malformed_json_override_falls_back_gracefully(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("not valid json at all {{{")
        monkeypatch.setenv(PRICING_OVERRIDE_ENV, str(bad))
        reset_pricing_cache()
        cost = calculate_cost("gpt-4o", _usage())
        assert cost is not None

    def test_env_override_adds_new_model_not_in_defaults(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # An ops team can price a new model that ships before a library release.
        override_path = tmp_path / "pricing.json"
        override_path.write_text(json.dumps({"my-internal-llm-v2": {"input": 0.01, "output": 0.02}}))
        monkeypatch.setenv(PRICING_OVERRIDE_ENV, str(override_path))
        reset_pricing_cache()

        cost = calculate_cost("my-internal-llm-v2", _usage(prompt=1000, completion=500))
        # 1000 * 0.01/1000 + 500 * 0.02/1000 = 0.01 + 0.01 = 0.02
        assert cost == pytest.approx(0.02)


class TestPricingTableClass:
    """LAY-3330 Claude Code Prompt requires a ``PricingTable`` class with:

    * default rates covering current OpenAI models
    * caller-provided overrides via constructor / from_dict / from_json_file
    * calculate_cost(model, input_tokens, output_tokens) -> CostRecord
    * fuzzy model matching for dated variants
    """

    def test_default_constructor_covers_openai_models(self):
        table = PricingTable()
        for m in ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo", "o1", "o1-mini", "o3", "o3-mini"]:
            assert table.has_model(m), f"PricingTable() missing default: {m}"

    def test_calculate_cost_returns_cost_record(self):
        table = PricingTable()
        record = table.calculate_cost("gpt-4o", input_tokens=1000, output_tokens=500)
        assert isinstance(record, CostRecord)
        assert record.model == "gpt-4o"
        assert record.input_tokens == 1000
        assert record.output_tokens == 500
        # gpt-4o: $0.0025/1k input + $0.01/1k output = $0.0025 + $0.005 = $0.0075
        assert record.cost_usd == pytest.approx(0.0075)

    def test_unknown_model_returns_record_with_none_cost(self):
        table = PricingTable()
        record = table.calculate_cost("totally-fake-model", input_tokens=100, output_tokens=50)
        assert record.cost_usd is None
        # And the input/output token counts are still surfaced so the caller
        # can decide how to log the unknown-model event.
        assert record.input_tokens == 100
        assert record.output_tokens == 50

    def test_fuzzy_match_on_class_method(self):
        table = PricingTable()
        dated = table.calculate_cost("gpt-4o-2024-08-06", input_tokens=1000, output_tokens=0)
        base = table.calculate_cost("gpt-4o", input_tokens=1000, output_tokens=0)
        assert dated.cost_usd == base.cost_usd

    def test_from_dict_overrides_defaults_entirely(self):
        custom = {"my-private-model": {"input": 1.0, "output": 2.0}}
        table = PricingTable.from_dict(custom)
        # Bundled defaults are NOT present in a from_dict table.
        assert table.has_model("my-private-model")
        assert not table.has_model("gpt-4o")

    def test_from_json_file(self, tmp_path):
        path = tmp_path / "rates.json"
        path.write_text(json.dumps({"team-llm": {"input": 0.005, "output": 0.01}}))
        table = PricingTable.from_json_file(str(path))
        record = table.calculate_cost("team-llm", input_tokens=2000, output_tokens=1000)
        # 2000 * 0.005/1000 + 1000 * 0.01/1000 = 0.01 + 0.01 = 0.02
        assert record.cost_usd == pytest.approx(0.02)

    def test_from_json_file_rejects_non_object_root(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps([1, 2, 3]))
        with pytest.raises(ValueError, match="must be an object"):
            PricingTable.from_json_file(str(path))

    def test_models_lists_keys(self):
        table = PricingTable.from_dict({"a": {"input": 0.1, "output": 0.2}, "b": {"input": 0.3, "output": 0.4}})
        models = table.models()
        assert sorted(models) == ["a", "b"]

    def test_cached_tokens_propagated_to_record(self):
        table = PricingTable()
        record = table.calculate_cost("gpt-4o", input_tokens=1000, output_tokens=0, cached_tokens=200)
        assert record.cached_tokens == 200
        # 800 non-cached at $0.0025/1k + 200 cached at 50% off = 200 * 0.00125/1k
        expected = (800 * 0.0025 / 1000) + (200 * (0.0025 * 0.5) / 1000)
        assert record.cost_usd == pytest.approx(expected)
