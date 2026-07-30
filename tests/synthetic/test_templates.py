from __future__ import annotations

from layerlens.synthetic.templates import (
    TEMPLATE_LIBRARY,
    TraceCategory,
    TraceTemplate,
    TemplateParameter,
)


class TestTemplateLibrary:
    def test_library_populated(self):
        assert len(TEMPLATE_LIBRARY) >= 4
        ids = set(TEMPLATE_LIBRARY)
        for expected in {
            "llm.chat.basic",
            "agent.tool_calling",
            "rag.retrieval",
            "multi_agent.handoff",
        }:
            assert expected in ids

    def test_every_template_has_defaults_for_every_parameter(self):
        for t in TEMPLATE_LIBRARY.values():
            for p in t.parameters:
                if not p.required:
                    assert p.name in t.defaults, f"{t.id}:{p.name} missing default"

    def test_categories_match_enum(self):
        for t in TEMPLATE_LIBRARY.values():
            assert isinstance(t.category, TraceCategory)


class TestTemplateModel:
    def test_parameter_choices_optional(self):
        p = TemplateParameter(name="x", type="string")
        assert p.choices is None
        assert p.required is False

    def test_template_bounds_sensible(self):
        t = TraceTemplate(id="x", category=TraceCategory.LLM, title="t")
        assert t.min_traces >= 1
        assert t.max_traces >= t.min_traces
