"""
AgentSpecParser -- Agent Spec Markdown Parser
===============================================

Parses an agent spec file (e.g. ``agent_spec.md``) into a structured
``SoulSpec`` object.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SoulSpec(BaseModel):
    """Structured representation of an agent spec file."""

    agent_name: str = Field(default="Unknown Agent")
    purpose: str = Field(default="")
    persona: str = Field(default="")
    ethical_constraints: list[str] = Field(default_factory=list)
    tool_boundaries: list[str] = Field(default_factory=list)
    extra_sections: dict[str, str] = Field(default_factory=dict)
    raw_content: str = Field(default="")
    source_path: str = Field(default="")

    def constraint_count(self) -> int:
        return len(self.ethical_constraints) + len(self.tool_boundaries)

    def summary(self) -> str:
        return (f"{self.agent_name}: {len(self.ethical_constraints)} ethical constraints, "
                f"{len(self.tool_boundaries)} tool boundaries")

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name, "purpose": self.purpose,
            "persona": self.persona, "ethical_constraints": self.ethical_constraints,
            "tool_boundaries": self.tool_boundaries,
        }


_SECTION_ALIASES: dict[str, str] = {
    "purpose": "purpose", "mission": "purpose", "objective": "purpose", "goal": "purpose",
    "persona": "persona", "personality": "persona", "character": "persona",
    "tone": "persona", "voice": "persona", "style": "persona",
    "ethical constraints": "ethical_constraints", "ethics": "ethical_constraints",
    "constraints": "ethical_constraints", "rules": "ethical_constraints",
    "boundaries": "ethical_constraints", "safety": "ethical_constraints",
    "guardrails": "ethical_constraints", "restrictions": "ethical_constraints",
    "tool boundaries": "tool_boundaries", "tools": "tool_boundaries",
    "capabilities": "tool_boundaries", "tool access": "tool_boundaries",
    "tool restrictions": "tool_boundaries", "tool permissions": "tool_boundaries",
    "allowed tools": "tool_boundaries",
}


class SoulFileParser:
    """Parses agent spec markdown files into structured SoulSpec objects."""

    def parse_file(self, path: str) -> SoulSpec:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Agent spec file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        if not content.strip():
            raise ValueError(f"Agent spec file is empty: {path}")
        spec = self.parse_string(content)
        spec.source_path = path
        return spec

    def parse_string(self, content: str) -> SoulSpec:
        spec = SoulSpec(raw_content=content)
        spec.agent_name = self._extract_agent_name(content)
        sections = self._split_sections(content)
        for heading, body in sections.items():
            canonical = self._normalize_heading(heading)
            if canonical == "purpose":
                spec.purpose = self._extract_paragraph(body)
            elif canonical == "persona":
                spec.persona = self._extract_paragraph(body)
            elif canonical == "ethical_constraints":
                spec.ethical_constraints = self._extract_list_items(body)
            elif canonical == "tool_boundaries":
                spec.tool_boundaries = self._extract_list_items(body)
            else:
                spec.extra_sections[heading] = body.strip()
        if not spec.ethical_constraints and not spec.tool_boundaries:
            all_items = self._extract_list_items(content)
            if all_items:
                spec.ethical_constraints = all_items
        return spec

    def _extract_agent_name(self, content: str) -> str:
        match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if match:
            return match.group(1).strip()
        for line in content.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped[:80]
        return "Unknown Agent"

    def _split_sections(self, content: str) -> dict[str, str]:
        sections: dict[str, str] = {}
        pattern = re.compile(r'^##\s+(.+)$', re.MULTILINE)
        matches = list(pattern.finditer(content))
        if not matches:
            return sections
        for i, match in enumerate(matches):
            heading = match.group(1).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            sections[heading] = content[start:end]
        return sections

    def _normalize_heading(self, heading: str) -> str:
        clean = re.sub(r'[*_`]', '', heading).strip().lower()
        if clean in _SECTION_ALIASES:
            return _SECTION_ALIASES[clean]
        for alias, canonical in _SECTION_ALIASES.items():
            if alias in clean or clean in alias:
                return canonical
        return clean

    def _extract_paragraph(self, body: str) -> str:
        lines: list[str] = []
        for line in body.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            if re.match(r'^[-*]\s+', stripped) or re.match(r'^\d+\.\s+', stripped):
                continue
            lines.append(stripped)
        return " ".join(lines)

    def _extract_list_items(self, body: str) -> list[str]:
        items: list[str] = []
        current_item: str | None = None
        for line in body.splitlines():
            stripped = line.strip()
            bullet_match = re.match(r'^[-*]\s+(.+)$', stripped)
            if bullet_match:
                if current_item is not None:
                    items.append(current_item.strip())
                current_item = bullet_match.group(1)
                continue
            num_match = re.match(r'^\d+\.\s+(.+)$', stripped)
            if num_match:
                if current_item is not None:
                    items.append(current_item.strip())
                current_item = num_match.group(1)
                continue
            if current_item is not None and stripped and line.startswith(('  ', '\t')):
                current_item += " " + stripped
                continue
            if current_item is not None:
                items.append(current_item.strip())
                current_item = None
        if current_item is not None:
            items.append(current_item.strip())
        return items


DEFAULT_SOUL_SPEC_MD = """# Demo Assistant

## Purpose
A helpful coding assistant that answers programming questions and reviews code.

## Persona
Professional, concise, and encouraging. Uses technical language appropriate
to the user's level. Prefers examples over abstract explanations.

## Ethical Constraints
- Must not generate malicious code (malware, exploits, credential stealers)
- Must not assist with hacking, unauthorized access, or social engineering
- Must not generate discriminatory, hateful, or explicit content
- Must decline requests for personal data extraction or privacy violations
- Must not impersonate real individuals or organizations
- Must acknowledge uncertainty rather than fabricating answers

## Tool Boundaries
- Can only use: code_search, file_read, run_tests, lint_check
- Cannot access: internet, databases, external APIs, file_write
- Cannot execute arbitrary shell commands
- Rate limited to 20 tool calls per session
- Cannot read files outside the project directory
"""


def get_default_soul_spec() -> SoulSpec:
    """Parse and return the built-in default agent spec for demos."""
    parser = SoulFileParser()
    return parser.parse_string(DEFAULT_SOUL_SPEC_MD)
