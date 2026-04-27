"""Vendored snapshot of ``stratix.core.events.l2_code``.

Source: ``A:/github/layerlens/ateam/stratix/core/events/l2_code.py``
Source SHA: 7359c0e38d74e02aa1b27c34daef7a958abbd002

Compatibility shims applied for Python 3.9 + Pydantic 2:
- PEP-604 union syntax (``X | None``) on Pydantic field annotations
  rewritten as ``Optional[X]`` (Pydantic 2 evaluates field type hints
  via ``typing.get_type_hints``, which fails on Python 3.9 even with
  ``from __future__ import annotations``).

Updates require re-vendoring — see ``__init__.py`` for the workflow.
"""

# STRATIX Layer 2 Events - Agent Logic Code
#
# {
#     "event_type": "agent.code",
#     "layer": "L2",
#     "code": {
#         "repo": "uri",
#         "commit": "git_sha",
#         "artifact_hash": "sha256",
#         "config_hash": "sha256"
#     }
# }

from __future__ import annotations

from typing import Any, Optional

from pydantic import Field, BaseModel, field_validator


class CodeInfo(BaseModel):
    """Code artifact information for L2 events."""

    repo: str = Field(description="Repository URI")
    commit: str = Field(description="Git commit SHA")
    artifact_hash: str = Field(description="SHA-256 hash of the code artifact")
    config_hash: str = Field(description="SHA-256 hash of the configuration")
    branch: Optional[str] = Field(default=None, description="Git branch name")
    tag: Optional[str] = Field(default=None, description="Git tag if applicable")
    build_info: Optional[dict[str, Any]] = Field(
        default=None, description="Additional build information"
    )

    @field_validator("artifact_hash", "config_hash")
    @classmethod
    def validate_hash(cls, v: str) -> str:
        """Validate hash format."""
        if not v.startswith("sha256:"):
            raise ValueError("Hash must start with 'sha256:'")
        hex_part = v[7:]
        if len(hex_part) != 64:
            raise ValueError("Hash must be sha256: followed by 64 hex characters")
        return v


class AgentCodeEvent(BaseModel):
    """Layer 2 Event: Agent Code.

    Represents the agent's code artifact for attestation.

    NORMATIVE: Must be emitted at trial start (and on hot reload).

    For framework adapters that wrap user-defined node callables (e.g.,
    LangGraph nodes), this event is also emitted on each node execution
    with the callable's identity-derived hash, enabling per-node
    artifact attestation and replay correlation.
    """

    event_type: str = Field(default="agent.code", description="Event type identifier")
    layer: str = Field(default="L2", description="Layer identifier")
    code: CodeInfo = Field(description="Code artifact information")

    @classmethod
    def create(
        cls,
        repo: str,
        commit: str,
        artifact_hash: str,
        config_hash: str,
        branch: Optional[str] = None,
        tag: Optional[str] = None,
        build_info: Optional[dict[str, Any]] = None,
    ) -> AgentCodeEvent:
        """Create an agent code event.

        Args:
            repo: Repository URI
            commit: Git commit SHA
            artifact_hash: SHA-256 hash of the code artifact (must start with ``sha256:``)
            config_hash: SHA-256 hash of the configuration (must start with ``sha256:``)
            branch: Optional git branch name
            tag: Optional git tag
            build_info: Optional build information

        Returns:
            AgentCodeEvent instance
        """
        return cls(
            code=CodeInfo(
                repo=repo,
                commit=commit,
                artifact_hash=artifact_hash,
                config_hash=config_hash,
                branch=branch,
                tag=tag,
                build_info=build_info,
            )
        )


__all__ = [
    "CodeInfo",
    "AgentCodeEvent",
]
