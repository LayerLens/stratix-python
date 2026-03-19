"""
STRATIX Layer 2 Events - Agent Logic Code

From Step 1 specification:
{
    "event_type": "agent.code",
    "layer": "L2",
    "code": {
        "repo": "uri",
        "commit": "git_sha",
        "artifact_hash": "sha256",
        "config_hash": "sha256"
    }
}
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class CodeInfo(BaseModel):
    """Code artifact information for L2 events."""
    repo: str = Field(
        description="Repository URI"
    )
    commit: str = Field(
        description="Git commit SHA"
    )
    artifact_hash: str = Field(
        description="SHA-256 hash of the code artifact"
    )
    config_hash: str = Field(
        description="SHA-256 hash of the configuration"
    )
    branch: str | None = Field(
        default=None,
        description="Git branch name"
    )
    tag: str | None = Field(
        default=None,
        description="Git tag if applicable"
    )
    build_info: dict[str, Any] | None = Field(
        default=None,
        description="Additional build information"
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
    """
    Layer 2 Event: Agent Code

    Represents the agent's code artifact for attestation.

    NORMATIVE: Must be emitted at trial start (and on hot reload).
    """
    event_type: str = Field(
        default="agent.code",
        description="Event type identifier"
    )
    layer: str = Field(
        default="L2",
        description="Layer identifier"
    )
    code: CodeInfo = Field(
        description="Code artifact information"
    )

    @classmethod
    def create(
        cls,
        repo: str,
        commit: str,
        artifact_hash: str,
        config_hash: str,
        branch: str | None = None,
        tag: str | None = None,
        build_info: dict[str, Any] | None = None,
    ) -> AgentCodeEvent:
        """
        Create an agent code event.

        Args:
            repo: Repository URI
            commit: Git commit SHA
            artifact_hash: SHA-256 hash of the code artifact
            config_hash: SHA-256 hash of the configuration
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
