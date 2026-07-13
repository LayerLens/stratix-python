"""Credless runtime smoke tests for the framework adapter samples (G3 + G4).

Every ``samples/adapters/frameworks/*.py`` sample is a *genuinely runnable*
program: it import-guards its framework library and cred-guards the LLM/service
credentials it needs, exactly like the eleven pre-existing samples. This suite
runs each sample's ``main()`` in a subprocess whose environment is stripped of
every credential and asserts it exits **cleanly** — i.e. it reaches its honest
import/cred guard (or, for the fully offline samples, actually captures real
instrumentation events) and never raises.

This is deliberately runtime (subprocess) rather than structural: the
structural + adapter-API-existence checks already live in ``test_samples.py``
and pick up new files automatically. This suite proves the samples *run*.

For the offline-capable ``vector_store`` sample the assertion is stronger: with
``chromadb`` installed it must capture at least one real ``retrieval.query``
event (no creds, no network) — proving the sample exercises the adapter for
real rather than only reaching a guard.
"""

from __future__ import annotations

import os
import sys
import subprocess

import pytest

_SAMPLE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "samples",
    "adapters",
    "frameworks",
)

# G3 (8 frameworks that previously lacked a sample) + G4 team samples. Each maps
# the framework/scenario name to its sample filename under _SAMPLE_DIR.
SAMPLE_FILES = {
    "agno": "agno_agent.py",
    "bedrock_agents": "bedrock_agents_invoke.py",
    "embedding": "embedding_openai.py",
    "google_adk": "google_adk_agent.py",
    "ms_agent_framework": "ms_agent_framework_chat.py",
    "smolagents": "smolagents_tool_agent.py",
    "strands": "strands_agent.py",
    "vector_store": "vector_store_query.py",
    # G4 multi-agent team samples (beyond the existing autogen/crewai).
    "google_adk_team": "google_adk_team.py",
    "semantic_kernel_team": "semantic_kernel_team.py",
}

# Every credential axis a sample might gate on — cleared so the credless run
# deterministically reaches an import guard, a cred guard, or (offline) real
# event capture, and never a live network call.
_CRED_VARS = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "LAYERLENS_STRATIX_API_KEY",
    "LAYERLENS_ATLAS_API_KEY",
    "LAYERLENS_STRATIX_BASE_URL",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_PROFILE",
    "BEDROCK_AGENT_ID",
    "BEDROCK_AGENT_ALIAS_ID",
)


def _run_credless(filename: str) -> subprocess.CompletedProcess[str]:
    """Run a sample's ``main()`` (``python <sample>``) with all creds removed."""
    path = os.path.join(_SAMPLE_DIR, filename)
    env = {k: v for k, v in os.environ.items() if k not in _CRED_VARS}
    # Force the AWS SDK to treat the environment as unconfigured too.
    env["AWS_EC2_METADATA_DISABLED"] = "true"
    return subprocess.run(
        [sys.executable, path],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )


@pytest.mark.parametrize("name,filename", sorted(SAMPLE_FILES.items()))
def test_sample_runs_cleanly_credless(name: str, filename: str) -> None:
    """Each sample exits 0 and never raises when run without credentials.

    It should reach its honest import/cred guard (or capture events offline),
    printing a human-readable line — not crash with a traceback.
    """
    path = os.path.join(_SAMPLE_DIR, filename)
    assert os.path.isfile(path), f"missing sample: {path}"

    result = _run_credless(filename)
    output = result.stdout + result.stderr
    assert result.returncode == 0, f"{filename} exited {result.returncode}:\n{output}"
    assert "Traceback (most recent call last)" not in output, f"{filename} raised:\n{output}"
    assert output.strip(), f"{filename} produced no output (no guard message / no events)"


def test_vector_store_captures_real_events_offline() -> None:
    """The offline ``vector_store`` sample captures a real ``retrieval.query``
    event with no creds and no network (Chroma ``EphemeralClient``)."""
    pytest.importorskip("chromadb")

    result = _run_credless(SAMPLE_FILES["vector_store"])
    output = result.stdout + result.stderr
    assert result.returncode == 0, f"vector_store exited {result.returncode}:\n{output}"
    assert "Traceback (most recent call last)" not in output, output

    # The shared helper prints '--- captured N events ---'; require N > 0 and a
    # real retrieval event in the captured stream.
    marker = "--- captured "
    assert marker in output, f"no capture summary in output:\n{output}"
    count = int(output.split(marker, 1)[1].split(" ", 1)[0])
    assert count > 0, f"expected captured events > 0, got {count}:\n{output}"
    assert "retrieval.query" in output, f"expected a retrieval.query event:\n{output}"
