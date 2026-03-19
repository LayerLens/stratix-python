"""ID generation for simulated traces and spans.

Generates deterministic or random IDs for:
- Trace IDs (32 hex chars / 16 bytes)
- Span IDs (16 hex chars / 8 bytes)
- Salesforce record IDs (15/18 char alphanumeric)
- W3C traceparent headers
- Response IDs (provider-specific formats)
"""

from __future__ import annotations

import random
import string
import uuid


class IDGenerator:
    """Deterministic or random ID generator.

    When seed is provided, generates reproducible IDs from PRNG.
    When seed is None, generates random IDs.
    """

    def __init__(self, seed: int | None = None):
        self._seed = seed
        self._rng = random.Random(seed)

    def trace_id(self) -> str:
        """Generate a 32-char hex trace ID (16 bytes)."""
        return self._hex_bytes(16)

    def span_id(self) -> str:
        """Generate a 16-char hex span ID (8 bytes)."""
        return self._hex_bytes(8)

    def _hex_bytes(self, n: int) -> str:
        """Generate n random bytes as hex string."""
        return "".join(f"{self._rng.randint(0, 255):02x}" for _ in range(n))

    def traceparent(self, trace_id: str, span_id: str, sampled: bool = True) -> str:
        """Generate W3C traceparent header.

        Format: 00-{trace_id}-{span_id}-{flags}
        """
        flags = "01" if sampled else "00"
        return f"00-{trace_id}-{span_id}-{flags}"

    def salesforce_id(self) -> str:
        """Generate a Salesforce-style 18-char record ID."""
        chars = string.ascii_uppercase + string.digits
        return "".join(self._rng.choice(chars) for _ in range(18))

    def response_id_openai(self) -> str:
        """Generate OpenAI-style response ID (chatcmpl-...)."""
        suffix = "".join(
            self._rng.choice(string.ascii_letters + string.digits) for _ in range(29)
        )
        return f"chatcmpl-{suffix}"

    def response_id_anthropic(self) -> str:
        """Generate Anthropic-style response ID (msg_...)."""
        suffix = "".join(
            self._rng.choice(string.ascii_letters + string.digits) for _ in range(24)
        )
        return f"msg_{suffix}"

    def response_id_vertex(self) -> str:
        """Generate Vertex AI-style response ID."""
        return str(uuid.UUID(int=self._rng.getrandbits(128), version=4))

    def response_id_bedrock(self) -> str:
        """Generate Bedrock-style request ID."""
        return str(uuid.UUID(int=self._rng.getrandbits(128), version=4))

    def system_fingerprint(self) -> str:
        """Generate OpenAI-style system fingerprint."""
        suffix = "".join(
            self._rng.choice(string.ascii_lowercase + string.digits) for _ in range(10)
        )
        return f"fp_{suffix}"

    def tool_call_id(self) -> str:
        """Generate a tool call ID (call_...)."""
        suffix = "".join(
            self._rng.choice(string.ascii_letters + string.digits) for _ in range(24)
        )
        return f"call_{suffix}"

    def session_id(self) -> str:
        """Generate a session ID for multi-turn conversations."""
        return str(uuid.UUID(int=self._rng.getrandbits(128), version=4))

    def run_id(self) -> str:
        """Generate a simulator run ID (run_...)."""
        suffix = "".join(
            self._rng.choice(string.ascii_lowercase + string.digits) for _ in range(8)
        )
        return f"run_{suffix}"

    def langfuse_trace_id(self) -> str:
        """Generate Langfuse-compatible trace ID (UUID)."""
        return str(uuid.UUID(int=self._rng.getrandbits(128), version=4))

    def langfuse_observation_id(self) -> str:
        """Generate Langfuse observation ID."""
        return str(uuid.UUID(int=self._rng.getrandbits(128), version=4))
