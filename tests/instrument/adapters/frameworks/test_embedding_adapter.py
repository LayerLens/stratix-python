"""Truncation-policy tests for the embedding + vector_store adapters.

The full embedding adapter test suite lives elsewhere (it requires
optional client SDKs). These tests verify only the cross-pollination
audit §2.4 wiring: each EmbeddingAdapter / VectorStoreAdapter
constructor declares :data:`DEFAULT_POLICY`, and the BaseAdapter
emit path correctly clips oversized payloads.
"""

from __future__ import annotations

from typing import Any, Dict, List

from layerlens.instrument.adapters._base import (
    DEFAULT_POLICY,
    AdapterStatus,
    CaptureConfig,
)
from layerlens.instrument.adapters.frameworks.embedding.embedding_adapter import (
    EmbeddingAdapter,
)
from layerlens.instrument.adapters.frameworks.embedding.vector_store_adapter import (
    VectorStoreAdapter,
)


class _RecordingStratix:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def emit(self, *args: Any, **kwargs: Any) -> None:
        if len(args) == 2 and isinstance(args[0], str):
            self.events.append({"event_type": args[0], "payload": args[1]})


# ---------------------------------------------------------------------------
# EmbeddingAdapter
# ---------------------------------------------------------------------------


def test_embedding_adapter_truncation_policy_is_default() -> None:
    adapter = EmbeddingAdapter()
    assert adapter._truncation_policy is DEFAULT_POLICY


def test_embedding_adapter_truncates_long_input_text() -> None:
    """Embedding inputs are user prompts/text — must be capped."""
    stratix = _RecordingStratix()
    adapter = EmbeddingAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()
    assert adapter.status == AdapterStatus.HEALTHY

    adapter.emit_dict_event(
        "embedding.create",
        {"model": "text-embedding-3-small", "input": "x" * 10000},
    )

    payload = stratix.events[-1]["payload"]
    assert isinstance(payload["input"], str)
    assert payload["input"].startswith("x" * 4096)
    audit = payload.get("_truncated_fields", [])
    assert any("input:chars-10000->4096" in entry for entry in audit), audit


def test_embedding_adapter_short_payload_not_audited() -> None:
    stratix = _RecordingStratix()
    adapter = EmbeddingAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    adapter.emit_dict_event(
        "embedding.create", {"model": "tiny", "input": "hello"}
    )
    payload = stratix.events[-1]["payload"]
    assert "_truncated_fields" not in payload


# ---------------------------------------------------------------------------
# VectorStoreAdapter
# ---------------------------------------------------------------------------


def test_vector_store_adapter_truncation_policy_is_default() -> None:
    adapter = VectorStoreAdapter()
    assert adapter._truncation_policy is DEFAULT_POLICY


def test_vector_store_adapter_truncates_oversize_query() -> None:
    """Retrieval queries with oversized text payloads are truncated."""
    stratix = _RecordingStratix()
    adapter = VectorStoreAdapter(stratix=stratix, capture_config=CaptureConfig.full())
    adapter.connect()

    adapter.emit_dict_event(
        "retrieval.query",
        {"top_k": 5, "text": "q" * 10000},
    )

    payload = stratix.events[-1]["payload"]
    assert isinstance(payload["text"], str)
    assert payload["text"].startswith("q" * 4096)
    audit = payload.get("_truncated_fields", [])
    assert any("text:chars-10000->4096" in entry for entry in audit)
