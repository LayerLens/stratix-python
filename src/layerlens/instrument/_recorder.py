from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from layerlens.attestation import HashChain

from ._types import SpanData
from ._upload import upload_trace, async_upload_trace

log: logging.Logger = logging.getLogger(__name__)


def _collect_spans(span: SpanData) -> List[Dict[str, Any]]:
    """Walk the span tree depth-first and return a flat list of span dicts.

    Uses SpanData.to_dict() to capture every field — structure, inputs,
    outputs, metadata, and errors. Children are excluded because we
    flatten the tree ourselves; any future SpanData fields are automatically
    included in the hash.
    """
    result: List[Dict[str, Any]] = []
    span_dict = span.to_dict()
    span_dict.pop("children")
    result.append(span_dict)
    for child in span.children:
        result.extend(_collect_spans(child))
    return result


class TraceRecorder:
    def __init__(self, client: Any) -> None:
        self._client = client
        self.root: Optional[SpanData] = None

    def _build_attestation(self) -> Dict[str, Any]:
        """Build a hash chain from the span tree and return attestation data."""
        if self.root is None:
            return {}
        chain = HashChain()
        spans = _collect_spans(self.root)
        for span_dict in spans:
            chain.add_event(span_dict)
        trial = chain.finalize()
        return {
            "chain": chain.to_dict(),
            "root_hash": trial.hash,
            "schema_version": "1.0",
        }

    def flush(self) -> None:
        if self.root is None:
            return
        trace_data = self.root.to_dict()
        try:
            attestation = self._build_attestation()
        except Exception:
            log.debug("Failed to build attestation chain", exc_info=True)
            attestation = {}
        upload_trace(self._client, trace_data, attestation)

    async def async_flush(self) -> None:
        if self.root is None:
            return
        trace_data = self.root.to_dict()
        try:
            attestation = self._build_attestation()
        except Exception:
            log.debug("Failed to build attestation chain", exc_info=True)
            attestation = {}
        await async_upload_trace(self._client, trace_data, attestation)
