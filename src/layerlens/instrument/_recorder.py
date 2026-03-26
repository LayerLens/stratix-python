from __future__ import annotations

import base64
import logging
import weakref
import threading
from typing import Any, Dict, List, Tuple, Optional

from layerlens.attestation import HashChain

from ._types import SpanData
from ._upload import upload_trace, async_upload_trace

log: logging.Logger = logging.getLogger(__name__)

# Per-client cache for auto-resolved signing keys.
# Uses weakref to the client so entries are evicted when the client is GC'd,
# preventing stale keys from being served to a new client at the same address.
_signing_key_cache: Dict[int, Tuple[Any, Optional[Tuple[str, bytes]]]] = {}  # (weakref.ref | callable, value)
_cache_lock = threading.Lock()

_SENTINEL = object()  # distinguishes "not passed" from "passed as None"
_NOT_RESOLVED = object()  # cache miss marker


def _cache_get(client: Any) -> Any:
    """Look up cached signing key for a client. Returns _NOT_RESOLVED on miss."""
    entry = _signing_key_cache.get(id(client), None)
    if entry is None:
        return _NOT_RESOLVED
    ref, value = entry
    # If the weakref is dead, the original client was GC'd and a new object
    # now occupies the same id(). Evict the stale entry.
    if ref() is None:
        del _signing_key_cache[id(client)]
        return _NOT_RESOLVED
    return value


def _cache_put(client: Any, value: Optional[Tuple[str, bytes]]) -> None:
    """Store signing key in cache, keyed by client identity."""
    try:
        ref = weakref.ref(client)
    except TypeError:
        # Client doesn't support weakrefs (e.g. some Mock objects).
        # Fall back to caching without liveness check.
        ref = lambda: client  # type: ignore[assignment]
    _signing_key_cache[id(client)] = (ref, value)


def _resolve_signing_key(client: Any) -> Optional[Tuple[str, bytes]]:
    """Fetch the org's active signing key, or auto-create one if none exists.

    Returns (key_id, secret_bytes) or None. Result is cached per client
    instance so we only hit the API once. If the org has no signing key,
    the SDK will attempt to create one automatically.
    """
    with _cache_lock:
        cached = _cache_get(client)
        if cached is not _NOT_RESOLVED:
            return cached  # type: ignore[no-any-return]

    # Fetch outside the lock to avoid holding it during I/O.
    result: Optional[Tuple[str, bytes]] = None
    try:
        if hasattr(client, "signing_keys"):
            key_data = client.signing_keys.get_active()
            if not _is_valid_key_data(key_data):
                # No active key — auto-create one for the org.
                log.info("No active signing key found, auto-creating one for attestation")
                key_data = client.signing_keys.create()
            if _is_valid_key_data(key_data):
                secret_bytes = base64.b64decode(key_data["secret"])
                result = (key_data["key_id"], secret_bytes)
                log.info("Attestation signing key resolved: %s", key_data["key_id"])
            else:
                log.info("Could not resolve or create signing key — traces will be unsigned")
    except Exception:
        log.warning("Failed to resolve signing key, traces will be unsigned", exc_info=True)

    with _cache_lock:
        # Another thread may have populated while we were fetching — first writer wins.
        existing = _cache_get(client)
        if existing is not _NOT_RESOLVED:
            return existing  # type: ignore[no-any-return]
        _cache_put(client, result)

    return result


def _is_valid_key_data(data: Any) -> bool:
    """Check that key data is a dict with both 'key_id' and 'secret'."""
    return isinstance(data, dict) and "secret" in data and "key_id" in data


def clear_signing_key_cache(client: Any = None) -> None:
    """Clear cached signing keys. Call after key rotation.

    Pass a specific client to clear only its cache, or None to clear all.
    """
    with _cache_lock:
        if client is None:
            _signing_key_cache.clear()
        else:
            _signing_key_cache.pop(id(client), None)


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
    def __init__(
        self,
        client: Any,
        signing_service: Any = _SENTINEL,
    ) -> None:
        self._client = client

        if signing_service is _SENTINEL:
            # Auto-resolve: fetch the org's active signing key
            self._signing_key = _resolve_signing_key(client)
        elif signing_service is None:
            # Explicit None: no signing
            self._signing_key = None
        else:
            # Explicit (key_id, secret) tuple
            self._signing_key = signing_service

        self.root: Optional[SpanData] = None

    def _build_attestation(self) -> Dict[str, Any]:
        """Build a hash chain from the span tree and return attestation data."""
        if self.root is None:
            return {}

        if self._signing_key is not None:
            key_id, secret = self._signing_key
            chain = HashChain(signing_key_id=key_id, signing_secret=secret)
        else:
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
            log.warning("Failed to build attestation chain", exc_info=True)
            attestation = {}
        upload_trace(self._client, trace_data, attestation)

    async def async_flush(self) -> None:
        if self.root is None:
            return
        trace_data = self.root.to_dict()
        try:
            attestation = self._build_attestation()
        except Exception:
            log.warning("Failed to build attestation chain", exc_info=True)
            attestation = {}
        await async_upload_trace(self._client, trace_data, attestation)
