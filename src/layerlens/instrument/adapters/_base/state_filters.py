"""Per-key allowlist / denylist / mask filters for adapter state payloads.

LangGraph's ``LangGraphStateAdapter`` (see
``src/layerlens/instrument/adapters/frameworks/langgraph/state.py`` on
``main`` — the mature reference implementation) supports include/exclude
key filters at the state-snapshot level so customers can scrub sensitive
state (api_keys, tokens, PII) WITHOUT modifying their agent code or
doing post-hoc redaction.

Lighter multi-agent adapters (agno, openai_agents, llama_index,
google_adk, strands, pydantic_ai, ms_agent_framework) emit dict-shaped
state into ``agent.input`` / ``agent.output`` / ``agent.state.change``
events. Without filtering, that state can carry credentials, PII, or
unbounded cardinality straight into telemetry sinks.

This module provides:

* :class:`StateFilter` — Pydantic-style config object capturing the
  three filter operations (exclude, mask, include-allowlist).
* :func:`filter_state` — pure function that applies a filter to a dict
  recursively.
* :data:`DEFAULT_PII_EXCLUDE_KEYS` — conservative default denylist that
  matches common PII / credential field names (case-insensitive
  substring match) so customers who forget to configure a filter still
  get sensible protection.
* :func:`default_state_filter` — factory for the default PII-aware
  filter installed by every framework adapter unless the customer
  overrides it.

The filter is intentionally cross-cutting: framework adapters expose a
``state_filter`` constructor parameter (defaulting to
``default_state_filter()``), keep it reachable via
``self._state_filter``, and pass dict-shaped payload fields through
:func:`filter_state` before they are emitted. Multi-tenancy is
preserved by applying the SAME default filter regardless of org — every
customer gets baseline PII protection out of the box.

Auditability: :func:`filter_state` returns a 2-tuple of
``(filtered_dict, filtered_keys)`` so callers can record the names of
any keys that were excluded or masked. Adapters surface this list as
``_filtered_keys`` metadata on the emitted event so customers can see
exactly what was clipped from the payload.

This module is **adapter-internal infrastructure**. It is NOT public
API for end users — there are no version guarantees on the helpers
exposed here, only on the BaseAdapter contract.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple, Iterable, Optional, FrozenSet
from dataclasses import field, dataclass

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------


REDACTED_PLACEHOLDER: str = "[REDACTED]"
"""String used to replace masked values.

Picked as a recognisable, non-PII string that is short enough not to
inflate payload size and obvious enough that downstream operators
immediately understand the value was clipped on purpose.
"""


DEFAULT_PII_EXCLUDE_KEYS: FrozenSet[str] = frozenset(
    {
        # Credentials
        "password",
        "passwd",
        "pwd",
        "api_key",
        "apikey",
        "api_secret",
        "secret",
        "secret_key",
        "access_token",
        "refresh_token",
        "auth_token",
        "bearer_token",
        "token",
        "session_token",
        "cookie",
        "cookies",
        "private_key",
        "client_secret",
        "service_account",
        # Personal identifiers
        "ssn",
        "social_security",
        "social_security_number",
        "tax_id",
        "national_id",
        "passport",
        "passport_number",
        "drivers_license",
        # Financial
        "credit_card",
        "credit_card_number",
        "card_number",
        "cvv",
        "cvc",
        "iban",
        "account_number",
        "routing_number",
        # Contact / location
        "email",
        "email_address",
        "phone",
        "phone_number",
        "address",
        "street_address",
        "home_address",
        "billing_address",
        "shipping_address",
        # Authn material
        "authorization",
        "x-api-key",
        "set-cookie",
    }
)
"""Default exclude-key denylist.

The check performed by :func:`filter_state` is **case-insensitive
substring** — so a key named ``"customer_email"`` matches the entry
``"email"`` and is filtered. This catches the long tail of vendor- or
team-specific field names (e.g. ``USER_API_KEY``,
``stripe_customer_email``, ``X-Api-Key``) without forcing the caller
to enumerate every variant.

The list is conservative on purpose: false positives (filtering a
field that was not actually PII) are recoverable by the customer
re-emitting telemetry with a custom :class:`StateFilter`. False
negatives (a credential leaking into a sink) are not.
"""


# ---------------------------------------------------------------------------
# StateFilter dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StateFilter:
    """Declarative filter applied to dict-shaped adapter state payloads.

    Three operations, applied in this order during :func:`filter_state`:

    1. **exclude_keys** — keys (case-insensitive substring match)
       removed from the output entirely.
    2. **mask_keys** — keys (case-insensitive substring match) whose
       values are replaced with :data:`REDACTED_PLACEHOLDER`. The key
       remains visible (so dashboards see the field exists), but the
       value is hidden.
    3. **include_keys** — if non-empty, the output is restricted to
       only these keys (case-insensitive equality match against the
       full key name; substring match would be too lossy when the
       caller explicitly named an allowlist).

    Order matters: ``exclude`` runs first (cheapest, fully removes
    keys), ``mask`` next (rewrites values), and ``include`` last so the
    allowlist can still narrow the masked-but-allowed surface.

    Parameters
    ----------
    include_keys:
        If non-empty, ONLY keys equal (case-insensitive) to one of
        these values are kept after exclude/mask. Default ``None``
        means "keep everything that survives exclude/mask".
    exclude_keys:
        Keys whose name *contains* any of these substrings (case-
        insensitive) are removed. Defaults to
        :data:`DEFAULT_PII_EXCLUDE_KEYS`.
    mask_keys:
        Keys whose name *contains* any of these substrings (case-
        insensitive) have their value replaced with
        :data:`REDACTED_PLACEHOLDER`. Default empty — opt-in.
    recursive:
        When ``True`` (default), nested dicts (and dicts inside lists)
        are also filtered. When ``False``, only the top-level dict is
        examined — useful when the caller knows nested structures are
        already safe.
    """

    include_keys: Optional[FrozenSet[str]] = None
    exclude_keys: FrozenSet[str] = field(default_factory=lambda: DEFAULT_PII_EXCLUDE_KEYS)
    mask_keys: FrozenSet[str] = field(default_factory=frozenset)
    recursive: bool = True

    def __post_init__(self) -> None:
        # Normalize all key collections to lowercase frozensets for
        # case-insensitive comparison. Frozen dataclass means we must
        # reach through ``object.__setattr__`` to mutate.
        if self.include_keys is not None:
            object.__setattr__(
                self,
                "include_keys",
                frozenset(k.lower() for k in self.include_keys),
            )
        object.__setattr__(
            self,
            "exclude_keys",
            frozenset(k.lower() for k in self.exclude_keys),
        )
        object.__setattr__(
            self,
            "mask_keys",
            frozenset(k.lower() for k in self.mask_keys),
        )

    # -- factory helpers --------------------------------------------------

    @classmethod
    def permissive(cls) -> "StateFilter":
        """Filter that does NOTHING — for tests / explicit opt-out."""
        return cls(include_keys=None, exclude_keys=frozenset(), mask_keys=frozenset())

    @classmethod
    def with_extra_excludes(cls, extra: Iterable[str]) -> "StateFilter":
        """Default PII filter PLUS the caller's additional excludes."""
        merged = frozenset(k.lower() for k in DEFAULT_PII_EXCLUDE_KEYS) | frozenset(k.lower() for k in extra)
        return cls(exclude_keys=merged)

    # -- public introspection --------------------------------------------

    def as_metadata(self) -> Dict[str, Any]:
        """Snapshot of this filter for inclusion in adapter / replay metadata."""
        meta: Dict[str, Any] = {
            "exclude_keys_count": len(self.exclude_keys),
            "mask_keys_count": len(self.mask_keys),
            "recursive": self.recursive,
        }
        if self.include_keys is not None:
            meta["include_keys_count"] = len(self.include_keys)
            # Allowlists are usually short and intentional — surface them
            # so customers can verify exactly what they configured.
            meta["include_keys"] = sorted(self.include_keys)
        return meta


# ---------------------------------------------------------------------------
# Default factory
# ---------------------------------------------------------------------------


def default_state_filter() -> StateFilter:
    """Return the conservative default filter installed by adapters.

    Excludes the built-in :data:`DEFAULT_PII_EXCLUDE_KEYS` denylist with
    no additional masks or allowlist. Customers who do nothing still
    get baseline PII protection on every emitted state payload.
    """
    return StateFilter()


# ---------------------------------------------------------------------------
# Core filtering function
# ---------------------------------------------------------------------------


def filter_state(
    state: Any,
    filter: StateFilter,
) -> Tuple[Any, List[str]]:
    """Apply *filter* to *state* and return (filtered_state, filtered_keys).

    ``state`` may be any value. Filtering is only applied when the value
    (or, recursively, a nested value) is a ``dict``. Non-dict primitives
    pass through unchanged.

    Returns
    -------
    filtered_state:
        The filtered value, with the same shape as the input but with
        sensitive keys excluded or masked.
    filtered_keys:
        Sorted list of unique key names (lowercased) that were either
        excluded or masked anywhere in the structure. Adapters surface
        this list as ``_filtered_keys`` metadata so customers can see
        what was clipped without exposing the values themselves.

    Notes
    -----
    * The *filter* parameter is positional but named ``filter`` for
      readability at call sites — even though it shadows the Python
      builtin, the scope is local and there is no real ambiguity.
    * Sets / tuples are NOT recursed into. Only ``dict`` and ``list``
      are walked. This matches the LangGraph reference implementation.
    """
    filtered_keys_set: set[str] = set()
    out = _filter_value(state, filter, filtered_keys_set)
    return out, sorted(filtered_keys_set)


def _filter_value(
    value: Any,
    flt: StateFilter,
    filtered_keys: set[str],
) -> Any:
    """Recursive helper that mutates *filtered_keys* as a side effect."""
    if isinstance(value, dict):
        return _filter_dict(value, flt, filtered_keys)
    if flt.recursive and isinstance(value, list):
        return [_filter_value(item, flt, filtered_keys) for item in value]
    return value


def _filter_dict(
    state: Dict[Any, Any],
    flt: StateFilter,
    filtered_keys: set[str],
) -> Dict[Any, Any]:
    """Apply exclude → mask → include to a single dict."""
    out: Dict[Any, Any] = {}
    for key, value in state.items():
        key_norm = str(key).lower()

        # 1. Exclude — drop the key entirely.
        if _matches_substring(key_norm, flt.exclude_keys):
            filtered_keys.add(key_norm)
            continue

        # 2. Mask — keep the key, replace the value.
        if _matches_substring(key_norm, flt.mask_keys):
            filtered_keys.add(key_norm)
            out[key] = REDACTED_PLACEHOLDER
            continue

        # 3. Recurse for nested dicts / lists when requested.
        if flt.recursive:
            value = _filter_value(value, flt, filtered_keys)

        out[key] = value

    # 4. Include allowlist — applied AFTER exclude/mask so the allowlist
    #    narrows the surviving surface, never widens it.
    if flt.include_keys is not None:
        narrowed: Dict[Any, Any] = {}
        for key, value in out.items():
            if str(key).lower() in flt.include_keys:
                narrowed[key] = value
            else:
                filtered_keys.add(str(key).lower())
        return narrowed

    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


_NON_WORD_RE = re.compile(r"[^a-z0-9]+")


def _normalize_key_for_match(key: str) -> str:
    """Strip non-alphanumeric chars so e.g. ``X-Api-Key`` matches ``api_key``.

    Both the candidate key and the configured substring are normalised
    via this function before substring containment is tested. This makes
    the filter resilient to header-style ``-``, screaming snake case,
    camelCase, and other delimiter variants.
    """
    return _NON_WORD_RE.sub("", key.lower())


def _matches_substring(key_norm: str, needles: FrozenSet[str]) -> bool:
    """Return ``True`` when *key_norm* contains any of *needles*.

    Both sides are normalised to alphanumeric-only lowercase so the
    comparison is robust across naming conventions.
    """
    if not needles:
        return False
    candidate = _normalize_key_for_match(key_norm)
    for needle in needles:
        if _normalize_key_for_match(needle) in candidate:
            return True
    return False


# ---------------------------------------------------------------------------
# Convenience: filter only specified payload fields
# ---------------------------------------------------------------------------


def filter_payload_fields(
    payload: Dict[str, Any],
    flt: StateFilter,
    fields: Iterable[str],
) -> List[str]:
    """In-place filter that only touches *fields* of *payload*.

    Many adapter payloads mix safe scalar metadata (``model``,
    ``latency_ms``, ``agent_name``) with potentially sensitive
    dict-shaped state (``input``, ``output``, ``messages``, ``deps``).
    Filtering the entire payload would rewrite the metadata too. This
    helper applies :func:`filter_state` ONLY to the named fields when
    they are dict-shaped (or list-of-dict-shaped) and leaves everything
    else untouched.

    The function records every key that was clipped, attaches the
    sorted list to ``payload['_filtered_keys']``, and returns the same
    list so the caller can inspect it without re-reading the payload.

    Returns the list of filtered keys (possibly empty).
    """
    all_filtered: set[str] = set()
    for fname in fields:
        if fname not in payload:
            continue
        original = payload[fname]
        if not isinstance(original, (dict, list)):
            continue
        filtered, keys = filter_state(original, flt)
        payload[fname] = filtered
        all_filtered.update(keys)

    if all_filtered:
        sorted_keys = sorted(all_filtered)
        # Merge with any pre-existing _filtered_keys (the caller may have
        # filtered another payload section earlier in the same emit).
        existing = payload.get("_filtered_keys")
        if isinstance(existing, list):
            merged = sorted(set(existing) | set(sorted_keys))
            payload["_filtered_keys"] = merged
            return merged
        payload["_filtered_keys"] = sorted_keys
        return sorted_keys
    return []


__all__ = [
    "DEFAULT_PII_EXCLUDE_KEYS",
    "REDACTED_PLACEHOLDER",
    "StateFilter",
    "default_state_filter",
    "filter_payload_fields",
    "filter_state",
]
