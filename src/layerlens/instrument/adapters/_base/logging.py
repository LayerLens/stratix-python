"""Tenant-aware logging utilities for LayerLens adapters.

Provides :class:`TenantContextLogAdapter` (a thin
:class:`logging.LoggerAdapter` subclass) and the convenience constructor
:func:`get_tenant_logger`. Every log record produced through the adapter
is enriched with the bound tenant's ``org_id`` in two surfaces:

* the ``extra`` keyword propagated to ``logging.LogRecord.__dict__`` so
  log handlers (JSON formatters, OTel log exporters, structlog
  processors) can promote it to a structured field;
* the ``"[org_id=...] "`` prefix on the formatted message so plain-text
  log lines carry the tenant binding without any handler config.

This implements Gap 4 of the multi-tenancy hardening contract
(CLAUDE.md "EVERY data operation must be scoped by tenant"). Adapter
log lines previously omitted ``org_id``, making per-tenant log
filtering and incident triage impossible. After this change every
emission, circuit-breaker state change, and sink dispatch failure logs
carries the tenant context end-to-end.

Thread-safety
-------------
The Python ``logging`` module is process-wide thread-safe; this adapter
adds no shared mutable state, only a per-instance ``org_id`` string.
Multiple adapters bound to different tenants can therefore share a
single underlying ``logging.Logger`` without leaking org_id across
instances.
"""

from __future__ import annotations

import logging
from typing import Any, Tuple, Mapping, MutableMapping

# Reserved key used both inside ``extra`` and as the formatted prefix.
# Matches :data:`layerlens.instrument.adapters._base.adapter.ORG_ID_FIELD`
# — the same canonical key used everywhere in the platform.
_ORG_ID_KEY: str = "org_id"


class TenantContextLogAdapter(logging.LoggerAdapter):  # type: ignore[type-arg]
    """:class:`logging.LoggerAdapter` that injects ``org_id`` into every record.

    Construction takes a base :class:`logging.Logger` and the tenant's
    ``org_id``. Each ``debug`` / ``info`` / ``warning`` / ``error`` /
    ``critical`` call:

    1. Adds ``org_id`` to the record's ``extra`` dict so structured
       handlers see it as a first-class field.
    2. Prepends ``"[org_id=<value>] "`` to the message body so flat-text
       log lines carry the tenant binding without handler config.

    The adapter is **per-instance** — adapters bound to different
    tenants must use distinct :class:`TenantContextLogAdapter` instances
    even when they share the same underlying logger name. The
    :func:`get_tenant_logger` factory enforces this by always creating a
    fresh instance.

    Example::

        log = TenantContextLogAdapter(logging.getLogger(__name__), org_id="org-A")
        log.warning("circuit breaker open")
        # record.extra["org_id"] == "org-A"
        # formatted message: "[org_id=org-A] circuit breaker open"
    """

    def __init__(self, logger: logging.Logger, org_id: str) -> None:
        if not isinstance(org_id, str) or not org_id.strip():
            raise ValueError(
                "TenantContextLogAdapter requires a non-empty org_id "
                "string. Construction without a tenant binding violates "
                "the multi-tenancy contract — see CLAUDE.md."
            )
        super().__init__(logger, {_ORG_ID_KEY: org_id})
        self._org_id: str = org_id

    @property
    def org_id(self) -> str:
        """The tenant ``org_id`` bound to this log adapter."""
        return self._org_id

    def process(
        self,
        msg: Any,
        kwargs: MutableMapping[str, Any],
    ) -> Tuple[Any, MutableMapping[str, Any]]:
        """Inject ``org_id`` into the record's ``extra`` and prefix the message.

        ``kwargs["extra"]`` is merged (caller-supplied keys win EXCEPT
        for ``org_id``, which the adapter always stamps to its bound
        tenant — caller cannot override the tenant binding via a stray
        ``extra={"org_id": ...}`` argument).
        """
        # Merge caller extras with the tenant binding. Tenant binding
        # always wins to prevent caller-supplied org_id from
        # impersonating a different tenant in log records.
        existing_extra: Mapping[str, Any] = kwargs.get("extra") or {}
        merged_extra: dict[str, Any] = dict(existing_extra)
        merged_extra[_ORG_ID_KEY] = self._org_id
        kwargs["extra"] = merged_extra

        # Prefix the message so plain-text handlers also surface the
        # tenant binding. Use repr-safe brackets to avoid collision with
        # other bracketed prefixes in adapter log lines.
        return f"[{_ORG_ID_KEY}={self._org_id}] {msg}", kwargs


def get_tenant_logger(name: str, org_id: str) -> TenantContextLogAdapter:
    """Construct a :class:`TenantContextLogAdapter` for ``name`` bound to ``org_id``.

    Convenience wrapper that mirrors the shape of
    :func:`logging.getLogger` so adapter modules can swap a plain
    ``logger = logging.getLogger(__name__)`` for
    ``logger = get_tenant_logger(__name__, self._org_id)`` with a
    one-line change.

    The underlying logger is shared across calls with the same name (as
    with :func:`logging.getLogger`); only the adapter wrapper is fresh
    per call so each adapter instance carries its own tenant binding.
    """
    base = logging.getLogger(name)
    return TenantContextLogAdapter(base, org_id)


__all__ = [
    "TenantContextLogAdapter",
    "get_tenant_logger",
]
