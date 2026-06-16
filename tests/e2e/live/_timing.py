"""Timing contract for the live suite — the single source of truth for every
poll loop that waits on the platform or on SDK background machinery.

The live suite is *eventually consistent by design* (see README.md §7): trace
persistence lags the upload, self-flushing adapters hand off to background
upload machinery, and the platform's integration-status sweeper runs on a
~30s cycle. Asserting any of those facts immediately after the triggering
action would be flaky by design, so every harness polls with the bounded
budgets defined here instead. Tune a timeout in exactly one place — this
module — never at a call site.

Consumers:

- ``_harness._poll_get``                    — trace read-back after upload
- ``_framework_harness.run_self_flushing_case`` — self-flush poll-drain
- ``_linkage.poll_status_healthy``          — integration-status poll

(The Langfuse scenario keeps its own ingestion poll: that budget belongs to
the *external* Langfuse service, not to the LayerLens platform contract.)
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Trace read-back (L2 persistence): _harness._poll_get
# --------------------------------------------------------------------------- #
TRACE_READBACK_ATTEMPTS: int = 12
"""How many times to re-``GET`` an uploaded trace before declaring it lost.

``traces.upload`` returns a ``trace_id`` as soon as ingestion *accepts* the
payload; the trace only becomes fetchable once the backend has persisted it
(ingestion API -> S3 -> trace store), which lags by a few seconds on a busy
or cold backend. 12 attempts x 1s has absorbed every persistence lag seen in
the audits; a trace still missing after that signals unhealthy ingestion, not
slowness (README §5 "trace not found after polling")."""

TRACE_READBACK_DELAY_S: float = 1.0
"""Pause between trace read-back attempts. 1s is coarse enough not to hammer
the API and fine enough that the common one-beat persistence lag costs ~1s."""


# --------------------------------------------------------------------------- #
# Self-flushing adapters: _framework_harness.run_self_flushing_case
# --------------------------------------------------------------------------- #
SELF_FLUSH_DEADLINE_S: float = 30.0
"""Total budget for a self-flushing adapter's trace id to be captured.

Self-flushing adapters (openai_agents, crewai, llamaindex, ...) upload their
own collector instead of emitting into the ambient one — some eagerly on
trace-end, some *deferred* via an event bus / background thread (crewai can
enqueue several seconds after the runner returns), and the upload itself is a
real HTTP call. 30s covers the slowest observed deferred flush with margin;
an adapter that has not flushed by then is mis-registered (wrong
``self_flushing`` flag — README §7 troubleshooting), not slow."""

SELF_FLUSH_POLL_INTERVAL_S: float = 0.5
"""Pause between checks for the captured trace id. Short, because the flush
usually lands within the first second or two and the check is in-process."""

SELF_FLUSH_DRAIN_TIMEOUT_S: int = 20
"""Per-iteration budget for ``_upload.shutdown_uploads`` to drain the SDK's
background upload queue. It must fit inside ``SELF_FLUSH_DEADLINE_S`` while
still allowing one slow real HTTP upload to finish; shutdown is safe to call
repeatedly (channels re-create on demand for anything enqueued later)."""


# --------------------------------------------------------------------------- #
# Platform-side inbound linkage: _linkage.poll_status_healthy
# --------------------------------------------------------------------------- #
LINKAGE_STATUS_TIMEOUT_DEFAULT_S: float = 90.0
"""Default budget for an integration's status to flip to ``Healthy``.

``traces.integration_id`` is stamped synchronously at trace-create, but the
status transition (``Inactive -> Healthy``) is owned by a periodic
platform-side sweeper with a ~30s cycle. 90s = ~3 sweep intervals, so one
missed/late sweep does not fail the run. Overridable per run via the
``LAYERLENS_LIVE_LINKAGE_TIMEOUT`` env var (see ``_linkage``)."""

LINKAGE_STATUS_POLL_INTERVAL_S: float = 5.0
"""Pause between integration-status reads. Each read is a real platform API
call, and the underlying sweeper only acts every ~30s — polling faster than
5s burns requests without observing anything new."""
