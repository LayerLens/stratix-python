"""Run-scoped cumulative-spend ledger + per-run dollar ceiling (A12 / A13).

A reusable primitive for the payment-autonomy corner. It answers two
agentic-autonomy invariants that an *instance* attribute cannot (the prior AP2
adapter held ``_cumulative_spend`` on ``self``, so two adapter instances in one
run — or a re-instrument after reconnect — each started at zero and could spend
``N×`` the intended run budget):

* **Run-scoped cumulative cap.** The running total lives in a ``ContextVar``
  (mirroring :mod:`layerlens.instrument._context`), so every adapter instance
  active in the same run/session shares ONE total. Reset per logical run, not
  per object.
* **Per-run dollar ceiling.** A configurable dollar ceiling that trips a guard
  BEFORE a charge is accrued — distinct from the collector's ``MAX_EVENTS``
  count cap (which counts telemetry events, not money). A charge that would push
  the run past the ceiling is refused; the caller emits ``policy.violation`` and
  does not call through.

Currency policy (A13, conservative / fail-closed)
-------------------------------------------------
Amounts carry an ISO-4217 currency (AP2 ``PaymentCurrencyAmount``). We do NOT
silently sum mixed currencies as if equal (e.g. ¥100 != £100). The ledger is
configured with ONE accounting currency. A charge:

* in the ledger's currency accrues to the running total and is checked against
  the ceiling;
* in a DIFFERENT currency is treated CONSERVATIVELY — it cannot be proven to be
  under a ceiling denominated in another currency, so when a ceiling is set a
  cross-currency charge is REFUSED (``CURRENCY_MISMATCH``) rather than waved
  through. With no ceiling configured, foreign-currency spend is tracked in a
  separate per-currency bucket (never folded into the home total) so observability
  is not blinded, but it is never compared across currencies as if equal.

This primitive is payment-agnostic: AP2 feeds it today; UCP ``complete_checkout``
and other spend paths will feed the SAME ledger later so one run's ceiling spans
every adapter that can move money.
"""

from __future__ import annotations

from typing import Dict, Optional, NamedTuple
from contextvars import ContextVar
from dataclasses import field, dataclass

# A generous default so the ceiling never surprises an integrator who did not opt
# in to a tighter budget, while still being a real backstop against a runaway
# autonomous loop. Override per run via ``configure_ledger(ceiling_usd=...)``.
DEFAULT_CEILING: float = 10_000.0
DEFAULT_CURRENCY: str = "USD"


class CeilingCheck(NamedTuple):
    """Outcome of a pre-charge ceiling check."""

    allowed: bool
    reason_code: Optional[str]  # None when allowed
    projected: float  # home-currency running total INCLUDING this charge (if home ccy)
    ceiling: Optional[float]
    currency: str


@dataclass
class SpendLedger:
    """Per-run spend state. Lives in a ContextVar, shared across adapters in a run."""

    #: Accounting currency for the home running total + the ceiling.
    currency: str = DEFAULT_CURRENCY
    #: Per-run dollar ceiling in :attr:`currency`. ``None`` disables the ceiling
    #: (cumulative is still tracked for observability).
    ceiling: Optional[float] = DEFAULT_CEILING
    #: Running total in the home :attr:`currency`.
    spent: float = 0.0
    #: Foreign-currency spend, bucketed per ISO-4217 code (never folded into
    #: :attr:`spent` — currencies are not summed as if equal).
    foreign: Dict[str, float] = field(default_factory=dict)


_current_ledger: ContextVar[Optional[SpendLedger]] = ContextVar("_current_ledger", default=None)


def _norm_ccy(currency: Optional[str]) -> str:
    """Normalize a currency code; a missing currency is treated conservatively as
    the home currency would NOT be — it is unknown, so it gets its own bucket."""
    if not currency:
        return "?"  # unknown currency: never folded into the home total
    return str(currency).upper()


def get_ledger() -> SpendLedger:
    """Return the current run's ledger, creating a default one if none is set.

    The first caller in a run that has not explicitly configured a ledger gets
    the generous default ceiling. Subsequent callers (other adapter instances)
    share it.
    """
    ledger = _current_ledger.get()
    if ledger is None:
        ledger = SpendLedger()
        _current_ledger.set(ledger)
    return ledger


def configure_ledger(
    *,
    ceiling_usd: Optional[float] = DEFAULT_CEILING,
    currency: str = DEFAULT_CURRENCY,
) -> SpendLedger:
    """Install a fresh ledger for the current run and return it.

    Call once at run start (or per :class:`~layerlens.instrument._context.RunState`)
    to set the per-run ceiling + accounting currency. Returns the new ledger so a
    caller can hold a reference; later adapters just call :func:`get_ledger`.
    """
    ledger = SpendLedger(currency=_norm_ccy(currency), ceiling=ceiling_usd)
    _current_ledger.set(ledger)
    return ledger


def set_ledger(ledger: Optional[SpendLedger]) -> object:
    """Bind *ledger* (or clear with ``None``) to the current context.

    Returns the ContextVar token so the caller can restore the previous ledger
    with :func:`reset_ledger` — the same pattern as ``_current_collector``.
    """
    return _current_ledger.set(ledger)


def reset_ledger(token: object) -> None:
    """Restore the ledger bound before the matching :func:`set_ledger`."""
    _current_ledger.reset(token)  # type: ignore[arg-type]


def current_spend(currency: Optional[str] = None) -> float:
    """Return the running total for *currency* (home currency when omitted)."""
    ledger = get_ledger()
    ccy = ledger.currency if currency is None else _norm_ccy(currency)
    if ccy == ledger.currency:
        return ledger.spent
    return ledger.foreign.get(ccy, 0.0)


def check_ceiling(amount: float, currency: Optional[str] = None) -> CeilingCheck:
    """Would charging *amount* (in *currency*) breach the per-run ceiling?

    Pure / side-effect-free: does NOT accrue. Call this BEFORE the charge; only
    :func:`add_spend` after the charge is allowed and executed.

    * Home-currency charge: allowed iff ``spent + amount <= ceiling`` (or no
      ceiling). Returns the projected new total.
    * Foreign-currency charge WITH a ceiling set: REFUSED (``CURRENCY_MISMATCH``)
      — we cannot prove a ¥ charge is under a £ ceiling. Fail closed.
    * Foreign-currency charge with NO ceiling: allowed (tracked separately).
    """
    ledger = get_ledger()
    ccy = ledger.currency if currency is None else _norm_ccy(currency)
    value = float(amount or 0.0)

    if ccy != ledger.currency:
        if ledger.ceiling is not None:
            return CeilingCheck(
                allowed=False,
                reason_code="CURRENCY_MISMATCH",
                projected=ledger.foreign.get(ccy, 0.0) + value,
                ceiling=ledger.ceiling,
                currency=ccy,
            )
        return CeilingCheck(
            allowed=True,
            reason_code=None,
            projected=ledger.foreign.get(ccy, 0.0) + value,
            ceiling=None,
            currency=ccy,
        )

    projected = ledger.spent + value
    if ledger.ceiling is not None and projected > ledger.ceiling:
        return CeilingCheck(
            allowed=False,
            reason_code="RUN_CEILING_EXCEEDED",
            projected=projected,
            ceiling=ledger.ceiling,
            currency=ccy,
        )
    return CeilingCheck(allowed=True, reason_code=None, projected=projected, ceiling=ledger.ceiling, currency=ccy)


def add_spend(amount: float, currency: Optional[str] = None) -> float:
    """Accrue *amount* (in *currency*) to the run total and return the new total.

    Home-currency spend accrues to :attr:`SpendLedger.spent`; foreign-currency
    spend accrues to its own per-currency bucket (never folded in). Call ONLY
    after :func:`check_ceiling` allowed it and the underlying charge executed.
    """
    ledger = get_ledger()
    ccy = ledger.currency if currency is None else _norm_ccy(currency)
    value = float(amount or 0.0)
    if ccy == ledger.currency:
        ledger.spent += value
        return ledger.spent
    ledger.foreign[ccy] = ledger.foreign.get(ccy, 0.0) + value
    return ledger.foreign[ccy]
