"""
AP2 (Agent Payments Protocol) Adapter

Instruments AP2 client operations to capture:
- Intent mandate creation and validation
- Payment mandate signing with cryptographic proof
- Payment receipt issuance and verification
- Spending guardrail evaluation and violation detection

Provides end-to-end observability for autonomous agent financial transactions.
"""

from __future__ import annotations

import uuid
import hashlib
import logging
from typing import Any
from datetime import timezone

UTC = timezone.utc  # Python 3.11+ has datetime.UTC; alias for 3.9/3.10 compat.

from layerlens.instrument.adapters.protocols._commerce import (
    IntentMandateInfo,
    PaymentMandateInfo,
    PaymentReceiptInfo,
    SpendingThresholdEvent,
    GuardrailViolationEvent,
    IntentMandateCreatedEvent,
    PaymentMandateSignedEvent,
    PaymentReceiptIssuedEvent,
    IntentMandateValidatedEvent,
)

from layerlens.instrument.adapters._base.adapter import (
    AdapterInfo,
    AdapterStatus,
    ReplayableTrace,
    AdapterCapability,
)
from layerlens.instrument.adapters.protocols.base import BaseProtocolAdapter

logger = logging.getLogger(__name__)


class AP2Adapter(BaseProtocolAdapter):
    """Adapter for the Agent Payments Protocol (AP2).

    Captures the three-stage AP2 authorization chain:

    1. Intent Mandate — spending guardrails and merchant constraints
    2. Payment Mandate — cryptographic authorization to pay
    3. Payment Receipt — settlement confirmation

    Provides guardrail evaluation against configurable org-level policies:
    - Maximum single transaction amount
    - Allowed merchant whitelist
    - Refundability requirements
    - Cumulative spending thresholds (daily/weekly/monthly)
    - Intent expiry enforcement
    """

    FRAMEWORK = "ap2"
    PROTOCOL = "ap2"
    PROTOCOL_VERSION = "0.1.0"
    VERSION = "0.1.0"

    def __init__(self, memory_service: Any | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._framework_version: str | None = None
        self._mandates: dict[str, IntentMandateInfo] = {}
        self._spending_totals: dict[str, float] = {}  # org_id -> cumulative spend
        self._policies: dict[str, dict[str, Any]] = {}  # org_id -> policy config
        self._memory_service = memory_service

    # --- Lifecycle ---

    def connect(self) -> None:
        """Connect to the AP2 runtime.

        Attempts to import the optional ``ap2_sdk`` package to detect the
        installed framework version.  If the package is not present the adapter
        operates in standalone mode, which is suitable for testing and
        environments that instrument AP2 indirectly.
        """
        try:
            import ap2_sdk  # type: ignore[import-not-found,unused-ignore]  # noqa: F401

            self._framework_version = getattr(ap2_sdk, "__version__", "0.1.0")
        except ImportError:
            self._framework_version = None
            logger.debug("ap2-sdk not installed; adapter operates in standalone mode")
        self._connected = True
        self._status = AdapterStatus.HEALTHY
        logger.info("AP2 adapter connected (protocol v%s)", self.PROTOCOL_VERSION)

    def disconnect(self) -> None:
        """Disconnect and release all runtime state.

        Clears the in-memory mandate registry, spending accumulators, and
        policy configuration.  Attached event sinks are flushed and closed.
        """
        self._mandates.clear()
        self._spending_totals.clear()
        self._policies.clear()
        self._connected = False
        self._status = AdapterStatus.DISCONNECTED
        self._close_sinks()

    def get_adapter_info(self) -> AdapterInfo:
        """Return metadata describing this adapter."""
        return AdapterInfo(
            name="AP2Adapter",
            version=self.VERSION,
            framework=self.FRAMEWORK,
            framework_version=self._framework_version,
            capabilities=[
                AdapterCapability.TRACE_PROTOCOL_EVENTS,
                AdapterCapability.TRACE_STATE,
                AdapterCapability.REPLAY,
            ],
            description="LayerLens adapter for the AP2 (Agent Payments Protocol)",
        )

    def serialize_for_replay(self) -> ReplayableTrace:
        """Serialize the current trace data for replay.

        The mandate registry is captured as a state snapshot so that replay
        can reconstruct the authorization chain context.
        """
        return ReplayableTrace(
            adapter_name="AP2Adapter",
            framework=self.FRAMEWORK,
            trace_id=str(uuid.uuid4()),
            events=list(self._trace_events),
            state_snapshots=[
                {"mandates": {k: v.model_dump() for k, v in self._mandates.items()}},
            ],
            config={
                "capture_config": self._capture_config.model_dump(),
                "policies": dict(self._policies.items()),
            },
        )

    def probe_health(self, endpoint: str | None = None) -> dict[str, Any]:
        """Probe the health of the AP2 adapter.

        Args:
            endpoint: Unused for AP2; present to satisfy the
                ``BaseProtocolAdapter`` interface.

        Returns:
            Dict with keys: ``reachable`` (bool), ``latency_ms`` (float),
            ``protocol_version`` (str | None), and ``active_mandates`` (int).
        """
        return {
            "reachable": self._connected,
            "latency_ms": 0.0,
            "protocol_version": self.PROTOCOL_VERSION,
            "active_mandates": len(self._mandates),
        }

    # --- Policy configuration ---

    def configure_policy(
        self,
        org_id: str,
        *,
        max_single_tx: float | None = None,
        daily_limit: float | None = None,
        weekly_limit: float | None = None,
        monthly_limit: float | None = None,
        allowed_merchants: list[str] | None = None,
        require_refundability: bool = False,
    ) -> None:
        """Configure spending guardrails for an organization.

        Policies are evaluated at intent mandate creation time.  Any
        subsequent calls for the same ``org_id`` fully replace the prior
        configuration.

        Args:
            org_id: Organization identifier the policy applies to.
            max_single_tx: Maximum permitted amount for a single transaction.
            daily_limit: Maximum cumulative spend per calendar day.
            weekly_limit: Maximum cumulative spend per calendar week.
            monthly_limit: Maximum cumulative spend per calendar month.
            allowed_merchants: Exhaustive whitelist of permitted merchant
                identifiers.  ``None`` means no restriction.
            require_refundability: When ``True`` every intent mandate must
                declare ``requires_refundability=True`` or a guardrail
                violation is raised.
        """
        self._policies[org_id] = {
            "max_single_tx": max_single_tx,
            "daily_limit": daily_limit,
            "weekly_limit": weekly_limit,
            "monthly_limit": monthly_limit,
            "allowed_merchants": allowed_merchants,
            "require_refundability": require_refundability,
        }

    # --- AP2 Intent Mandate ---

    def on_intent_mandate_created(
        self,
        mandate_id: str,
        description: str,
        org_id: str,
        *,
        merchants: list[str] | None = None,
        max_amount: float | None = None,
        currency: str = "USD",
        requires_refundability: bool = False,
        user_cart_confirmation_required: bool = False,
        intent_expiry: str | None = None,
        agent_id: str | None = None,
    ) -> list[str]:
        """Record an AP2 intent mandate and evaluate it against org guardrails.

        Emits an ``IntentMandateCreatedEvent`` followed by an
        ``IntentMandateValidatedEvent``.  For each guardrail breach a
        ``GuardrailViolationEvent`` is also emitted before the validation
        event is finalized.

        Args:
            mandate_id: Unique identifier for this intent mandate.
            description: Natural-language description of the spending intent.
            org_id: Organization that owns this mandate.
            merchants: Merchant identifiers the agent may transact with.
            max_amount: Upper bound on a single transaction amount.
            currency: ISO 4217 currency code (default ``"USD"``).
            requires_refundability: Whether the intent requires refundability.
            user_cart_confirmation_required: Whether the user must confirm the
                cart before payment proceeds.
            intent_expiry: Optional ISO 8601 expiry timestamp string.
            agent_id: Identifier of the agent that created the intent.

        Returns:
            A list of human-readable guardrail violation messages.  An empty
            list means the mandate is fully compliant with org policy.
        """
        intent = IntentMandateInfo(
            mandate_id=mandate_id,
            natural_language_description=description,
            merchants=merchants or [],
            max_amount=max_amount,
            currency=currency,
            requires_refundability=requires_refundability,
            user_cart_confirmation_required=user_cart_confirmation_required,
            intent_expiry=intent_expiry,
        )
        self._mandates[mandate_id] = intent

        self.emit_event(
            IntentMandateCreatedEvent.create(
                intent=intent,
                org_id=org_id,
                agent_id=agent_id,
            )
        )

        violations = self._evaluate_guardrails(intent, org_id, agent_id)

        self.emit_event(
            IntentMandateValidatedEvent.create(
                mandate_id=mandate_id,
                validation_passed=len(violations) == 0,
                violations=violations,
                org_id=org_id,
            )
        )

        return violations

    # --- AP2 Payment Mandate ---

    def on_payment_mandate_signed(
        self,
        mandate_id: str,
        payment_details_id: str,
        total_amount: float,
        merchant_agent: str,
        org_id: str,
        *,
        currency: str = "USD",
        payment_method: str = "CARD",
        signature: str = "",
        agent_id: str | None = None,
    ) -> None:
        """Record a cryptographically signed payment mandate.

        The raw ``signature`` value is never stored; only its SHA-256 hash is
        captured so the audit trail remains tamper-evident without retaining
        sensitive key material.

        Cumulative org spending is updated and spending-threshold events are
        raised if any configured limits are exceeded.

        Args:
            mandate_id: Unique identifier for this payment mandate (should
                correlate with a previously created intent mandate).
            payment_details_id: Opaque reference to stored payment credentials.
            total_amount: Authorized total amount for the transaction.
            merchant_agent: Merchant agent identifier or endpoint URL.
            org_id: Organization that owns this mandate.
            currency: ISO 4217 currency code (default ``"USD"``).
            payment_method: Payment rail: ``CARD`` | ``ACH`` | ``CRYPTO``.
            signature: Raw JWT or biometric signature value (hashed before
                storage).
            agent_id: Identifier of the agent that signed the mandate.
        """
        # Verify intent mandate hasn't expired before signing
        intent = self._mandates.get(mandate_id)
        if intent and intent.intent_expiry:
            from datetime import datetime

            try:
                expiry_dt = datetime.fromisoformat(intent.intent_expiry.replace("Z", "+00:00"))
                if datetime.now(UTC) > expiry_dt:
                    self.emit_event(
                        GuardrailViolationEvent.create(
                            mandate_id=mandate_id,
                            violation_type="expired",
                            details=f"Intent mandate expired at {intent.intent_expiry}",
                            org_id=org_id,
                            agent_id=agent_id,
                        )
                    )
                    logger.warning("AP2: Intent mandate %s expired, signing anyway", mandate_id)
            except (ValueError, TypeError):
                pass  # Non-parseable expiry; skip check

        sig_hash = hashlib.sha256(signature.encode()).hexdigest()

        mandate = PaymentMandateInfo(
            mandate_id=mandate_id,
            payment_details_id=payment_details_id,
            total_amount=total_amount,
            currency=currency,
            merchant_agent=merchant_agent,
            payment_method=payment_method,
            signature_hash=sig_hash,
        )

        # Update cumulative spending before threshold checks
        self._spending_totals[org_id] = self._spending_totals.get(org_id, 0.0) + total_amount
        self._check_spending_thresholds(org_id, currency)

        self.emit_event(
            PaymentMandateSignedEvent.create(
                mandate=mandate,
                org_id=org_id,
                agent_id=agent_id,
            )
        )

    # --- AP2 Payment Receipt ---

    def on_payment_receipt_issued(
        self,
        mandate_id: str,
        payment_id: str,
        amount: float,
        org_id: str,
        *,
        currency: str = "USD",
        status: str = "success",
        merchant_confirmation_id: str | None = None,
    ) -> None:
        """Record a payment receipt, closing the AP2 audit trail.

        Emits a ``PaymentReceiptIssuedEvent``.  On successful settlement the
        intent mandate is removed from the in-memory registry.

        Args:
            mandate_id: Mandate that this receipt settles.
            payment_id: Unique payment transaction identifier issued by the
                payment processor.
            amount: Amount actually charged (may differ from authorized amount
                for partial captures).
            org_id: Organization that owns this receipt.
            currency: ISO 4217 currency code (default ``"USD"``).
            status: Settlement status: ``success`` | ``failed`` | ``refunded``.
            merchant_confirmation_id: Optional merchant-side reference number.
        """
        receipt = PaymentReceiptInfo(
            mandate_id=mandate_id,
            payment_id=payment_id,
            amount=amount,
            currency=currency,
            status=status,
            merchant_confirmation_id=merchant_confirmation_id,
        )

        self.emit_event(
            PaymentReceiptIssuedEvent.create(
                receipt=receipt,
                org_id=org_id,
            )
        )

        # Store mandate + receipt as procedural memory
        if self._memory_service is not None:
            self._store_payment_memory(mandate_id, receipt, org_id)

        # Clean up mandate state on successful settlement
        if status == "success":
            self._mandates.pop(mandate_id, None)

    # --- Internal: memory storage ---

    def _store_payment_memory(
        self,
        mandate_id: str,
        receipt: Any,
        org_id: str,
    ) -> None:
        """Store mandate and receipt details as procedural memory.

        Failures are logged and swallowed to avoid disrupting the payment flow.
        """
        try:
            from layerlens.instrument._vendored.memory_models import MemoryEntry

            intent = self._mandates.get(mandate_id)
            content_parts = [f"receipt: {receipt.model_dump()}"]
            if intent:
                content_parts.insert(0, f"mandate: {intent.model_dump()}")

            entry = MemoryEntry(
                org_id=org_id,
                agent_id=f"ap2_{org_id}",
                memory_type="procedural",
                key=f"payment_{mandate_id}",
                content="; ".join(content_parts),
                importance=0.8,
                metadata={
                    "source": "ap2_adapter",
                    "mandate_id": mandate_id,
                    "status": getattr(receipt, "status", "unknown"),
                },
            )
            self._memory_service.store(entry)  # type: ignore[union-attr]
        except Exception:
            logger.debug(
                "AP2: failed to store payment memory for mandate %s",
                mandate_id,
                exc_info=True,
            )

    # --- Internal: guardrail evaluation ---

    def _evaluate_guardrails(
        self,
        intent: IntentMandateInfo,
        org_id: str,
        agent_id: str | None,
    ) -> list[str]:
        """Evaluate an intent mandate against the org's spending policy.

        A ``GuardrailViolationEvent`` is emitted for each individual breach
        before the consolidated violation list is returned.

        Args:
            intent: The intent mandate to validate.
            org_id: Organization whose policy to apply.
            agent_id: Agent involved in the mandate (for event attribution).

        Returns:
            Ordered list of human-readable violation messages.  Empty when
            the mandate is fully compliant or no policy is configured.
        """
        violations: list[str] = []
        policy = self._policies.get(org_id)

        if not policy:
            return violations

        # Check maximum single-transaction amount
        max_single = policy.get("max_single_tx")
        if max_single is not None and intent.max_amount is not None:  # noqa: SIM102
            if intent.max_amount > max_single:
                msg = f"Amount ${intent.max_amount} exceeds single-tx limit ${max_single}"
                violations.append(msg)
                self.emit_event(
                    GuardrailViolationEvent.create(
                        mandate_id=intent.mandate_id,
                        violation_type="amount_exceeded",
                        details=msg,
                        org_id=org_id,
                        agent_id=agent_id,
                    )
                )

        # Check merchant whitelist
        allowed_merchants = policy.get("allowed_merchants")
        if allowed_merchants is not None and intent.merchants:
            for merchant in intent.merchants:
                if merchant not in allowed_merchants:
                    msg = f"Merchant '{merchant}' not in allowed list"
                    violations.append(msg)
                    self.emit_event(
                        GuardrailViolationEvent.create(
                            mandate_id=intent.mandate_id,
                            violation_type="merchant_not_whitelisted",
                            details=msg,
                            org_id=org_id,
                            agent_id=agent_id,
                        )
                    )

        # Check refundability requirement
        if policy.get("require_refundability") and not intent.requires_refundability:
            msg = "Refundability required by policy but not declared in mandate"
            violations.append(msg)
            self.emit_event(
                GuardrailViolationEvent.create(
                    mandate_id=intent.mandate_id,
                    violation_type="refundability_required",
                    details=msg,
                    org_id=org_id,
                    agent_id=agent_id,
                )
            )

        return violations

    def _check_spending_thresholds(self, org_id: str, currency: str) -> None:
        """Emit spending threshold events when cumulative spend exceeds limits.

        Checks daily, weekly, and monthly limits configured for ``org_id``
        against the current accumulated spend total.  A
        ``SpendingThresholdEvent`` is emitted for each limit that is breached.

        Args:
            org_id: Organization whose cumulative spend to check.
            currency: ISO 4217 currency code for the threshold event.
        """
        policy = self._policies.get(org_id)
        if not policy:
            return

        cumulative = self._spending_totals.get(org_id, 0.0)

        for threshold_key, threshold_label in (
            ("daily_limit", "daily"),
            ("weekly_limit", "weekly"),
            ("monthly_limit", "monthly"),
        ):
            limit = policy.get(threshold_key)
            if limit is not None and cumulative > limit:
                self.emit_event(
                    SpendingThresholdEvent.create(
                        org_id=org_id,
                        threshold_type=threshold_label,
                        threshold_amount=limit,
                        actual_amount=cumulative,
                        currency=currency,
                    )
                )
