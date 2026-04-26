"""
Commerce & Payment Protocol Events

Layer L7 — Commerce protocol events for AP2 (Agent Payments),
UCP (Universal Commerce Protocol), and A2UI (Agent-to-User Interface).
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class IntentMandateInfo(BaseModel):
    """Parsed AP2 intent mandate with spending guardrails."""

    mandate_id: str = Field(description="Unique mandate identifier")
    natural_language_description: str = Field(description="Human-readable spending intent")
    merchants: list[str] = Field(
        default_factory=list, description="Allowlisted merchant identifiers"
    )
    max_amount: Optional[float] = Field(default=None, description="Maximum permitted spend")
    currency: str = Field(default="USD", description="ISO 4217 currency code")
    requires_refundability: bool = Field(
        default=False, description="Whether refundability is required"
    )
    user_cart_confirmation_required: bool = Field(
        default=False,
        description="Whether user must confirm cart before payment",
    )
    intent_expiry: Optional[str] = Field(default=None, description="Expiry timestamp in ISO 8601")


class PaymentMandateInfo(BaseModel):
    """Signed payment authorization details."""

    mandate_id: str = Field(description="Unique mandate identifier")
    payment_details_id: str = Field(description="Opaque reference to stored payment details")
    total_amount: float = Field(description="Authorized total amount")
    currency: str = Field(default="USD", description="ISO 4217 currency code")
    merchant_agent: str = Field(description="Merchant agent identifier or URL")
    payment_method: str = Field(default="CARD", description="Payment method: CARD | ACH | CRYPTO")
    signature_hash: str = Field(description="sha256 of JWT/biometric signature")


class PaymentReceiptInfo(BaseModel):
    """Settlement confirmation."""

    mandate_id: str = Field(description="Mandate this receipt settles")
    payment_id: str = Field(description="Unique payment transaction identifier")
    amount: float = Field(description="Amount actually charged")
    currency: str = Field(default="USD", description="ISO 4217 currency code")
    status: str = Field(description="Settlement status: success | failed | refunded")
    merchant_confirmation_id: Optional[str] = Field(
        default=None,
        description="Merchant-side confirmation reference",
    )


class SupplierInfo(BaseModel):
    """UCP supplier identity and capability advertisement."""

    supplier_id: str = Field(description="Unique supplier identifier")
    name: str = Field(description="Human-readable supplier name")
    profile_url: str = Field(description="URL of the supplier's UCP profile")
    capabilities: list[str] = Field(
        default_factory=list,
        description="Declared capability identifiers",
    )


class LineItemInfo(BaseModel):
    """A single line item in a UCP checkout session."""

    item_id: str = Field(description="Supplier-scoped item identifier")
    name: Optional[str] = Field(default=None, description="Human-readable item name")
    quantity: int = Field(default=1, description="Quantity ordered")
    unit_price: Optional[float] = Field(default=None, description="Price per unit")
    currency: str = Field(default="USD", description="ISO 4217 currency code")


# ---------------------------------------------------------------------------
# L7a — AP2 (Agent Payments Protocol) Events
# ---------------------------------------------------------------------------


class IntentMandateCreatedEvent(BaseModel):
    """
    L7a: Emitted when an AP2 intent mandate is created by the user or agent.

    Captures spending intent before any payment authorization occurs.
    """

    event_type: str = Field(
        default="commerce.payment.intent_created",
        description="Event type identifier",
    )
    layer: str = Field(default="L7a", description="Layer identifier")
    intent: IntentMandateInfo = Field(description="Parsed intent mandate")
    org_id: str = Field(description="Organization that owns this mandate")
    agent_id: Optional[str] = Field(default=None, description="Agent that created the intent")

    @classmethod
    def create(
        cls,
        intent: IntentMandateInfo,
        org_id: str,
        *,
        agent_id: str | None = None,
    ) -> IntentMandateCreatedEvent:
        return cls(
            intent=intent,
            org_id=org_id,
            agent_id=agent_id,
        )


class IntentMandateValidatedEvent(BaseModel):
    """
    L7a: Emitted when an AP2 intent mandate is validated against guardrails.

    Records whether the mandate passed policy checks and any violations found.
    """

    event_type: str = Field(
        default="commerce.payment.intent_validated",
        description="Event type identifier",
    )
    layer: str = Field(default="L7a", description="Layer identifier")
    mandate_id: str = Field(description="Mandate that was validated")
    validation_passed: bool = Field(description="True if all guardrails passed")
    violations: list[str] = Field(
        default_factory=list,
        description="Guardrail violation messages",
    )
    org_id: str = Field(description="Organization that owns this mandate")

    @classmethod
    def create(
        cls,
        mandate_id: str,
        validation_passed: bool,
        org_id: str,
        *,
        violations: list[str] | None = None,
    ) -> IntentMandateValidatedEvent:
        return cls(
            mandate_id=mandate_id,
            validation_passed=validation_passed,
            violations=violations or [],
            org_id=org_id,
        )


class PaymentMandateSignedEvent(BaseModel):
    """
    L7a: Emitted when a payment mandate is cryptographically signed and authorized.

    Records the full signed authorization before settlement.
    """

    event_type: str = Field(
        default="commerce.payment.mandate_signed",
        description="Event type identifier",
    )
    layer: str = Field(default="L7a", description="Layer identifier")
    mandate: PaymentMandateInfo = Field(description="Signed payment mandate details")
    org_id: str = Field(description="Organization that owns this mandate")
    agent_id: Optional[str] = Field(default=None, description="Agent that signed the mandate")

    @classmethod
    def create(
        cls,
        mandate: PaymentMandateInfo,
        org_id: str,
        *,
        agent_id: str | None = None,
    ) -> PaymentMandateSignedEvent:
        return cls(
            mandate=mandate,
            org_id=org_id,
            agent_id=agent_id,
        )


class PaymentReceiptIssuedEvent(BaseModel):
    """
    L7a: Emitted when a payment receipt is issued following settlement.

    Terminal event in the AP2 payment lifecycle.
    """

    event_type: str = Field(
        default="commerce.payment.receipt_issued",
        description="Event type identifier",
    )
    layer: str = Field(default="L7a", description="Layer identifier")
    receipt: PaymentReceiptInfo = Field(description="Settlement confirmation details")
    org_id: str = Field(description="Organization that owns this receipt")

    @classmethod
    def create(
        cls,
        receipt: PaymentReceiptInfo,
        org_id: str,
    ) -> PaymentReceiptIssuedEvent:
        return cls(
            receipt=receipt,
            org_id=org_id,
        )


class GuardrailViolationEvent(BaseModel):
    """
    L7a: Emitted when a payment attempt violates a spending guardrail.

    Always emitted regardless of whether the payment was blocked.
    """

    event_type: str = Field(
        default="commerce.payment.guardrail_violation",
        description="Event type identifier",
    )
    layer: str = Field(default="L7a", description="Layer identifier")
    mandate_id: str = Field(description="Mandate that triggered the violation")
    violation_type: str = Field(
        description=(
            "Violation category: amount_exceeded | merchant_not_whitelisted"
            " | expired | refundability_required"
        ),
    )
    details: str = Field(description="Human-readable violation explanation")
    org_id: str = Field(description="Organization that owns this mandate")
    agent_id: Optional[str] = Field(default=None, description="Agent involved in the attempt")
    blocked: bool = Field(default=True, description="Whether the payment was blocked")

    @classmethod
    def create(
        cls,
        mandate_id: str,
        violation_type: str,
        details: str,
        org_id: str,
        *,
        agent_id: str | None = None,
        blocked: bool = True,
    ) -> GuardrailViolationEvent:
        return cls(
            mandate_id=mandate_id,
            violation_type=violation_type,
            details=details,
            org_id=org_id,
            agent_id=agent_id,
            blocked=blocked,
        )


class SpendingThresholdEvent(BaseModel):
    """
    L7a: Emitted when cumulative spend crosses a configured threshold.

    Supports single-transaction, daily, weekly, and monthly threshold monitoring.
    """

    event_type: str = Field(
        default="commerce.payment.threshold_exceeded",
        description="Event type identifier",
    )
    layer: str = Field(default="L7a", description="Layer identifier")
    org_id: str = Field(description="Organization whose threshold was exceeded")
    threshold_type: str = Field(
        description="Threshold window: single_tx | daily | weekly | monthly",
    )
    threshold_amount: float = Field(description="Configured threshold value")
    actual_amount: float = Field(description="Actual spend that exceeded the threshold")
    currency: str = Field(default="USD", description="ISO 4217 currency code")

    @classmethod
    def create(
        cls,
        org_id: str,
        threshold_type: str,
        threshold_amount: float,
        actual_amount: float,
        *,
        currency: str = "USD",
    ) -> SpendingThresholdEvent:
        return cls(
            org_id=org_id,
            threshold_type=threshold_type,
            threshold_amount=threshold_amount,
            actual_amount=actual_amount,
            currency=currency,
        )


# ---------------------------------------------------------------------------
# L7b — UCP (Universal Commerce Protocol) Events
# ---------------------------------------------------------------------------


class SupplierDiscoveredEvent(BaseModel):
    """
    L7b: Emitted when a UCP supplier is discovered via well-known endpoint,
    registry lookup, or referral.
    """

    event_type: str = Field(
        default="commerce.supplier.discovered",
        description="Event type identifier",
    )
    layer: str = Field(default="L7b", description="Layer identifier")
    supplier: SupplierInfo = Field(description="Discovered supplier details")
    org_id: str = Field(description="Organization performing the discovery")
    discovery_method: str = Field(
        default="well_known",
        description="How supplier was found: well_known | registry | referral",
    )

    @classmethod
    def create(
        cls,
        supplier: SupplierInfo,
        org_id: str,
        *,
        discovery_method: str = "well_known",
    ) -> SupplierDiscoveredEvent:
        return cls(
            supplier=supplier,
            org_id=org_id,
            discovery_method=discovery_method,
        )


class CatalogBrowsedEvent(BaseModel):
    """
    L7b: Emitted when an agent browses a supplier's UCP catalog.

    High-frequency: captures browse activity without individual item detail.
    """

    event_type: str = Field(
        default="commerce.catalog.browsed",
        description="Event type identifier",
    )
    layer: str = Field(default="L7b", description="Layer identifier")
    supplier_id: str = Field(description="Supplier whose catalog was browsed")
    items_viewed: int = Field(default=0, description="Number of catalog items viewed")
    items_selected: int = Field(default=0, description="Number of items added to session")
    org_id: str = Field(description="Organization performing the browse")

    @classmethod
    def create(
        cls,
        supplier_id: str,
        org_id: str,
        *,
        items_viewed: int = 0,
        items_selected: int = 0,
    ) -> CatalogBrowsedEvent:
        return cls(
            supplier_id=supplier_id,
            org_id=org_id,
            items_viewed=items_viewed,
            items_selected=items_selected,
        )


class CheckoutCreatedEvent(BaseModel):
    """
    L7b: Emitted when a UCP checkout session is initiated.

    Captures the full line-item basket before payment handoff to AP2.
    """

    event_type: str = Field(
        default="commerce.checkout.created",
        description="Event type identifier",
    )
    layer: str = Field(default="L7b", description="Layer identifier")
    checkout_session_id: str = Field(description="Unique checkout session identifier")
    supplier_id: str = Field(description="Supplier hosting the checkout")
    line_items: list[LineItemInfo] = Field(
        default_factory=list,
        description="Items in the checkout basket",
    )
    total_amount: float = Field(description="Pre-tax total of all line items")
    currency: str = Field(default="USD", description="ISO 4217 currency code")
    idempotency_key: Optional[str] = Field(
        default=None,
        description="Client-supplied idempotency key",
    )
    org_id: str = Field(description="Organization initiating checkout")

    @classmethod
    def create(
        cls,
        checkout_session_id: str,
        supplier_id: str,
        total_amount: float,
        org_id: str,
        *,
        line_items: list[LineItemInfo] | None = None,
        currency: str = "USD",
        idempotency_key: str | None = None,
    ) -> CheckoutCreatedEvent:
        return cls(
            checkout_session_id=checkout_session_id,
            supplier_id=supplier_id,
            line_items=line_items or [],
            total_amount=total_amount,
            currency=currency,
            idempotency_key=idempotency_key,
            org_id=org_id,
        )


class CheckoutCompletedEvent(BaseModel):
    """
    L7b: Emitted when a UCP checkout session reaches a terminal state.

    Links to an order and payment reference once the supplier confirms.
    """

    event_type: str = Field(
        default="commerce.checkout.completed",
        description="Event type identifier",
    )
    layer: str = Field(default="L7b", description="Layer identifier")
    checkout_session_id: str = Field(description="Checkout session that completed")
    order_id: Optional[str] = Field(default=None, description="Supplier-issued order identifier")
    payment_reference: Optional[str] = Field(
        default=None,
        description="Reference to the AP2 payment that settled this order",
    )
    status: str = Field(
        default="completed",
        description="Terminal status: completed | failed | cancelled",
    )
    org_id: str = Field(description="Organization that initiated the checkout")

    @classmethod
    def create(
        cls,
        checkout_session_id: str,
        org_id: str,
        *,
        order_id: str | None = None,
        payment_reference: str | None = None,
        status: str = "completed",
    ) -> CheckoutCompletedEvent:
        return cls(
            checkout_session_id=checkout_session_id,
            order_id=order_id,
            payment_reference=payment_reference,
            status=status,
            org_id=org_id,
        )


class OrderRefundedEvent(BaseModel):
    """
    L7b: Emitted when a UCP order is fully or partially refunded.
    """

    event_type: str = Field(
        default="commerce.order.refunded",
        description="Event type identifier",
    )
    layer: str = Field(default="L7b", description="Layer identifier")
    order_id: str = Field(description="Order being refunded")
    refund_amount: float = Field(description="Amount refunded")
    currency: str = Field(default="USD", description="ISO 4217 currency code")
    reason: Optional[str] = Field(default=None, description="Human-readable refund reason")
    org_id: str = Field(description="Organization that placed the original order")

    @classmethod
    def create(
        cls,
        order_id: str,
        refund_amount: float,
        org_id: str,
        *,
        currency: str = "USD",
        reason: str | None = None,
    ) -> OrderRefundedEvent:
        return cls(
            order_id=order_id,
            refund_amount=refund_amount,
            currency=currency,
            reason=reason,
            org_id=org_id,
        )


# ---------------------------------------------------------------------------
# L7c — A2UI (Agent-to-User Interface) Events
# ---------------------------------------------------------------------------


class SurfaceCreatedEvent(BaseModel):
    """
    L7c: Emitted when an A2UI surface is instantiated by an agent.

    A surface is a top-level UI context (e.g., checkout widget, confirmation dialog).
    """

    event_type: str = Field(
        default="commerce.ui.surface_created",
        description="Event type identifier",
    )
    layer: str = Field(default="L7c", description="Layer identifier")
    surface_id: str = Field(description="Unique surface instance identifier")
    root_component_id: Optional[str] = Field(
        default=None,
        description="Identifier of the root component in the surface tree",
    )
    component_count: int = Field(default=0, description="Number of components in the surface")
    org_id: str = Field(description="Organization that owns this surface")

    @classmethod
    def create(
        cls,
        surface_id: str,
        org_id: str,
        *,
        root_component_id: str | None = None,
        component_count: int = 0,
    ) -> SurfaceCreatedEvent:
        return cls(
            surface_id=surface_id,
            root_component_id=root_component_id,
            component_count=component_count,
            org_id=org_id,
        )


class UserActionTriggeredEvent(BaseModel):
    """
    L7c: Emitted when a user interacts with a component on an A2UI surface.

    The action context is hashed to avoid capturing PII; use context_hash
    for deduplication and replay correlation only.
    """

    event_type: str = Field(
        default="commerce.ui.user_action",
        description="Event type identifier",
    )
    layer: str = Field(default="L7c", description="Layer identifier")
    surface_id: str = Field(description="Surface on which the action occurred")
    action_name: str = Field(description="Semantic action name (e.g. confirm_purchase)")
    component_id: Optional[str] = Field(
        default=None,
        description="Component that received the interaction",
    )
    context_hash: Optional[str] = Field(
        default=None,
        description="sha256 of the full action context (never cleartext)",
    )
    org_id: str = Field(description="Organization that owns this surface")

    @classmethod
    def create(
        cls,
        surface_id: str,
        action_name: str,
        org_id: str,
        *,
        component_id: str | None = None,
        context_hash: str | None = None,
    ) -> UserActionTriggeredEvent:
        return cls(
            surface_id=surface_id,
            action_name=action_name,
            component_id=component_id,
            context_hash=context_hash,
            org_id=org_id,
        )
