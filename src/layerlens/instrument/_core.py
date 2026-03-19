"""
STRATIX Python SDK Core

The main STRATIX class that provides SDK initialization and configuration.

From Step 4 specification:
- SDK initialization MUST be a one-liner for most users
- Initialization MUST:
  1. Load or reference the active Step 2 policy (policy_id, version, hash)
  2. Initialize OTel exporter/collector settings
  3. Establish a local sequence_id allocator per agent
  4. Establish a local vector clock participant id
  5. Register framework-specific state adapters (if available)
- Initialization MUST bind a tracer context (thread-local / async-local)
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from layerlens.instrument.schema.attestation import HashChainBuilder
from layerlens.instrument.schema.event import STRATIXEvent, STRATIXEventBuilder
from layerlens.instrument.schema.events import (
    AgentCodeEvent,
    AgentInputEvent,
    AgentOutputEvent,
    EnvironmentConfigEvent,
    PolicyViolationEvent,
    ViolationType,
)
from layerlens.instrument.schema.events.l4_environment import EnvironmentType
from layerlens.instrument.schema.privacy import PrivacyLevel
from layerlens.instrument._context import STRATIXContext, context_scope, set_current_context

if TYPE_CHECKING:
    from layerlens.instrument._state import StateAdapter


T = TypeVar("T")


class STRATIX:
    """
    Main STRATIX SDK class.

    Provides:
    - One-liner initialization
    - Decorator-based instrumentation
    - Context propagation
    - Event emission
    - Policy enforcement

    Usage:
        stratix = STRATIX(
            policy_ref="stratix-policy-cs-v1@1.0.0",
            agent_id="support_agent",
            framework="langgraph",
            exporter="otel",
            endpoint="otel-collector:4317"
        )

        @stratix.trace_tool(name="lookup_order", version="1.0.0")
        def lookup_order(order_id: str) -> dict:
            ...
    """

    def __init__(
        self,
        policy_ref: str,
        agent_id: str,
        framework: str | None = None,
        exporter: str = "otel",
        endpoint: str | None = None,
        signing_key_id: str | None = None,
        privacy_default: PrivacyLevel = PrivacyLevel.CLEARTEXT,
        state_adapter: "StateAdapter | None" = None,
        auto_emit_code: bool = True,
        auto_emit_config: bool = True,
    ):
        """
        Initialize the STRATIX SDK.

        Args:
            policy_ref: Policy reference (e.g., "stratix-policy-cs-v1@1.0.0")
            agent_id: Unique identifier for this agent
            framework: Agent framework name (langgraph, langchain, etc.)
            exporter: Exporter type ("otel", "datadog", "splunk")
            endpoint: Exporter endpoint URL
            signing_key_id: Signing key identifier for attestation
            privacy_default: Default privacy level for events
            state_adapter: Optional framework-specific state adapter
            auto_emit_code: Automatically emit agent.code event on start
            auto_emit_config: Automatically emit environment.config on start
        """
        self._policy_ref = policy_ref
        self._agent_id = agent_id
        self._framework = framework
        self._exporter = exporter
        self._endpoint = endpoint
        self._signing_key_id = signing_key_id
        self._privacy_default = privacy_default
        self._state_adapter = state_adapter
        self._auto_emit_code = auto_emit_code
        self._auto_emit_config = auto_emit_config

        # Parse policy reference
        self._policy_id, self._policy_version = self._parse_policy_ref(policy_ref)

        # Initialize hash chain builder
        self._hash_chain = HashChainBuilder(signing_key_id=signing_key_id)

        # Track if we've violated policy (stops further hashing)
        self._policy_violated = False

        # Current context
        self._root_context: STRATIXContext | None = None

        # Event buffer (for batching if needed)
        self._event_buffer: list[STRATIXEvent] = []

        # Exporter (lazy initialized)
        self._exporter_instance: Any = None

    @staticmethod
    def _parse_policy_ref(policy_ref: str) -> tuple[str, str]:
        """Parse policy reference into ID and version."""
        if "@" in policy_ref:
            parts = policy_ref.rsplit("@", 1)
            return parts[0], parts[1]
        return policy_ref, "latest"

    @property
    def policy_ref(self) -> str:
        """Get the policy reference."""
        return self._policy_ref

    @property
    def policy_id(self) -> str:
        """Get the policy ID."""
        return self._policy_id

    @property
    def policy_version(self) -> str:
        """Get the policy version."""
        return self._policy_version

    @property
    def agent_id(self) -> str:
        """Get the agent ID."""
        return self._agent_id

    @property
    def framework(self) -> str | None:
        """Get the framework name."""
        return self._framework

    @property
    def is_policy_violated(self) -> bool:
        """Check if policy has been violated."""
        return self._policy_violated

    def start_trial(
        self,
        evaluation_id: str | None = None,
        trial_id: str | None = None,
        trace_id: str | None = None,
    ) -> STRATIXContext:
        """
        Start a new trial.

        This creates a new context and optionally emits initial events.

        Args:
            evaluation_id: Evaluation ID (generated if not provided)
            trial_id: Trial ID (generated if not provided)
            trace_id: Trace ID (generated if not provided)

        Returns:
            The trial context
        """
        # Create root context
        ctx = STRATIXContext(
            stratix=self,
            evaluation_id=evaluation_id,
            trial_id=trial_id,
            trace_id=trace_id,
        )
        self._root_context = ctx

        # Set as current context
        set_current_context(ctx)

        # Start root span
        ctx.start_span()

        # Auto-emit initial events
        if self._auto_emit_code:
            self._emit_agent_code(ctx)
        if self._auto_emit_config:
            self._emit_environment_config(ctx)

        return ctx

    def _emit_agent_code(self, ctx: STRATIXContext) -> None:
        """Emit agent.code event for the trial."""
        # In a real implementation, this would get actual repo/commit info
        event_payload = AgentCodeEvent.create(
            repo="unknown",
            commit="unknown",
            artifact_hash="sha256:" + "0" * 64,
            config_hash="sha256:" + "0" * 64,
        )
        self._emit_event(ctx, event_payload)

    def _emit_environment_config(self, ctx: STRATIXContext) -> None:
        """Emit environment.config event for the trial."""
        event_payload = EnvironmentConfigEvent.create(
            env_type=EnvironmentType.CLOUD,
            attributes={
                "framework": self._framework,
                "policy_ref": self._policy_ref,
            },
        )
        self._emit_event(ctx, event_payload)

    def _emit_event(
        self,
        ctx: STRATIXContext,
        payload: Any,
        privacy_level: PrivacyLevel | None = None,
    ) -> STRATIXEvent | None:
        """
        Emit an event.

        Args:
            ctx: The current context
            payload: Event payload
            privacy_level: Privacy level (uses default if not provided)

        Returns:
            The created event, or None if policy violated
        """
        if self._policy_violated:
            return None

        # Get sequence ID and update vector clock
        seq_id = ctx.next_sequence_id()
        vc = ctx.increment_vector_clock()

        # Create the event
        event = STRATIXEvent.create(
            payload=payload,
            agent_id=self._agent_id,
            evaluation_id=ctx.evaluation_id,
            trial_id=ctx.trial_id,
            trace_id=ctx.trace_id,
            parent_span_id=ctx.parent_span_id,
            sequence_id=seq_id,
            vector_clock=vc,
            privacy_level=privacy_level or self._privacy_default,
            previous_hash=self._hash_chain.last_hash,
            signing_key_id=self._signing_key_id,
        )

        # Add to hash chain
        try:
            self._hash_chain.add_event(event.to_dict())
        except RuntimeError:
            # Chain terminated due to violation
            return None

        # Buffer the event
        self._event_buffer.append(event)

        # Export if configured
        self._export_event(event)

        return event

    def _export_event(self, event: STRATIXEvent) -> None:
        """Export an event to the configured exporter."""
        # Lazy initialize exporter
        if self._exporter_instance is None and self._endpoint:
            self._initialize_exporter()

        if self._exporter_instance is not None:
            try:
                self._exporter_instance.export(event)
            except Exception:
                # Log but don't fail on export errors
                pass

    def _initialize_exporter(self) -> None:
        """Initialize the exporter based on configuration."""
        if self._exporter == "otel" and self._endpoint:
            from layerlens.instrument.exporters._otel import OTelExporter
            self._exporter_instance = OTelExporter(endpoint=self._endpoint)

    def emit_policy_violation(
        self,
        violation_type: ViolationType,
        root_cause: str,
        remediation: str,
        failed_layer: str | None = None,
    ) -> None:
        """
        Emit a policy violation and terminate the hash chain.

        NORMATIVE: Evaluation terminates immediately; no further hashing occurs.

        Args:
            violation_type: Type of violation
            root_cause: Root cause description
            remediation: Remediation suggestion
            failed_layer: Layer where violation occurred
        """
        ctx = self._root_context
        if ctx is None:
            return

        # Emit the violation event (before terminating chain)
        event_payload = PolicyViolationEvent.create(
            violation_type=violation_type,
            root_cause=root_cause,
            remediation=remediation,
            failed_layer=failed_layer,
            failed_sequence_id=ctx.sequence_id,
        )

        # This will be the last event
        self._emit_event(ctx, event_payload)

        # Terminate the hash chain
        self._hash_chain.terminate("policy_violation")
        self._policy_violated = True

    def end_trial(self) -> dict[str, Any] | None:
        """
        End the current trial.

        Returns:
            Trial summary including attestation, or None if violated
        """
        if self._policy_violated:
            return {
                "status": "non-attestable",
                "reason": "policy_violation",
                "events": len(self._event_buffer),
            }

        try:
            trial_attestation = self._hash_chain.finalize_trial()
            return {
                "status": "attestable",
                "trial_hash": trial_attestation.hash,
                "events": len(self._event_buffer),
                "chain_verified": self._hash_chain.verify_chain_integrity(),
            }
        except RuntimeError:
            return {
                "status": "non-attestable",
                "reason": "chain_terminated",
                "events": len(self._event_buffer),
            }

    # Decorator methods
    def trace_tool(
        self,
        name: str,
        version: str = "unavailable",
        integration: str = "library",
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """
        Decorator for tool/action instrumentation.

        From Step 4 specification:
        - Decorators provide automatic capture of input/output
        - Latency measurement
        - Exception capture
        - Deterministic sequence boundaries
        - Automatic privacy enforcement

        Args:
            name: Tool name
            version: Tool version
            integration: Integration type (library, service, agent)

        Returns:
            Decorator function
        """
        from layerlens.instrument._decorators import trace_tool
        return trace_tool(self, name, version, integration)

    def trace_model(
        self,
        provider: str,
        name: str,
        version: str = "unavailable",
    ) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """
        Decorator for model invocation instrumentation.

        Args:
            provider: Model provider (openai, anthropic, etc.)
            name: Model name
            version: Model version

        Returns:
            Decorator function
        """
        from layerlens.instrument._decorators import trace_model
        return trace_model(self, provider, name, version)

    # Explicit emit methods
    def emit(self, payload: Any, privacy_level: PrivacyLevel | None = None) -> STRATIXEvent | None:
        """
        Explicitly emit an event.

        This is the escape hatch for cases not covered by decorators.

        Args:
            payload: Event payload
            privacy_level: Privacy level

        Returns:
            The created event, or None if policy violated
        """
        from layerlens.instrument._context import get_current_context
        ctx = get_current_context()
        if ctx is None:
            raise RuntimeError("No active STRATIX context. Call start_trial() first.")
        return self._emit_event(ctx, payload, privacy_level)

    def emit_input(self, message: str, role: str = "human") -> STRATIXEvent | None:
        """Emit an agent input event."""
        from layerlens.instrument.schema.events.l1_io import MessageRole
        role_enum = MessageRole(role)
        payload = AgentInputEvent.create(message=message, role=role_enum)
        return self.emit(payload)

    def emit_output(self, message: str) -> STRATIXEvent | None:
        """Emit an agent output event."""
        payload = AgentOutputEvent.create(message=message)
        return self.emit(payload)

    def context(self) -> context_scope:
        """
        Get a context manager for the current context.

        Usage:
            with stratix.context() as ctx:
                # Do work in this context
        """
        if self._root_context is None:
            raise RuntimeError("No active trial. Call start_trial() first.")
        return context_scope(self._root_context)

    def get_events(self) -> list[STRATIXEvent]:
        """Get all events emitted in this session."""
        return list(self._event_buffer)

    # ---- Feedback convenience ----

    def submit_feedback(
        self,
        trace_id: str,
        thumbs: str | None = None,
        rating: float | None = None,
        comment: str | None = None,
        span_id: str | None = None,
        user_id: str | None = None,
        tags: list[str] | None = None,
    ) -> Any:
        """
        Submit explicit feedback for a trace.

        Convenience wrapper around :class:`~stratix.feedback.FeedbackCollector`.

        Args:
            trace_id: The trace receiving feedback.
            thumbs: Thumbs rating (``"up"`` or ``"down"``).
            rating: Numeric rating (0.0-1.0 or 1-5 scale).
            comment: Free-text feedback.
            span_id: Optional span-level targeting.
            user_id: Who provided the feedback.
            tags: Categorical tags.

        Returns:
            The created ExplicitFeedbackEvent.

        Raises:
            NotImplementedError: This method requires the server-side feedback
                collector. Use ``layerlens.Stratix`` API client methods instead.
        """
        raise NotImplementedError(
            "submit_feedback() requires the server-side feedback collector. "
            "Use the layerlens.Stratix API client to submit feedback instead."
        )

    # ---- Replay convenience ----

    async def replay(
        self,
        trace_id: str,
        store: Any,
        input_overrides: dict[str, Any] | None = None,
        model_override: str | None = None,
        config_overrides: dict[str, Any] | None = None,
    ) -> Any:
        """
        Replay a previously recorded trace.

        Convenience wrapper around :class:`~stratix.replay.ReplayController`.

        Args:
            trace_id: ID of the original trace to replay.
            store: A :class:`~stratix.replay.ReplayStore` instance containing
                   the original trace. Required — the store must already
                   contain the trace to replay.
            input_overrides: Input values to override.
            model_override: Replace the original model (P1).
            config_overrides: Framework config overrides (P1).

        Returns:
            The ReplayResult.

        Raises:
            NotImplementedError: This method requires the server-side replay
                controller. Use ``layerlens.Stratix`` API client methods instead.
        """
        raise NotImplementedError(
            "replay() requires the server-side replay controller. "
            "Use the layerlens.Stratix API client to replay traces instead."
        )
