"""
AutoGen Method Wrappers

Creates traced versions of ConversableAgent methods that intercept calls
and route events to the AutoGenAdapter lifecycle hooks.

All wrappers preserve the original method's behavior and handle adapter
exceptions silently to prevent tracing from breaking the application.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from layerlens.instrument.adapters.autogen.lifecycle import AutoGenAdapter

logger = logging.getLogger(__name__)


def create_traced_send(
    adapter: AutoGenAdapter,
    agent: Any,
    original_send: Callable,
) -> Callable:
    """
    Create a traced version of agent.send().

    Captures the message being sent and the recipient, then delegates
    to the original send method.
    """
    def traced_send(message: Any, recipient: Any, **kwargs: Any) -> Any:
        try:
            adapter.on_send(sender=agent, message=message, recipient=recipient)
        except Exception:
            logger.warning("Error in traced send pre-hook", exc_info=True)

        return original_send(message, recipient, **kwargs)

    traced_send._stratix_original = original_send
    return traced_send


def create_traced_receive(
    adapter: AutoGenAdapter,
    agent: Any,
    original_receive: Callable,
) -> Callable:
    """
    Create a traced version of agent.receive().

    Captures the received message and sender, then delegates
    to the original receive method.
    """
    def traced_receive(message: Any, sender: Any, **kwargs: Any) -> Any:
        try:
            adapter.on_receive(receiver=agent, message=message, sender=sender)
        except Exception:
            logger.warning("Error in traced receive pre-hook", exc_info=True)

        return original_receive(message, sender, **kwargs)

    traced_receive._stratix_original = original_receive
    return traced_receive


def create_traced_generate_reply(
    adapter: AutoGenAdapter,
    agent: Any,
    original_generate_reply: Callable,
) -> Callable:
    """
    Create a traced version of agent.generate_reply().

    Captures timing and the generated reply, then delegates to the
    original method.
    """
    def traced_generate_reply(messages: Any = None, sender: Any = None, **kwargs: Any) -> Any:
        start_ns = time.time_ns()
        error: Exception | None = None

        try:
            reply = original_generate_reply(messages=messages, sender=sender, **kwargs)
        except Exception as exc:
            error = exc
            reply = None
            raise
        finally:
            try:
                elapsed_ms = (time.time_ns() - start_ns) / 1_000_000
                if error is not None:
                    # Emit model.invoke with error information for failed calls
                    adapter.emit_dict_event("model.invoke", {
                        "framework": "autogen",
                        "agent": getattr(agent, "name", str(agent)),
                        "model": adapter._extract_model_name(agent),
                        "latency_ms": elapsed_ms,
                        "error": str(error),
                    })
                else:
                    adapter.on_generate_reply(
                        agent=agent,
                        messages=messages,
                        reply=reply,
                        latency_ms=elapsed_ms,
                    )
            except Exception:
                logger.warning("Error in traced generate_reply post-hook", exc_info=True)

        return reply

    traced_generate_reply._stratix_original = original_generate_reply
    return traced_generate_reply


def create_traced_execute_code(
    adapter: AutoGenAdapter,
    agent: Any,
    original_execute_code: Callable,
) -> Callable:
    """
    Create a traced version of agent.execute_code_blocks().

    Captures code blocks, execution result, and timing.
    """
    def traced_execute_code(code_blocks: Any, **kwargs: Any) -> Any:
        start_ns = time.time_ns()

        result = original_execute_code(code_blocks, **kwargs)

        try:
            elapsed_ms = (time.time_ns() - start_ns) / 1_000_000
            adapter.on_execute_code(
                agent=agent,
                code_blocks=code_blocks,
                result=result,
                latency_ms=elapsed_ms,
            )
        except Exception:
            logger.warning("Error in traced execute_code post-hook", exc_info=True)

        return result

    traced_execute_code._stratix_original = original_execute_code
    return traced_execute_code
