"""Error injection for simulated traces.

Errors are injected per-span with independent probability.
An error on span N does not prevent span N+1 from generating.
"""

from __future__ import annotations

import random
from typing import Any

from ..config import ErrorConfig
from ..span_model import SimulatedSpan, SimulatedTrace, SpanType
from .auth_failure import AuthFailureInjector
from .base import BaseErrorInjector
from .content_filter import ContentFilterInjector
from .rate_limit import RateLimitInjector
from .server_error import ServerErrorInjector
from .timeout import TimeoutInjector

_INJECTORS: dict[str, type[BaseErrorInjector]] = {
    "rate_limit": RateLimitInjector,
    "timeout": TimeoutInjector,
    "auth_failure": AuthFailureInjector,
    "content_filter": ContentFilterInjector,
    "server_error": ServerErrorInjector,
}


def inject_errors(
    trace: SimulatedTrace,
    config: ErrorConfig,
    seed: int | None = None,
) -> SimulatedTrace:
    """Inject errors into a trace based on config probabilities.

    Each LLM span is independently evaluated for error injection.
    """
    if not config.enabled:
        return trace

    rng = random.Random(seed)

    error_probabilities = [
        ("rate_limit", config.rate_limit_probability),
        ("timeout", config.timeout_probability),
        ("auth_failure", config.auth_failure_probability),
        ("content_filter", config.content_filter_probability),
        ("server_error", config.server_error_probability),
    ]

    for span in trace.spans:
        if span.span_type != SpanType.LLM:
            continue

        for error_name, probability in error_probabilities:
            if probability > 0 and rng.random() < probability:
                injector_cls = _INJECTORS.get(error_name)
                if injector_cls:
                    if error_name == "server_error":
                        injector = injector_cls(seed=rng.randint(0, 2**31))
                    else:
                        injector = injector_cls()
                    injector.inject(span)
                    break  # Only one error per span

    return trace


__all__ = [
    "BaseErrorInjector",
    "RateLimitInjector",
    "TimeoutInjector",
    "AuthFailureInjector",
    "ContentFilterInjector",
    "ServerErrorInjector",
    "inject_errors",
]
