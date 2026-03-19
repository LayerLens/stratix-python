"""Output formatters for 3 wire formats."""

from .base import BaseOutputFormatter

_OUTPUT_REGISTRY: dict[str, type[BaseOutputFormatter]] = {}


def register_output(name: str, formatter_class: type[BaseOutputFormatter]) -> None:
    """Register an output formatter."""
    _OUTPUT_REGISTRY[name] = formatter_class


def get_output_formatter(name: str) -> BaseOutputFormatter:
    """Get an output formatter instance by name."""
    if not _OUTPUT_REGISTRY:
        _load_outputs()
    cls = _OUTPUT_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown output format: {name}. Available: {list(_OUTPUT_REGISTRY.keys())}"
        )
    return cls()


def list_outputs() -> list[str]:
    """List all registered output format names."""
    if not _OUTPUT_REGISTRY:
        _load_outputs()
    return sorted(_OUTPUT_REGISTRY.keys())


def _load_outputs() -> None:
    """Lazy-load all output formatters."""
    from .stratix_native import STRATIXNativeFormatter
    from .langfuse_json import LangfuseJSONFormatter
    from .otlp_json import OTLPJSONFormatter

    register_output("otlp_json", OTLPJSONFormatter)
    register_output("langfuse_json", LangfuseJSONFormatter)
    register_output("stratix_native", STRATIXNativeFormatter)


__all__ = [
    "BaseOutputFormatter",
    "get_output_formatter",
    "list_outputs",
    "register_output",
]
