"""
Semantic Kernel Filter Implementations

Provides STRATIX-instrumented filter classes for the SK filter API:
- LayerLensFunctionFilter: Function invocation pre/post hooks
- LayerLensPromptRenderFilter: Prompt template rendering hooks
- LayerLensAutoFunctionFilter: Auto-invoked function hooks
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from layerlens.instrument.adapters.frameworks.semantic_kernel.lifecycle import SemanticKernelAdapter

logger = logging.getLogger(__name__)


class LayerLensFunctionFilter:
    """
    Intercepts SK function invocations via the FunctionInvocationFilter API.

    Captures plugin name, function name, arguments, result, and latency.
    """

    def __init__(self, adapter: SemanticKernelAdapter) -> None:
        self._adapter = adapter
        self._contexts: dict[int, dict[str, Any]] = {}

    async def __call__(self, context: Any, next: Any = None) -> None:
        """SK filter callable interface: (context, next=...) -> Awaitable[None]."""
        return await self.on_function_invocation(context, next)

    async def on_function_invocation(
        self,
        context: Any,
        next_handler: Any = None,
    ) -> None:
        """Pre/post hook for function invocation."""
        plugin_name = self._extract_plugin_name(context)
        function_name = self._extract_function_name(context)
        arguments = self._extract_arguments(context)

        try:
            trace_ctx = self._adapter.on_function_start(
                plugin_name=plugin_name,
                function_name=function_name,
                arguments=arguments,
            )
        except Exception:
            logger.warning("Error in function start hook", exc_info=True)
            trace_ctx = {}

        error = None
        try:
            if next_handler:
                await next_handler(context)
        except Exception as exc:
            error = exc
            raise
        finally:
            try:
                result = self._extract_result(context)
                self._adapter.on_function_end(
                    context=trace_ctx,
                    result=result,
                    error=error,
                )
            except Exception:
                logger.warning("Error in function end hook", exc_info=True)

    def on_function_invocation_sync(
        self,
        plugin_name: str,
        function_name: str,
        arguments: dict[str, Any] | None = None,
        result: Any = None,
        error: Exception | None = None,
    ) -> None:
        """Synchronous hook for testing and non-async usage."""
        try:
            trace_ctx = self._adapter.on_function_start(
                plugin_name=plugin_name,
                function_name=function_name,
                arguments=arguments,
            )
            self._adapter.on_function_end(
                context=trace_ctx,
                result=result,
                error=error,
            )
        except Exception:
            logger.warning("Error in sync function hook", exc_info=True)

    @staticmethod
    def _extract_plugin_name(context: Any) -> str:
        """Extract plugin name from SK invocation context."""
        if hasattr(context, "function"):
            fn = context.function
            return getattr(fn, "plugin_name", "") or getattr(fn, "skill_name", "") or ""
        return getattr(context, "plugin_name", "") or ""

    @staticmethod
    def _extract_function_name(context: Any) -> str:
        if hasattr(context, "function"):
            fn = context.function
            return getattr(fn, "name", "") or ""
        return getattr(context, "function_name", "") or ""

    @staticmethod
    def _extract_arguments(context: Any) -> dict[str, Any] | None:
        args = getattr(context, "arguments", None)
        if args is None:
            return None
        if isinstance(args, dict):
            return args
        if hasattr(args, "items"):
            return dict(args.items())
        return None

    @staticmethod
    def _extract_result(context: Any) -> Any:
        return getattr(context, "result", None)


class LayerLensPromptRenderFilter:
    """
    Intercepts SK prompt rendering via the PromptRenderFilter API.

    Captures template text and rendered prompt string.
    """

    def __init__(self, adapter: SemanticKernelAdapter) -> None:
        self._adapter = adapter

    async def __call__(self, context: Any, next: Any = None) -> None:
        """SK filter callable interface."""
        return await self.on_prompt_render(context, next)

    async def on_prompt_render(
        self,
        context: Any,
        next_handler: Any = None,
    ) -> None:
        """Pre/post hook for prompt rendering."""
        function_name = getattr(context, "function_name", None) or ""
        template = getattr(context, "prompt_template", None)

        if next_handler:
            await next_handler(context)

        try:
            rendered = getattr(context, "rendered_prompt", None)
            self._adapter.on_prompt_render(
                template=str(template) if template else None,
                rendered_prompt=str(rendered) if rendered else None,
                function_name=function_name,
            )
        except Exception:
            logger.warning("Error in prompt render hook", exc_info=True)

    def on_prompt_render_sync(
        self,
        template: str | None = None,
        rendered_prompt: str | None = None,
        function_name: str | None = None,
    ) -> None:
        """Synchronous hook for testing."""
        try:
            self._adapter.on_prompt_render(
                template=template,
                rendered_prompt=rendered_prompt,
                function_name=function_name,
            )
        except Exception:
            logger.warning("Error in sync prompt render hook", exc_info=True)


class LayerLensAutoFunctionFilter:
    """
    Intercepts LLM-initiated (auto-invoked) function calls via
    the AutoFunctionInvocationFilter API.

    Marks all emitted events with auto_invoked=True.
    """

    def __init__(self, adapter: SemanticKernelAdapter) -> None:
        self._adapter = adapter

    async def __call__(self, context: Any, next: Any = None) -> None:
        """SK filter callable interface."""
        return await self.on_auto_function_invocation(context, next)

    async def on_auto_function_invocation(
        self,
        context: Any,
        next_handler: Any = None,
    ) -> None:
        """Pre/post hook for auto-invoked functions."""
        plugin_name = LayerLensFunctionFilter._extract_plugin_name(context)
        function_name = LayerLensFunctionFilter._extract_function_name(context)
        arguments = LayerLensFunctionFilter._extract_arguments(context)

        try:
            trace_ctx = self._adapter.on_function_start(
                plugin_name=plugin_name,
                function_name=function_name,
                arguments=arguments,
                auto_invoked=True,
            )
        except Exception:
            logger.warning("Error in auto function start hook", exc_info=True)
            trace_ctx = {}

        error = None
        try:
            if next_handler:
                await next_handler(context)
        except Exception as exc:
            error = exc
            raise
        finally:
            try:
                result = LayerLensFunctionFilter._extract_result(context)
                self._adapter.on_function_end(
                    context=trace_ctx,
                    result=result,
                    error=error,
                    auto_invoked=True,
                )
            except Exception:
                logger.warning("Error in auto function end hook", exc_info=True)

    def on_auto_function_invocation_sync(
        self,
        plugin_name: str,
        function_name: str,
        arguments: dict[str, Any] | None = None,
        result: Any = None,
        error: Exception | None = None,
    ) -> None:
        """Synchronous hook for testing."""
        try:
            trace_ctx = self._adapter.on_function_start(
                plugin_name=plugin_name,
                function_name=function_name,
                arguments=arguments,
                auto_invoked=True,
            )
            self._adapter.on_function_end(
                context=trace_ctx,
                result=result,
                error=error,
                auto_invoked=True,
            )
        except Exception:
            logger.warning("Error in sync auto function hook", exc_info=True)
