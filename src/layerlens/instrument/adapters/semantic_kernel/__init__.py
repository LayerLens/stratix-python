"""
STRATIX Semantic Kernel Adapter

Provides plugin invocation tracing, planner execution tracking,
and memory operation capture for Microsoft Semantic Kernel.
"""

from layerlens.instrument.adapters.semantic_kernel.lifecycle import (
    SemanticKernelAdapter,
)

ADAPTER_CLASS = SemanticKernelAdapter

__all__ = ["SemanticKernelAdapter", "ADAPTER_CLASS"]
