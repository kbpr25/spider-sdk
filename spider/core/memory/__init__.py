"""
S.P.I.D.E.R. Memory Package
===========================

Provides intelligent memory management for the SDK.

Components:
- RetroContextEngine: JIT rolling memory (RETRO-based)
"""

from .retro_engine import (
    RetroContextEngine,
    RetroStreamWrapper,
    ContextStore,
    ContextChunk,
    RetrievalEvent,
    TriggerType,
    LogicalBlockDetector,
)

__all__ = [
    "RetroContextEngine",
    "RetroStreamWrapper",
    "ContextStore",
    "ContextChunk",
    "RetrievalEvent",
    "TriggerType",
    "LogicalBlockDetector",
]
