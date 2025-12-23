"""
S.P.I.D.E.R. - Strategic Protocol for Intelligent Distributed Execution and Reasoning
======================================================================================

The Verification Layer That AI Agents Are Missing.

This SDK provides mathematical verification and distributed consensus
for building reliable, self-healing AI agents.

Quick Start:
    >>> from spider import SpiderEngine
    >>> engine = SpiderEngine()
    >>> engine.start()
    >>> result = engine.solve("fix the divide function")

Components:
    - SpiderEngine: Main orchestrator (The Weaver)
    - CodebaseIndexer: Bloom Filter for O(1) lookups
    - SymbolicVerifier: Z3-based formal verification
"""

__version__ = "1.0.0"
version_info = (1, 0, 0)
__author__ = "S.P.I.D.E.R. Team"
__license__ = "AGPL-3.0"

# =============================================================================
# PUBLIC API - Lazy imports to avoid heavy dependencies on startup
# =============================================================================

def __getattr__(name):
    """Lazy import handler for optional heavy modules."""
    if name == "SpiderEngine":
        from spider.main import SpiderEngine
        return SpiderEngine
    elif name == "BloomFilter":
        from spider.core.dsa.bloom import BloomFilter
        return BloomFilter
    elif name == "CodebaseIndexer":
        from spider.core.dsa.bloom import CodebaseIndexer
        return CodebaseIndexer
    elif name == "UltimateSolver":
        from spider.core.agent.ultimate import UltimateSolver
        return UltimateSolver
    elif name == "SolverMode":
        from spider.core.agent.ultimate import SolverMode
        return SolverMode
    raise AttributeError(f"module 'spider' has no attribute '{name}'")

__all__ = [
    "SpiderEngine",
    "BloomFilter", 
    "CodebaseIndexer",
    "UltimateSolver",
    "SolverMode",
]