"""
S.P.I.D.E.R. Simulator Package
==============================

Provides simulation capabilities for fast MCTS exploration.

Components:
- PhantomShell: Virtual OS shell simulation (100x faster than Docker)
- VirtualFileSystem: In-memory file system
"""

from .os_phantom import PhantomShell, VirtualFileSystem, CommandExecutor

__all__ = ["PhantomShell", "VirtualFileSystem", "CommandExecutor"]
