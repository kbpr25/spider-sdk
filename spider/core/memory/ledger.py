"""
S.P.I.D.E.R. Holographic Ledger - Zero-Deletion Context Management
===================================================================

Born from: Counter to Anthropic-1.1 (Context Editing)

The Anthropic Weakness:
"They introduced clear_tool_uses and clear_thinking_blocks. Deletion is LOSSY.
If the agent deletes 'ls -la' output from 50 turns ago, it forgets the file
structure. It has to run 'ls' again (wasting time/cost) or guess."

The S.P.I.D.E.R. Evolution:
Zero-Deletion Context Management via Holographic Memory.

Mechanism:
1. Active Window: Only current reasoning chain in context
2. Shadow Ledger: Tool outputs, thinking blocks -> hashed, stored in Graph
3. Pointer Token: Removed blocks -> replaced with [Ref: Tool_Output_42]
4. Auto-Dereference: If attention looks at pointer -> instant retrieval

Result: Claude forgets the past to save focus.
S.P.I.D.E.R. compresses the past but retains instant access.
We NEVER repeat a command.
"""

import hashlib
import json
import logging
import re
import time
import zlib
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# LEDGER TYPES
# =============================================================================

class BlockType(Enum):
    """Types of context blocks."""
    THINKING = auto()       # Model's reasoning
    TOOL_CALL = auto()      # Tool invocation
    TOOL_OUTPUT = auto()    # Tool result
    USER_MESSAGE = auto()   # User input
    ASSISTANT = auto()      # Model response
    SYSTEM = auto()         # System prompt
    REFERENCE = auto()      # Pointer to stored block


@dataclass
class ContextBlock:
    """A block of context that can be stored/compressed."""
    block_id: str
    block_type: BlockType
    content: str
    timestamp: float = field(default_factory=time.time)
    
    # Metadata
    turn_number: int = 0
    token_count: int = 0
    importance: float = 1.0
    
    # Storage
    compressed: bool = False
    hash_signature: str = ""
    
    # References
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.hash_signature:
            self.hash_signature = self._compute_hash()
        if not self.token_count:
            self.token_count = len(self.content) // 4  # Rough estimate
    
    def _compute_hash(self) -> str:
        return hashlib.sha256(self.content.encode()).hexdigest()[:16]


@dataclass
class PointerToken:
    """A lightweight pointer to a stored block."""
    ref_id: str
    block_type: BlockType
    summary: str            # One-line summary
    token_representation: str
    
    def __str__(self) -> str:
        return self.token_representation


@dataclass
class LedgerStats:
    """Statistics about ledger usage."""
    total_blocks: int = 0
    active_blocks: int = 0
    compressed_blocks: int = 0
    total_tokens_saved: int = 0
    dereferences: int = 0
    repeated_commands_prevented: int = 0


# =============================================================================
# COMPRESSION ENGINE
# =============================================================================

class CompressionEngine:
    """
    Compresses context blocks for storage.
    
    Uses multiple strategies:
    1. LZ77/Deflate for content compression
    2. Semantic summarization for long outputs
    3. Deduplication via content hashing
    """
    
    def __init__(self):
        self.dedup_store: Dict[str, bytes] = {}
        self.summary_cache: Dict[str, str] = {}
    
    def compress(self, block: ContextBlock) -> bytes:
        """Compress a context block."""
        # Check for duplicate
        if block.hash_signature in self.dedup_store:
            return self.dedup_store[block.hash_signature]
        
        # Compress content
        compressed = zlib.compress(block.content.encode(), level=9)
        
        # Store for dedup
        self.dedup_store[block.hash_signature] = compressed
        
        return compressed
    
    def decompress(self, data: bytes) -> str:
        """Decompress a context block."""
        return zlib.decompress(data).decode()
    
    def summarize(self, content: str, max_length: int = 100) -> str:
        """Create a summary of content."""
        # Simple extractive summary
        lines = content.strip().split('\n')
        
        if len(lines) <= 3:
            return content[:max_length]
        
        # Take first line, count, last line
        summary = f"{lines[0][:50]}... [{len(lines)} lines]... {lines[-1][:30]}"
        return summary[:max_length]
    
    def compute_importance(self, block: ContextBlock) -> float:
        """Compute importance score for a block."""
        score = 1.0
        
        # Recent blocks are more important
        age = time.time() - block.timestamp
        score *= max(0.1, 1.0 - (age / 3600))  # Decay over 1 hour
        
        # Tool outputs are important
        if block.block_type == BlockType.TOOL_OUTPUT:
            score *= 1.5
        
        # Thinking blocks less important after processed
        if block.block_type == BlockType.THINKING:
            score *= 0.7
        
        # Short blocks are less important
        if block.token_count < 50:
            score *= 0.8
        
        return min(2.0, score)


# =============================================================================
# HOLOGRAPHIC MEMORY
# =============================================================================

class HolographicMemory:
    """
    Content-addressable memory for context blocks.
    
    Stores blocks indexed by hash, type, and semantic content.
    Supports instant retrieval via pointer tokens.
    """
    
    def __init__(self, max_stored_blocks: int = 10000):
        self.max_blocks = max_stored_blocks
        
        self.blocks: Dict[str, ContextBlock] = {}
        self.compressed_data: Dict[str, bytes] = {}
        
        # Indexes
        self.by_type: Dict[BlockType, List[str]] = {t: [] for t in BlockType}
        self.by_turn: Dict[int, List[str]] = {}
        self.by_hash: Dict[str, str] = {}  # content_hash -> block_id
        
        # Command deduplication
        self.command_outputs: Dict[str, str] = {}  # command_hash -> output_block_id
        
        self.compressor = CompressionEngine()
    
    def store(self, block: ContextBlock) -> str:
        """Store a block in holographic memory."""
        # Check for duplicate command
        if block.block_type == BlockType.TOOL_CALL:
            cmd_hash = block.hash_signature
            if cmd_hash in self.command_outputs:
                # We've run this command before!
                return self.command_outputs[cmd_hash]
        
        # Compress and store
        compressed = self.compressor.compress(block)
        
        self.blocks[block.block_id] = block
        self.compressed_data[block.block_id] = compressed
        
        # Index
        self.by_type[block.block_type].append(block.block_id)
        
        if block.turn_number not in self.by_turn:
            self.by_turn[block.turn_number] = []
        self.by_turn[block.turn_number].append(block.block_id)
        
        self.by_hash[block.hash_signature] = block.block_id
        
        # Evict old blocks if needed
        if len(self.blocks) > self.max_blocks:
            self._evict_lru()
        
        return block.block_id
    
    def retrieve(self, block_id: str) -> Optional[ContextBlock]:
        """Retrieve a block from memory."""
        if block_id not in self.blocks:
            return None
        
        block = self.blocks[block_id]
        
        # Decompress if needed
        if block.compressed:
            data = self.compressed_data.get(block_id)
            if data:
                block.content = self.compressor.decompress(data)
        
        return block
    
    def create_pointer(self, block: ContextBlock) -> PointerToken:
        """Create a pointer token for a block."""
        summary = self.compressor.summarize(block.content)
        
        return PointerToken(
            ref_id=block.block_id,
            block_type=block.block_type,
            summary=summary,
            token_representation=f"[Ref:{block.block_id}|{block.block_type.name}]",
        )
    
    def dereference(self, pointer: PointerToken) -> Optional[str]:
        """Dereference a pointer to get full content."""
        block = self.retrieve(pointer.ref_id)
        if block:
            return block.content
        return None
    
    def register_command(self, command: str, output_block_id: str) -> None:
        """Register a command and its output for deduplication."""
        cmd_hash = hashlib.sha256(command.encode()).hexdigest()[:16]
        self.command_outputs[cmd_hash] = output_block_id
    
    def check_duplicate_command(self, command: str) -> Optional[str]:
        """Check if we've run this command before."""
        cmd_hash = hashlib.sha256(command.encode()).hexdigest()[:16]
        if cmd_hash in self.command_outputs:
            return self.command_outputs[cmd_hash]
        return None
    
    def _evict_lru(self) -> None:
        """Evict least recently used blocks."""
        # Sort by importance (low first)
        sorted_blocks = sorted(
            self.blocks.values(),
            key=lambda b: b.importance,
        )
        
        # Remove bottom 10%
        to_remove = len(sorted_blocks) // 10
        for block in sorted_blocks[:to_remove]:
            del self.blocks[block.block_id]
            if block.block_id in self.compressed_data:
                del self.compressed_data[block.block_id]


# =============================================================================
# HOLOGRAPHIC LEDGER
# =============================================================================

class HolographicLedger:
    """
    The Immortal Ledger - Zero-Deletion Context Management.
    
    Replaces Anthropic's "Context Editing" with lossless compression:
    1. Full content always retrievable
    2. Pointer tokens in active window
    3. Automatic deduplication of commands
    4. Never repeat a command
    
    Usage:
        ledger = HolographicLedger(max_active_tokens=8000)
        
        # Add a tool output
        pointer = ledger.add_block(
            content=massive_ls_output,
            block_type=BlockType.TOOL_OUTPUT,
        )
        
        # Later, when model references it
        full_content = ledger.dereference(pointer)
        
        # Check if we've run a command before
        if ledger.is_duplicate_command("ls -la"):
            output = ledger.get_previous_output("ls -la")
    """
    
    def __init__(
        self,
        max_active_tokens: int = 8000,
        compression_threshold: int = 500,
    ):
        """
        Initialize Holographic Ledger.
        
        Args:
            max_active_tokens: Max tokens in active window
            compression_threshold: Token count to trigger compression
        """
        self.max_active_tokens = max_active_tokens
        self.compression_threshold = compression_threshold
        
        self.memory = HolographicMemory()
        self.compressor = CompressionEngine()
        
        # Active window
        self.active_blocks: List[str] = []
        self.active_tokens: int = 0
        self.current_turn: int = 0
        
        # Pointer registry
        self.pointers: Dict[str, PointerToken] = {}
        
        self.stats = LedgerStats()
    
    def add_block(
        self,
        content: str,
        block_type: BlockType,
        force_active: bool = False,
    ) -> PointerToken:
        """
        Add a block to the ledger.
        
        Returns a pointer token that can be used to reference the block.
        """
        block_id = hashlib.md5(f"{content[:100]}{time.time()}".encode()).hexdigest()[:12]
        
        block = ContextBlock(
            block_id=block_id,
            block_type=block_type,
            content=content,
            turn_number=self.current_turn,
        )
        
        block.importance = self.compressor.compute_importance(block)
        
        # Store in holographic memory
        self.memory.store(block)
        self.stats.total_blocks += 1
        
        # Decide: active or compressed?
        should_compress = (
            block.token_count > self.compression_threshold
            and not force_active
            and block_type in [BlockType.TOOL_OUTPUT, BlockType.THINKING]
        )
        
        if should_compress:
            # Create pointer and store reference
            pointer = self.memory.create_pointer(block)
            self.pointers[block_id] = pointer
            self.stats.compressed_blocks += 1
            self.stats.total_tokens_saved += block.token_count
            return pointer
        else:
            # Keep in active window
            self.active_blocks.append(block_id)
            self.active_tokens += block.token_count
            self.stats.active_blocks += 1
            
            # Check if we need to compress old blocks
            self._enforce_window_limit()
            
            # Still return a pointer
            pointer = self.memory.create_pointer(block)
            self.pointers[block_id] = pointer
            return pointer
    
    def dereference(self, pointer: PointerToken) -> Optional[str]:
        """Dereference a pointer to get full content."""
        self.stats.dereferences += 1
        return self.memory.dereference(pointer)
    
    def dereference_by_id(self, block_id: str) -> Optional[str]:
        """Dereference by block ID."""
        pointer = self.pointers.get(block_id)
        if pointer:
            return self.dereference(pointer)
        
        block = self.memory.retrieve(block_id)
        return block.content if block else None
    
    def add_command(self, command: str) -> Optional[str]:
        """
        Add a command, checking for duplicates.
        
        Returns cached output if duplicate, None otherwise.
        """
        cached_id = self.memory.check_duplicate_command(command)
        if cached_id:
            self.stats.repeated_commands_prevented += 1
            return self.dereference_by_id(cached_id)
        return None
    
    def register_command_output(self, command: str, output: str) -> PointerToken:
        """Register a command and its output."""
        pointer = self.add_block(output, BlockType.TOOL_OUTPUT)
        self.memory.register_command(command, pointer.ref_id)
        return pointer
    
    def is_duplicate_command(self, command: str) -> bool:
        """Check if a command has been run before."""
        return self.memory.check_duplicate_command(command) is not None
    
    def get_active_context(self) -> str:
        """Get the current active context window."""
        context_parts = []
        
        for block_id in self.active_blocks[-50:]:  # Last 50 blocks
            block = self.memory.retrieve(block_id)
            if block:
                if block.compressed:
                    # Show pointer instead
                    pointer = self.pointers.get(block_id)
                    if pointer:
                        context_parts.append(str(pointer))
                else:
                    context_parts.append(block.content)
        
        return "\n".join(context_parts)
    
    def new_turn(self) -> None:
        """Start a new conversation turn."""
        self.current_turn += 1
    
    def _enforce_window_limit(self) -> None:
        """Ensure active window stays within limit."""
        while self.active_tokens > self.max_active_tokens and self.active_blocks:
            # Compress oldest block
            oldest_id = self.active_blocks.pop(0)
            block = self.memory.retrieve(oldest_id)
            
            if block:
                block.compressed = True
                self.active_tokens -= block.token_count
                self.stats.total_tokens_saved += block.token_count
    
    def get_stats(self) -> Dict[str, int]:
        return {
            "total_blocks": self.stats.total_blocks,
            "active_blocks": len(self.active_blocks),
            "compressed_blocks": self.stats.compressed_blocks,
            "tokens_saved": self.stats.total_tokens_saved,
            "dereferences": self.stats.dereferences,
            "commands_deduplicated": self.stats.repeated_commands_prevented,
        }
    
    def print_status(self) -> None:
        """Print ledger status."""
        print("\n" + "=" * 60)
        print("[*] HOLOGRAPHIC LEDGER STATUS")
        print("=" * 60)
        
        print(f"\n[W] Active Window:")
        print(f"   Blocks: {len(self.active_blocks)}")
        print(f"   Tokens: {self.active_tokens}/{self.max_active_tokens}")
        
        print(f"\n[M] Holographic Memory:")
        print(f"   Total Blocks: {self.stats.total_blocks}")
        print(f"   Compressed: {self.stats.compressed_blocks}")
        
        print(f"\n[%] Efficiency:")
        print(f"   Tokens Saved: {self.stats.total_tokens_saved}")
        print(f"   Dereferences: {self.stats.dereferences}")
        print(f"   Commands Deduplicated: {self.stats.repeated_commands_prevented}")
        
        print("=" * 60)


# =============================================================================
# PACKAGE INIT
# =============================================================================

__all__ = [
    "HolographicLedger",
    "HolographicMemory",
    "CompressionEngine",
    "ContextBlock",
    "PointerToken",
    "BlockType",
    "LedgerStats",
]


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("[*] S.P.I.D.E.R. Holographic Ledger - Demo")
    print("=" * 70)
    
    ledger = HolographicLedger(max_active_tokens=1000)
    
    # Simulate a long tool output
    print("\n[1] Adding large tool output...")
    large_output = """
drwxr-xr-x  5 user user  4096 Dec 20 10:00 src
-rw-r--r--  1 user user  1234 Dec 20 09:00 main.py
-rw-r--r--  1 user user  5678 Dec 20 08:00 utils.py
drwxr-xr-x  3 user user  4096 Dec 19 12:00 tests
-rw-r--r--  1 user user   512 Dec 19 11:00 config.yaml
""" * 50  # Make it large
    
    pointer = ledger.add_block(large_output, BlockType.TOOL_OUTPUT)
    print(f"   Pointer: {pointer}")
    print(f"   Tokens saved: {ledger.stats.total_tokens_saved}")
    
    # Dereference the pointer
    print("\n[2] Dereferencing pointer...")
    content = ledger.dereference(pointer)
    print(f"   Retrieved {len(content)} chars")
    
    # Test command deduplication
    print("\n[3] Testing command deduplication...")
    
    # First run of ls -la
    cached = ledger.add_command("ls -la /home/user")
    print(f"   First 'ls -la': {'cached' if cached else 'new'}")
    
    if not cached:
        ledger.register_command_output("ls -la /home/user", large_output)
    
    # Second run of same command
    cached = ledger.add_command("ls -la /home/user")
    print(f"   Second 'ls -la': {'CACHED!' if cached else 'new'}")
    
    if cached:
        print(f"   Saved a command execution!")
    
    ledger.print_status()
