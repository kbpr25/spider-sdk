"""
S.P.I.D.E.R. CRDT - Conflict-free Replicated Data Types
========================================================

The "Telepathic Grid" - enables multiple agents to edit the same codebase
concurrently without ANY coordination, locks, or merge conflicts.

Mathematical Foundation:
- LSEQ (Log-Structured Sequence) allocation
- Dense paths in infinite trees
- Commutative operations (order doesn't matter)

Key Insight:
  Instead of array indices [0, 1, 2], we use TREE PATHS:
  - 'A' at [1]
  - 'B' at [2]
  - Insert between → [1, 5]
  - No space? Go deeper → [1, 5, 3]
  
  Result: INFINITE space between any two characters.
"""

import bisect
import random
from dataclasses import dataclass, field
from functools import total_ordering
from typing import Any, List, Optional, Tuple, Dict


# =============================================================================
# IDENTIFIER - The Mathematical Core
# =============================================================================

@total_ordering
@dataclass(frozen=True)
class Identifier:
    """
    A unique, sortable identifier for a character in the codebase.
    
    Math: Lexicographical sorting of (positions, site_id, counter).
    
    Properties:
    - Unique across all agents (site_id + counter)
    - Totally ordered (allows sorting)
    - Immutable (frozen=True)
    - Dense (can always insert between any two)
    
    Example:
        id1 = Identifier([1], "A", 0)      # Position [1]
        id2 = Identifier([2], "A", 1)      # Position [2]
        id3 = Identifier([1, 5], "B", 0)   # Between id1 and id2
    """
    positions: Tuple[int, ...]  # The tree path (immutable tuple)
    site_id: str                 # The agent's unique ID
    counter: int = 0             # Logical clock for this agent
    
    def __post_init__(self):
        # Ensure positions is a tuple for hashability
        if isinstance(self.positions, list):
            object.__setattr__(self, 'positions', tuple(self.positions))
    
    def __lt__(self, other: 'Identifier') -> bool:
        """
        Lexicographical comparison.
        
        Compare position paths first, then site_id for tie-breaking.
        This ensures a TOTAL ORDER across all identifiers.
        """
        if not isinstance(other, Identifier):
            return NotImplemented
        
        # Compare position by position
        for p, q in zip(self.positions, other.positions):
            if p < q:
                return True
            if p > q:
                return False
        
        # If one path is a prefix of the other, shorter comes first
        if len(self.positions) < len(other.positions):
            return True
        if len(self.positions) > len(other.positions):
            return False
        
        # Positions are equal, use site_id as tie-breaker
        # This ensures deterministic ordering even for concurrent inserts
        return (self.site_id, self.counter) < (other.site_id, other.counter)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Identifier):
            return NotImplemented
        return (self.positions, self.site_id, self.counter) == \
               (other.positions, other.site_id, other.counter)
    
    def __hash__(self) -> int:
        return hash((self.positions, self.site_id, self.counter))
    
    def __repr__(self) -> str:
        path = '.'.join(map(str, self.positions))
        return f"Id({path}@{self.site_id}:{self.counter})"
    
    @classmethod
    def start_marker(cls, site_id: str) -> 'Identifier':
        """Create the start-of-document marker."""
        return cls(positions=(0,), site_id=site_id, counter=-1)
    
    @classmethod
    def end_marker(cls, site_id: str) -> 'Identifier':
        """Create the end-of-document marker."""
        return cls(positions=(2**31,), site_id=site_id, counter=-1)


# =============================================================================
# LSEQ ALLOCATOR - Infinite Space Generator
# =============================================================================

class LSEQAllocator:
    """
    Implements LSEQ (Log-Structured Sequence) allocation.
    
    The key innovation: we grow the tree EXPONENTIALLY at each depth,
    which means we NEVER run out of space between two identifiers.
    
    Math:
    - Depth 0: base = 16 values (0-15)
    - Depth 1: base = 32 values (0-31)
    - Depth 2: base = 64 values (0-63)
    - ...
    - Depth n: base = 16 * 2^n values
    
    This is called "variable-width" or "exponential" allocation.
    
    Strategies:
    - boundary+: allocate near the left boundary (for left-to-right typing)
    - boundary-: allocate near the right boundary (for right-to-left typing)
    - random: random allocation (for scattered edits)
    """
    
    BASE_BITS = 4  # Starting with 2^4 = 16 values at depth 0
    
    def __init__(self, site_id: str):
        self.site_id = site_id
        self.counter = 0
        self._strategy_cache: Dict[int, str] = {}  # depth -> strategy
    
    def _base_at_depth(self, depth: int) -> int:
        """Calculate the base (max value + 1) at a given depth."""
        return 2 ** (self.BASE_BITS + depth)
    
    def _get_strategy(self, depth: int) -> str:
        """
        Get allocation strategy for this depth.
        
        Uses a cached random choice per depth to maintain consistency
        while adding unpredictability across depths.
        """
        if depth not in self._strategy_cache:
            self._strategy_cache[depth] = random.choice(['boundary+', 'boundary-'])
        return self._strategy_cache[depth]
    
    def alloc(
        self,
        pos_before: Tuple[int, ...],
        pos_after: Tuple[int, ...],
    ) -> Tuple[int, ...]:
        """
        Allocate a new position strictly between pos_before and pos_after.
        
        The LSEQ magic: we can ALWAYS find space by going deeper in the tree.
        
        Args:
            pos_before: The identifier positions of the left neighbor
            pos_after: The identifier positions of the right neighbor
            
        Returns:
            New position tuple strictly between the two
        """
        result = []
        depth = 0
        
        while True:
            base = self._base_at_depth(depth)
            
            # Get values at current depth (default to boundaries)
            p = pos_before[depth] if depth < len(pos_before) else 0
            q = pos_after[depth] if depth < len(pos_after) else base
            
            # If we're still following pos_before's path, we must stay >= p
            if depth < len(pos_before) and result == list(pos_before[:depth]):
                pass  # p constraint applies
            else:
                p = 0  # No constraint from left
            
            # If we're still following pos_after's path, we must stay <= q
            if depth < len(pos_after) and result == list(pos_after[:depth]):
                pass  # q constraint applies
            else:
                q = base  # No constraint from right
            
            # How much space between p and q?
            interval = q - p - 1
            
            if interval > 0:
                # We have space! Allocate here.
                strategy = self._get_strategy(depth)
                
                if strategy == 'boundary+':
                    # Prefer left side (for sequential typing)
                    step = min(10, interval)
                    new_val = p + random.randint(1, step)
                else:
                    # Prefer right side
                    step = min(10, interval)
                    new_val = q - random.randint(1, step)
                
                result.append(new_val)
                return tuple(result)
            
            # No space at this level! Descend into the tree.
            # We copy the value from pos_before if available
            if depth < len(pos_before):
                result.append(pos_before[depth])
            else:
                result.append(0)
            
            depth += 1
            
            # Safety limit (should never hit this in practice)
            if depth > 64:
                raise RuntimeError("LSEQ allocation depth exceeded - this shouldn't happen!")
    
    def next_id(self, pos_before: Tuple[int, ...], pos_after: Tuple[int, ...]) -> Identifier:
        """
        Allocate a new Identifier between two positions.
        
        Returns:
            New Identifier with unique position, site_id, and counter
        """
        new_pos = self.alloc(pos_before, pos_after)
        self.counter += 1
        return Identifier(positions=new_pos, site_id=self.site_id, counter=self.counter)


# =============================================================================
# ATOM - A Character with its Identity
# =============================================================================

@dataclass
class Atom:
    """
    An atom is a character with its unique identifier.
    
    In CRDT terms, this is an "element" of the sequence.
    The identifier NEVER changes once created, even as the document evolves.
    """
    identifier: Identifier
    value: str
    tombstone: bool = False  # True if deleted (but we keep it for causality)
    
    def __repr__(self) -> str:
        mark = "†" if self.tombstone else ""
        return f"Atom({self.identifier}, '{self.value}'{mark})"


# =============================================================================
# CRDT STRING - The Telepathic Document
# =============================================================================

class CRDTString:
    """
    A string that allows concurrent edits from multiple agents
    without ANY coordination, locks, or merge conflicts.
    
    Properties:
    - Eventual Consistency: All replicas converge to the same state
    - Strong Convergence: Replicas that have seen the same operations are identical
    - Commutativity: Order of receiving operations doesn't matter
    - Idempotency: Applying the same operation twice has no effect
    
    Usage:
        # Two agents editing concurrently
        agent_a = CRDTString("A")
        agent_b = CRDTString("B")
        
        # Agent A types "Hi"
        op1 = agent_a.insert("H", 0)
        op2 = agent_a.insert("i", 1)
        
        # Agent B receives the operations (in any order!)
        agent_b.apply(op1)
        agent_b.apply(op2)
        
        # Both agents now have "Hi"
        assert agent_a.render() == agent_b.render() == "Hi"
    """
    
    def __init__(self, site_id: str):
        self.site_id = site_id
        self.allocator = LSEQAllocator(site_id)
        
        # The ordered list of atoms
        # We start with boundary markers that are never visible
        self._atoms: List[Atom] = [
            Atom(Identifier.start_marker(site_id), ""),
            Atom(Identifier.end_marker(site_id), ""),
        ]
        
        # Index for fast lookup by identifier
        self._id_to_index: Dict[Identifier, int] = {
            self._atoms[0].identifier: 0,
            self._atoms[1].identifier: 1,
        }
        
        # Statistics
        self._stats = {
            'local_inserts': 0,
            'remote_inserts': 0,
            'deletes': 0,
            'duplicates_ignored': 0,
        }
    
    def _find_insert_position(self, identifier: Identifier) -> int:
        """
        Find the correct position to insert an identifier using binary search.
        
        Returns the index where the new atom should be inserted.
        """
        left, right = 0, len(self._atoms)
        
        while left < right:
            mid = (left + right) // 2
            if self._atoms[mid].identifier < identifier:
                left = mid + 1
            else:
                right = mid
        
        return left
    
    def insert(self, char: str, index: int) -> Tuple[Identifier, str]:
        """
        Insert a character at a visual index.
        
        Args:
            char: The character to insert
            index: The visual position (0 = start of visible text)
            
        Returns:
            Tuple of (identifier, char) - the operation to broadcast
        """
        # Convert visual index to atom index (skip tombstones and start marker)
        visible_count = 0
        atom_index = 0
        
        for i, atom in enumerate(self._atoms):
            if i == 0:  # Start marker
                continue
            if visible_count == index:
                atom_index = i
                break
            if not atom.tombstone and atom.value:
                visible_count += 1
            atom_index = i + 1
        
        # Get the identifiers of neighbors
        id_before = self._atoms[atom_index - 1].identifier
        id_after = self._atoms[atom_index].identifier if atom_index < len(self._atoms) else self._atoms[-1].identifier
        
        # Allocate a new identifier between them
        new_id = self.allocator.next_id(id_before.positions, id_after.positions)
        
        # Create and insert the atom
        new_atom = Atom(new_id, char)
        self._atoms.insert(atom_index, new_atom)
        
        # Update index lookup (invalidate affected entries)
        self._rebuild_index()
        
        self._stats['local_inserts'] += 1
        
        return (new_id, char)
    
    def remote_insert(self, identifier: Identifier, char: str) -> bool:
        """
        Apply an insert operation from another agent.
        
        This is where the CRDT magic happens:
        - The identifier determines EXACTLY where to insert
        - No coordination needed
        - Idempotent (safe to apply multiple times)
        
        Args:
            identifier: The unique identifier from the remote agent
            char: The character that was inserted
            
        Returns:
            True if inserted, False if already existed (duplicate)
        """
        # Check for duplicates (idempotency)
        if identifier in self._id_to_index:
            self._stats['duplicates_ignored'] += 1
            return False
        
        # Find correct position using binary search
        insert_pos = self._find_insert_position(identifier)
        
        # Insert the atom
        new_atom = Atom(identifier, char)
        self._atoms.insert(insert_pos, new_atom)
        
        # Update index
        self._rebuild_index()
        
        self._stats['remote_inserts'] += 1
        return True
    
    def delete(self, index: int) -> Optional[Identifier]:
        """
        Delete a character at a visual index (tombstone it).
        
        In CRDTs, we don't actually remove atoms - we mark them as deleted.
        This preserves the causal history and ensures convergence.
        
        Args:
            index: The visual position to delete
            
        Returns:
            The identifier of the deleted atom (for broadcasting)
        """
        # Convert visual index to atom
        visible_count = 0
        
        for atom in self._atoms[1:]:  # Skip start marker
            if not atom.tombstone and atom.value:
                if visible_count == index:
                    atom.tombstone = True
                    self._stats['deletes'] += 1
                    return atom.identifier
                visible_count += 1
        
        return None
    
    def remote_delete(self, identifier: Identifier) -> bool:
        """
        Apply a delete operation from another agent.
        
        Args:
            identifier: The identifier of the atom to delete
            
        Returns:
            True if deleted, False if not found or already deleted
        """
        if identifier in self._id_to_index:
            idx = self._id_to_index[identifier]
            if not self._atoms[idx].tombstone:
                self._atoms[idx].tombstone = True
                self._stats['deletes'] += 1
                return True
        return False
    
    def _rebuild_index(self):
        """Rebuild the identifier -> index lookup."""
        self._id_to_index = {atom.identifier: i for i, atom in enumerate(self._atoms)}
    
    def render(self) -> str:
        """
        Render the visible string.
        
        Concatenates all non-tombstoned characters in order.
        """
        return "".join(
            atom.value 
            for atom in self._atoms 
            if not atom.tombstone and atom.value
        )
    
    def apply(self, operation: Tuple[str, Identifier, str]):
        """
        Apply an operation (insert or delete).
        
        Args:
            operation: Tuple of (op_type, identifier, value)
                       op_type is 'insert' or 'delete'
        """
        op_type, identifier, value = operation
        if op_type == 'insert':
            self.remote_insert(identifier, value)
        elif op_type == 'delete':
            self.remote_delete(identifier)
    
    def __len__(self) -> int:
        """Return the visible length."""
        return sum(1 for a in self._atoms if not a.tombstone and a.value)
    
    def __repr__(self) -> str:
        return f"CRDTString(site={self.site_id}, len={len(self)}, render='{self.render()}')"
    
    @property
    def stats(self) -> Dict[str, int]:
        """Get CRDT statistics."""
        return self._stats.copy()
    
    def debug_atoms(self) -> str:
        """Print detailed atom information for debugging."""
        lines = [f"CRDTString '{self.site_id}' - {len(self._atoms)} atoms:"]
        for i, atom in enumerate(self._atoms):
            mark = " [START]" if i == 0 else " [END]" if i == len(self._atoms) - 1 else ""
            tomb = " †" if atom.tombstone else ""
            lines.append(f"  {i}: {atom.identifier} = '{atom.value}'{tomb}{mark}")
        return "\n".join(lines)


# =============================================================================
# DEMO
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("🧠 CRDT STRING - Telepathic Editing Demo")
    print("=" * 60)
    
    # Create two agents
    alice = CRDTString("Alice")
    bob = CRDTString("Bob")
    
    # Alice types "Hello"
    print("\nAlice types 'Hello':")
    ops_alice = []
    for i, char in enumerate("Hello"):
        op = alice.insert(char, i)
        ops_alice.append(('insert', *op))
        print(f"  Insert '{char}' -> {op[0]}")
    
    print(f"  Alice sees: '{alice.render()}'")
    
    # Simulate network: Bob receives in REVERSE order (worst case)
    print("\nBob receives Alice's operations (in reverse order!):")
    for op in reversed(ops_alice):
        bob.apply(op)
        print(f"  Applied: {op}")
    
    print(f"  Bob sees: '{bob.render()}'")
    
    # Verify convergence
    print("\n" + "=" * 60)
    if alice.render() == bob.render():
        print(f"✅ CONVERGENCE: Both see '{alice.render()}'")
    else:
        print(f"❌ DIVERGENCE: Alice='{alice.render()}', Bob='{bob.render()}'")