"""
S.P.I.D.E.R. Vector Clocks - Causal Ordering for Distributed Systems
=====================================================================

Vector Clocks enable answering the fundamental question of distributed systems:
"Did event A happen before event B?"

The Math:
  C(e) = max(C(e'), C(e)) + 1
  
  A vector clock is a mapping: NodeID → LogicalTime
  
  Happens-Before (→):
    - VC(a) < VC(b) means a → b (a happened before b)
    - VC(a) || VC(b) means concurrent (no causal relationship)
    
This is essential for:
- Detecting conflicts in CRDT merges
- Ordering distributed debugging events
- Ensuring causal consistency in replicated data
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import json


# =============================================================================
# VECTOR CLOCK
# =============================================================================

class VectorClock:
    """
    A Vector Clock for tracking causality in distributed systems.
    
    Each node maintains a vector of logical timestamps, one per node.
    When an event occurs:
    1. The local node increments its own counter
    2. When sending a message, attach the vector clock
    3. When receiving, merge clocks: take max of each component
    
    Example:
        # Three nodes: A, B, C
        vc_a = VectorClock("A")
        vc_b = VectorClock("B")
        
        # A does some work
        vc_a.tick()  # A: {A:1}
        
        # A sends to B
        vc_b.receive(vc_a)
        vc_b.tick()  # B: {A:1, B:1}
        
        # Now we can compare:
        vc_a.happened_before(vc_b)  # True: A's work happened before B's
    """
    
    def __init__(self, node_id: str):
        """
        Initialize a vector clock for a node.
        
        Args:
            node_id: Unique identifier for this node
        """
        self.node_id = node_id
        self._clock: Dict[str, int] = defaultdict(int)
    
    def tick(self) -> 'VectorClock':
        """
        Increment the local logical time.
        
        Called when a local event occurs (e.g., code change, proposal).
        
        Returns:
            self (for chaining)
        """
        self._clock[self.node_id] += 1
        return self
    
    def send(self) -> Dict[str, int]:
        """
        Get the clock state to send with a message.
        
        Returns:
            Copy of the current clock state
        """
        return dict(self._clock)
    
    def receive(self, other: 'VectorClock') -> 'VectorClock':
        """
        Merge another vector clock into this one.
        
        Formula: ∀k: new[k] = max(self[k], other[k])
        
        Then increment our own counter (for the receive event).
        
        Args:
            other: The vector clock received from another node
            
        Returns:
            self (for chaining)
        """
        for node_id, time in other._clock.items():
            self._clock[node_id] = max(self._clock[node_id], time)
        return self
    
    def merge_from_dict(self, clock_dict: Dict[str, int]) -> 'VectorClock':
        """Merge from a dictionary (e.g., from network message)."""
        for node_id, time in clock_dict.items():
            self._clock[node_id] = max(self._clock[node_id], time)
        return self
    
    def get(self, node_id: str) -> int:
        """Get the logical time for a specific node."""
        return self._clock.get(node_id, 0)
    
    def __getitem__(self, node_id: str) -> int:
        """Allow indexing: vc["A"]."""
        return self.get(node_id)
    
    def happened_before(self, other: 'VectorClock') -> bool:
        """
        Check if this clock happened before another.
        
        Formula: self < other iff:
          ∀k: self[k] ≤ other[k] AND ∃k: self[k] < other[k]
        
        Args:
            other: The vector clock to compare against
            
        Returns:
            True if self happened-before other
        """
        all_keys = set(self._clock.keys()) | set(other._clock.keys())
        
        at_least_one_less = False
        
        for key in all_keys:
            self_val = self._clock.get(key, 0)
            other_val = other._clock.get(key, 0)
            
            if self_val > other_val:
                return False  # Self is ahead on at least one component
            if self_val < other_val:
                at_least_one_less = True
        
        return at_least_one_less
    
    def concurrent_with(self, other: 'VectorClock') -> bool:
        """
        Check if two clocks are concurrent (no causal relationship).
        
        Two events are concurrent iff neither happened-before the other.
        This indicates a potential conflict in CRDT terms.
        
        Args:
            other: The vector clock to compare against
            
        Returns:
            True if the events are concurrent
        """
        return not self.happened_before(other) and not other.happened_before(self)
    
    def __lt__(self, other: 'VectorClock') -> bool:
        """Allow using < operator."""
        return self.happened_before(other)
    
    def __le__(self, other: 'VectorClock') -> bool:
        """Allow using <= operator."""
        return self.happened_before(other) or self == other
    
    def __eq__(self, other: object) -> bool:
        """Check if two clocks are exactly equal."""
        if not isinstance(other, VectorClock):
            return NotImplemented
        
        all_keys = set(self._clock.keys()) | set(other._clock.keys())
        return all(
            self._clock.get(k, 0) == other._clock.get(k, 0)
            for k in all_keys
        )
    
    def __repr__(self) -> str:
        items = ", ".join(f"{k}:{v}" for k, v in sorted(self._clock.items()))
        return f"VC({self.node_id})[{items}]"
    
    def to_json(self) -> str:
        """Serialize to JSON for network transmission."""
        return json.dumps({
            'node_id': self.node_id,
            'clock': dict(self._clock)
        })
    
    @classmethod
    def from_json(cls, json_str: str) -> 'VectorClock':
        """Deserialize from JSON."""
        data = json.loads(json_str)
        vc = cls(data['node_id'])
        vc._clock = defaultdict(int, data['clock'])
        return vc
    
    def copy(self) -> 'VectorClock':
        """Create a copy of this vector clock."""
        new_vc = VectorClock(self.node_id)
        new_vc._clock = defaultdict(int, self._clock)
        return new_vc


# =============================================================================
# VERSIONED VALUE (Value + Vector Clock)
# =============================================================================

@dataclass
class VersionedValue:
    """
    A value tagged with a vector clock for conflict detection.
    
    Used to track which version of a value is "newer" and detect conflicts.
    """
    value: Any
    clock: VectorClock
    
    def update(self, new_value: Any) -> 'VersionedValue':
        """Update the value and tick the clock."""
        self.clock.tick()
        self.value = new_value
        return self
    
    def conflicts_with(self, other: 'VersionedValue') -> bool:
        """Check if two versions conflict (are concurrent)."""
        return self.clock.concurrent_with(other.clock)
    
    def supersedes(self, other: 'VersionedValue') -> bool:
        """Check if this version supersedes (happened after) another."""
        return other.clock.happened_before(self.clock)


# =============================================================================
# CAUSAL HISTORY
# =============================================================================

class CausalHistory:
    """
    Track the causal history of events in a distributed system.
    
    Maintains a log of events with their vector clocks for debugging
    and conflict resolution.
    """
    
    def __init__(self):
        self._events: List[Tuple[VectorClock, str, Any]] = []
    
    def record(self, clock: VectorClock, event_type: str, data: Any = None):
        """Record an event with its causal timestamp."""
        self._events.append((clock.copy(), event_type, data))
    
    def happened_before(self, idx_a: int, idx_b: int) -> bool:
        """Check if event at idx_a happened before event at idx_b."""
        if idx_a >= len(self._events) or idx_b >= len(self._events):
            return False
        return self._events[idx_a][0].happened_before(self._events[idx_b][0])
    
    def find_concurrent(self) -> List[Tuple[int, int]]:
        """Find all pairs of concurrent events (potential conflicts)."""
        concurrent_pairs = []
        
        for i in range(len(self._events)):
            for j in range(i + 1, len(self._events)):
                if self._events[i][0].concurrent_with(self._events[j][0]):
                    concurrent_pairs.append((i, j))
        
        return concurrent_pairs
    
    def causal_order(self) -> List[int]:
        """
        Return events in causal order (topological sort).
        
        Events that happened-before others come first.
        Concurrent events maintain their original order.
        """
        n = len(self._events)
        if n == 0:
            return []
        
        # Build adjacency list
        edges = []
        for i in range(n):
            for j in range(n):
                if i != j and self._events[i][0].happened_before(self._events[j][0]):
                    edges.append((i, j))
        
        # Topological sort (Kahn's algorithm)
        in_degree = [0] * n
        for _, j in edges:
            in_degree[j] += 1
        
        queue = [i for i in range(n) if in_degree[i] == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            for i, j in edges:
                if i == node:
                    in_degree[j] -= 1
                    if in_degree[j] == 0:
                        queue.append(j)
        
        return result
    
    def __len__(self) -> int:
        return len(self._events)
    
    def __repr__(self) -> str:
        return f"CausalHistory({len(self._events)} events)"


# =============================================================================
# DEMO
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("🕐 S.P.I.D.E.R. VECTOR CLOCKS - Demo")
    print("=" * 60)
    print()
    print("The Math: C(e) = max(C(e'), C(e)) + 1")
    print()
    
    # Create three nodes
    alice = VectorClock("Alice")
    bob = VectorClock("Bob")
    charlie = VectorClock("Charlie")
    
    print("Scenario: Alice, Bob, Charlie collaborate on code")
    print("-" * 60)
    
    # Alice makes a change
    alice.tick()
    print(f"1. Alice edits code: {alice}")
    
    # Alice sends to Bob
    bob.receive(alice)
    bob.tick()
    print(f"2. Bob receives and edits: {bob}")
    
    # Concurrently, Alice makes another change
    alice.tick()
    print(f"3. Alice makes another edit: {alice}")
    
    # Charlie receives from Alice
    charlie.receive(alice)
    charlie.tick()
    print(f"4. Charlie receives from Alice: {charlie}")
    
    # Now check causal relationships
    print()
    print("Causal Relationships:")
    print("-" * 60)
    
    # Alice's first version vs Bob's
    print(f"Alice < Bob? {alice.happened_before(bob)}")
    print(f"Bob < Alice? {bob.happened_before(alice)}")
    print(f"Alice || Bob (concurrent)? {alice.concurrent_with(bob)}")
    
    print()
    print("=" * 60)
    print("✅ Vector Clocks operational!")
