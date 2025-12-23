"""
S.P.I.D.E.R. Consistent Hashing - Distributed Key Sharding
============================================================

Consistent Hashing solves the "hot rebalance" problem:
When a node joins/leaves, only K/n keys need to move (not all keys).

The Math:
  hash(key) mod 2^m → position on unit circle
  
  Virtual Nodes:
  - Each physical node gets V virtual positions
  - Spreads load more evenly
  - P(imbalance) decreases exponentially with V

Key Features:
- O(log n) lookup using sorted ring
- O(1) average key movement on rebalance
- Virtual nodes for load balancing
"""

import bisect
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


# =============================================================================
# CONSISTENT HASH RING
# =============================================================================

class ConsistentHashRing:
    """
    A consistent hash ring for distributing keys across nodes.
    
    Uses virtual nodes to ensure even distribution.
    
    Math:
    - Hash space: [0, 2^32)
    - Each node gets V virtual positions
    - Key goes to the first node clockwise from hash(key)
    
    Example:
        ring = ConsistentHashRing(replicas=100)
        ring.add_node("server-1")
        ring.add_node("server-2")
        ring.add_node("server-3")
        
        node = ring.get_node("my-key")  # Returns responsible server
    """
    
    def __init__(self, replicas: int = 100, hash_function: str = 'md5'):
        """
        Initialize the hash ring.
        
        Args:
            replicas: Number of virtual nodes per physical node
            hash_function: Hash function to use ('md5', 'sha1', 'sha256')
        """
        self.replicas = replicas
        self.hash_function = hash_function
        
        # Sorted list of (hash_value, node_id)
        self._ring: List[int] = []
        self._ring_to_node: Dict[int, str] = {}
        
        # Track physical nodes
        self._nodes: Set[str] = set()
        
        # Statistics
        self._stats = {
            'lookups': 0,
            'nodes_added': 0,
            'nodes_removed': 0,
        }
    
    def _hash(self, key: str) -> int:
        """
        Hash a key to a position on the ring.
        
        Returns a 32-bit integer.
        """
        if self.hash_function == 'md5':
            h = hashlib.md5(key.encode()).digest()
        elif self.hash_function == 'sha1':
            h = hashlib.sha1(key.encode()).digest()
        elif self.hash_function == 'sha256':
            h = hashlib.sha256(key.encode()).digest()
        else:
            raise ValueError(f"Unknown hash function: {self.hash_function}")
        
        # Take first 4 bytes as 32-bit integer
        return int.from_bytes(h[:4], byteorder='big')
    
    def add_node(self, node_id: str):
        """
        Add a node to the ring.
        
        Creates 'replicas' virtual nodes at different positions.
        
        Args:
            node_id: Unique identifier for the node
        """
        if node_id in self._nodes:
            return
        
        self._nodes.add(node_id)
        self._stats['nodes_added'] += 1
        
        # Add virtual nodes
        for i in range(self.replicas):
            virtual_key = f"{node_id}:{i}"
            hash_val = self._hash(virtual_key)
            
            # Insert in sorted order
            idx = bisect.bisect_left(self._ring, hash_val)
            self._ring.insert(idx, hash_val)
            self._ring_to_node[hash_val] = node_id
    
    def remove_node(self, node_id: str):
        """
        Remove a node from the ring.
        
        Only keys that were assigned to this node need to be reassigned.
        
        Args:
            node_id: The node to remove
        """
        if node_id not in self._nodes:
            return
        
        self._nodes.discard(node_id)
        self._stats['nodes_removed'] += 1
        
        # Remove all virtual nodes for this node
        for i in range(self.replicas):
            virtual_key = f"{node_id}:{i}"
            hash_val = self._hash(virtual_key)
            
            if hash_val in self._ring_to_node:
                del self._ring_to_node[hash_val]
                self._ring.remove(hash_val)
    
    def get_node(self, key: str) -> Optional[str]:
        """
        Get the node responsible for a key.
        
        Uses binary search to find the first node clockwise from the key's hash.
        
        Time Complexity: O(log n)
        
        Args:
            key: The key to look up
            
        Returns:
            Node ID, or None if ring is empty
        """
        if not self._ring:
            return None
        
        self._stats['lookups'] += 1
        
        hash_val = self._hash(key)
        
        # Find the first node clockwise (next highest hash)
        idx = bisect.bisect_right(self._ring, hash_val)
        
        # Wrap around to the beginning if we're past the end
        if idx >= len(self._ring):
            idx = 0
        
        return self._ring_to_node[self._ring[idx]]
    
    def get_nodes(self, key: str, count: int = 3) -> List[str]:
        """
        Get multiple responsible nodes for replication.
        
        Returns 'count' unique physical nodes, walking clockwise.
        
        Args:
            key: The key to look up
            count: Number of replicas desired
            
        Returns:
            List of unique node IDs
        """
        if not self._ring:
            return []
        
        hash_val = self._hash(key)
        idx = bisect.bisect_right(self._ring, hash_val)
        
        nodes = []
        seen = set()
        
        for _ in range(len(self._ring)):
            if idx >= len(self._ring):
                idx = 0
            
            node = self._ring_to_node[self._ring[idx]]
            
            if node not in seen:
                nodes.append(node)
                seen.add(node)
                
                if len(nodes) >= count:
                    break
            
            idx += 1
        
        return nodes
    
    def get_key_distribution(self, sample_keys: List[str]) -> Dict[str, int]:
        """
        Analyze the distribution of keys across nodes.
        
        Args:
            sample_keys: List of keys to analyze
            
        Returns:
            Dictionary of node_id -> key count
        """
        distribution: Dict[str, int] = {node: 0 for node in self._nodes}
        
        for key in sample_keys:
            node = self.get_node(key)
            if node:
                distribution[node] = distribution.get(node, 0) + 1
        
        return distribution
    
    @property
    def nodes(self) -> Set[str]:
        """Get all physical nodes in the ring."""
        return self._nodes.copy()
    
    @property
    def stats(self) -> Dict[str, int]:
        """Get ring statistics."""
        return self._stats.copy()
    
    def __len__(self) -> int:
        """Return number of physical nodes."""
        return len(self._nodes)
    
    def __repr__(self) -> str:
        return f"ConsistentHashRing(nodes={len(self._nodes)}, vnodes={len(self._ring)})"


# =============================================================================
# VIRTUAL NODE MANAGER
# =============================================================================

class VirtualNodeManager:
    """
    Manages virtual node assignment for load balancing.
    
    Dynamically adjusts virtual node count based on node capacity.
    """
    
    def __init__(self, base_replicas: int = 100):
        self.base_replicas = base_replicas
        self._node_weights: Dict[str, float] = {}
    
    def set_weight(self, node_id: str, weight: float):
        """
        Set the weight (capacity) of a node.
        
        Higher weight = more virtual nodes = more keys.
        
        Args:
            node_id: The node
            weight: Relative weight (1.0 = normal, 2.0 = double capacity)
        """
        self._node_weights[node_id] = max(0.1, weight)
    
    def get_replicas(self, node_id: str) -> int:
        """Get the number of virtual nodes for a node."""
        weight = self._node_weights.get(node_id, 1.0)
        return int(self.base_replicas * weight)


# =============================================================================
# RING MIGRATION
# =============================================================================

def calculate_migration(
    old_ring: ConsistentHashRing,
    new_ring: ConsistentHashRing,
    keys: List[str],
) -> Dict[str, Tuple[str, str]]:
    """
    Calculate which keys need to migrate between nodes.
    
    Args:
        old_ring: The ring before the change
        new_ring: The ring after the change
        keys: List of keys to check
        
    Returns:
        Dictionary of key -> (old_node, new_node) for keys that moved
    """
    migrations = {}
    
    for key in keys:
        old_node = old_ring.get_node(key)
        new_node = new_ring.get_node(key)
        
        if old_node != new_node:
            migrations[key] = (old_node, new_node)
    
    return migrations


# =============================================================================
# DEMO
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("💍 S.P.I.D.E.R. CONSISTENT HASHING - Demo")
    print("=" * 60)
    print()
    print("The Math: hash(key) mod 2^m → position on ring")
    print("Key benefit: Only K/n keys move when nodes change")
    print()
    
    # Create ring with 3 nodes
    ring = ConsistentHashRing(replicas=100)
    ring.add_node("server-1")
    ring.add_node("server-2")
    ring.add_node("server-3")
    
    print(f"Ring: {ring}")
    print()
    
    # Distribute some keys
    keys = [f"file_{i}.py" for i in range(100)]
    dist = ring.get_key_distribution(keys)
    
    print("Key Distribution (100 files):")
    for node, count in sorted(dist.items()):
        bar = "█" * (count // 2)
        print(f"  {node}: {count:3d} {bar}")
    
    print()
    
    # Simulate node failure
    print("Simulating node failure: server-2 goes down...")
    old_ring = ring
    new_ring = ConsistentHashRing(replicas=100)
    new_ring.add_node("server-1")
    new_ring.add_node("server-3")
    
    migrations = calculate_migration(old_ring, new_ring, keys)
    print(f"Keys that need to migrate: {len(migrations)}")
    print(f"Migration ratio: {len(migrations)/len(keys)*100:.1f}%")
    print()
    
    # New distribution
    new_dist = new_ring.get_key_distribution(keys)
    print("New Distribution:")
    for node, count in sorted(new_dist.items()):
        bar = "█" * (count // 2)
        print(f"  {node}: {count:3d} {bar}")
    
    print()
    print("=" * 60)
    print("✅ Consistent Hashing operational!")
