"""
S.P.I.D.E.R. Gossip Protocol - Epidemic Information Dissemination
==================================================================

Gossip protocols spread information like a virus through a network.
Highly resilient - works even with node failures and network partitions.

The Math:
  After k rounds, P(node has info) = 1 - (1 - 1/n)^k
  
  For k = n*ln(n), probability approaches 1 exponentially.
  
Key Properties:
- Decentralized: No leader required
- Scalable: O(log n) rounds to reach all nodes
- Resilient: Tolerates partial network failures
- Eventually Consistent: All nodes converge

Used For:
- Cluster membership
- Health monitoring
- Metadata propagation
- Anti-entropy (replica sync)
"""

import hashlib
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import threading


# =============================================================================
# MESSAGE TYPES
# =============================================================================

class MessageType(Enum):
    """Types of gossip messages."""
    SYNC = "sync"           # Full state sync
    DELTA = "delta"         # Only changes
    MEMBERSHIP = "membership"  # Cluster membership
    HEARTBEAT = "heartbeat"  # I'm alive
    RUMOR = "rumor"          # One-time propagation


@dataclass
class GossipMessage:
    """A message in the gossip protocol."""
    msg_type: MessageType
    sender_id: str
    payload: Any
    version: int = 0
    timestamp: float = field(default_factory=time.time)
    ttl: int = 10  # Time-to-live (rounds)
    
    def key(self) -> str:
        """Unique key for deduplication."""
        return f"{self.sender_id}:{self.msg_type.value}:{self.version}"


# =============================================================================
# NODE STATE
# =============================================================================

@dataclass
class NodeState:
    """State of a node in the cluster."""
    node_id: str
    address: str = ""
    status: str = "alive"
    metadata: Dict[str, Any] = field(default_factory=dict)
    heartbeat_count: int = 0
    last_seen: float = field(default_factory=time.time)
    
    def is_alive(self, timeout: float = 30.0) -> bool:
        """Check if node is considered alive."""
        return time.time() - self.last_seen < timeout


# =============================================================================
# GOSSIP ENGINE
# =============================================================================

class GossipEngine:
    """
    Epidemic-style gossip protocol for information dissemination.
    
    Each round:
    1. Pick random peers
    2. Exchange state (push/pull)
    3. Merge received state
    
    Math: After k rounds, P(all nodes have info) approaches 1 - e^(-fanout*k/n)
    
    Example:
        engine = GossipEngine(node_id="server-1", fanout=3)
        engine.add_peer("server-2", "192.168.1.2:7000")
        engine.add_peer("server-3", "192.168.1.3:7000")
        
        # Spread a rumor
        engine.broadcast({"event": "new_leader", "leader": "server-1"})
        
        # On receive (from network)
        engine.receive(message_from_network)
    """
    
    def __init__(
        self,
        node_id: str,
        fanout: int = 3,
        gossip_interval: float = 1.0,
        dead_timeout: float = 30.0,
    ):
        """
        Initialize the gossip engine.
        
        Args:
            node_id: Unique identifier for this node
            fanout: Number of peers to gossip to each round
            gossip_interval: Seconds between gossip rounds
            dead_timeout: Seconds before marking a node as dead
        """
        self.node_id = node_id
        self.fanout = fanout
        self.gossip_interval = gossip_interval
        self.dead_timeout = dead_timeout
        
        # Cluster state
        self._peers: Dict[str, NodeState] = {}
        self._self_state = NodeState(node_id=node_id)
        
        # Rumor state (for one-time propagation)
        self._rumors: Dict[str, GossipMessage] = {}
        self._seen_rumors: Set[str] = set()
        
        # Application state (key-value)
        self._state: Dict[str, Tuple[Any, int]] = {}  # key -> (value, version)
        
        # Callbacks
        self._on_state_change: List[Callable] = []
        self._on_membership_change: List[Callable] = []
        
        # Statistics
        self._stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'rounds': 0,
            'rumors_propagated': 0,
        }
        
        # Threading
        self._running = False
        self._lock = threading.RLock()
    
    def add_peer(self, node_id: str, address: str = ""):
        """Add a peer to the cluster."""
        with self._lock:
            if node_id not in self._peers:
                self._peers[node_id] = NodeState(node_id=node_id, address=address)
    
    def remove_peer(self, node_id: str):
        """Remove a peer from the cluster."""
        with self._lock:
            self._peers.pop(node_id, None)
    
    def set_state(self, key: str, value: Any):
        """
        Set a key-value in the distributed state.
        
        This will be gossiped to all peers.
        """
        with self._lock:
            current = self._state.get(key, (None, 0))
            self._state[key] = (value, current[1] + 1)
    
    def get_state(self, key: str) -> Optional[Any]:
        """Get a value from the distributed state."""
        with self._lock:
            item = self._state.get(key)
            return item[0] if item else None
    
    def broadcast(self, data: Any, ttl: int = 10):
        """
        Broadcast a rumor to the cluster.
        
        Uses rumor mongering - keeps propagating until enough nodes have it.
        
        Args:
            data: Data to broadcast
            ttl: Time-to-live (rounds before stopping propagation)
        """
        with self._lock:
            version = len(self._rumors)
            msg = GossipMessage(
                msg_type=MessageType.RUMOR,
                sender_id=self.node_id,
                payload=data,
                version=version,
                ttl=ttl,
            )
            self._rumors[msg.key()] = msg
            self._stats['rumors_propagated'] += 1
    
    def gossip_round(self, send_func: Optional[Callable] = None) -> List[GossipMessage]:
        """
        Execute one round of gossip.
        
        Selects random peers and prepares messages for them.
        
        Args:
            send_func: Optional callback to send messages (for integration)
            
        Returns:
            List of messages to send (if no send_func provided)
        """
        with self._lock:
            self._stats['rounds'] += 1
            self._self_state.heartbeat_count += 1
            
            # Select random peers (fanout)
            available_peers = [p for p in self._peers.values() if p.is_alive(self.dead_timeout)]
            
            if not available_peers:
                return []
            
            selected = random.sample(available_peers, min(self.fanout, len(available_peers)))
            
            # Prepare messages
            messages = []
            
            for peer in selected:
                # Create push message with our state
                msg = GossipMessage(
                    msg_type=MessageType.SYNC,
                    sender_id=self.node_id,
                    payload={
                        'heartbeat': self._self_state.heartbeat_count,
                        'state': dict(self._state),
                        'membership': {
                            nid: {
                                'status': ns.status,
                                'heartbeat': ns.heartbeat_count,
                            }
                            for nid, ns in self._peers.items()
                        }
                    },
                    version=self._self_state.heartbeat_count,
                )
                
                messages.append((peer.node_id, msg))
                self._stats['messages_sent'] += 1
                
                if send_func:
                    send_func(peer.node_id, msg)
            
            # Also propagate rumors
            for key, rumor in list(self._rumors.items()):
                if rumor.ttl > 0:
                    for peer in selected:
                        messages.append((peer.node_id, rumor))
                        self._stats['messages_sent'] += 1
                    rumor.ttl -= 1
                else:
                    del self._rumors[key]
            
            return messages
    
    def receive(self, message: GossipMessage):
        """
        Process a received gossip message.
        
        Merges state using "latest wins" for versions.
        
        Args:
            message: The received message
        """
        with self._lock:
            self._stats['messages_received'] += 1
            
            # Update sender's last seen time
            if message.sender_id in self._peers:
                self._peers[message.sender_id].last_seen = time.time()
            else:
                self._peers[message.sender_id] = NodeState(
                    node_id=message.sender_id,
                    last_seen=time.time(),
                )
            
            if message.msg_type == MessageType.SYNC:
                self._process_sync(message)
            elif message.msg_type == MessageType.RUMOR:
                self._process_rumor(message)
            elif message.msg_type == MessageType.HEARTBEAT:
                self._process_heartbeat(message)
    
    def _process_sync(self, message: GossipMessage):
        """Process a SYNC message - merge state."""
        payload = message.payload
        
        # Merge state (higher version wins)
        for key, (value, version) in payload.get('state', {}).items():
            current = self._state.get(key, (None, -1))
            if version > current[1]:
                self._state[key] = (value, version)
                for callback in self._on_state_change:
                    callback(key, value)
        
        # Merge membership
        for node_id, info in payload.get('membership', {}).items():
            if node_id not in self._peers:
                self._peers[node_id] = NodeState(node_id=node_id)
            peer = self._peers[node_id]
            
            if info.get('heartbeat', 0) > peer.heartbeat_count:
                peer.heartbeat_count = info['heartbeat']
                peer.last_seen = time.time()
    
    def _process_rumor(self, message: GossipMessage):
        """Process a RUMOR message - propagate if new."""
        key = message.key()
        
        if key in self._seen_rumors:
            return  # Already seen
        
        self._seen_rumors.add(key)
        
        # Store and propagate
        if message.ttl > 0:
            new_msg = GossipMessage(
                msg_type=message.msg_type,
                sender_id=message.sender_id,
                payload=message.payload,
                version=message.version,
                ttl=message.ttl - 1,
            )
            self._rumors[key] = new_msg
        
        # Notify callbacks
        for callback in self._on_state_change:
            callback(f"rumor:{key}", message.payload)
    
    def _process_heartbeat(self, message: GossipMessage):
        """Process a HEARTBEAT message."""
        if message.sender_id in self._peers:
            self._peers[message.sender_id].last_seen = time.time()
            self._peers[message.sender_id].heartbeat_count = message.version
    
    def on_state_change(self, callback: Callable):
        """Register a callback for state changes."""
        self._on_state_change.append(callback)
    
    def on_membership_change(self, callback: Callable):
        """Register a callback for membership changes."""
        self._on_membership_change.append(callback)
    
    def get_alive_peers(self) -> List[str]:
        """Get list of alive peer node IDs."""
        with self._lock:
            return [
                p.node_id 
                for p in self._peers.values() 
                if p.is_alive(self.dead_timeout)
            ]
    
    def get_dead_peers(self) -> List[str]:
        """Get list of dead peer node IDs."""
        with self._lock:
            return [
                p.node_id 
                for p in self._peers.values() 
                if not p.is_alive(self.dead_timeout)
            ]
    
    @property
    def stats(self) -> Dict[str, int]:
        """Get gossip statistics."""
        with self._lock:
            return self._stats.copy()
    
    def __repr__(self) -> str:
        alive = len(self.get_alive_peers())
        return f"GossipEngine(node={self.node_id}, peers={alive})"


# =============================================================================
# ANTI-ENTROPY (Merkle-based sync)
# =============================================================================

class AntiEntropy:
    """
    Merkle tree-based anti-entropy for efficient state synchronization.
    
    Instead of sending full state, compare Merkle roots and sync only differences.
    """
    
    def __init__(self):
        self._data: Dict[str, bytes] = {}
    
    def _hash(self, data: bytes) -> bytes:
        """Hash data using SHA-256."""
        return hashlib.sha256(data).digest()
    
    def merkle_root(self) -> bytes:
        """Calculate the Merkle root of all data."""
        if not self._data:
            return b'\x00' * 32
        
        # Sort keys for deterministic ordering
        sorted_keys = sorted(self._data.keys())
        
        # Hash each key-value pair
        leaves = [
            self._hash(f"{k}:{self._data[k].hex()}".encode())
            for k in sorted_keys
        ]
        
        # Build tree
        while len(leaves) > 1:
            if len(leaves) % 2 == 1:
                leaves.append(leaves[-1])
            
            leaves = [
                self._hash(leaves[i] + leaves[i+1])
                for i in range(0, len(leaves), 2)
            ]
        
        return leaves[0] if leaves else b'\x00' * 32
    
    def needs_sync(self, other_root: bytes) -> bool:
        """Check if we need to sync with another node."""
        return self.merkle_root() != other_root


# =============================================================================
# DEMO
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("📢 S.P.I.D.E.R. GOSSIP PROTOCOL - Demo")
    print("=" * 60)
    print()
    print("The Math: P(spread) = 1 - (1 - 1/n)^k")
    print("After k rounds, most nodes have received the message")
    print()
    
    # Create cluster of 5 nodes
    nodes = {
        f"node-{i}": GossipEngine(node_id=f"node-{i}", fanout=2)
        for i in range(5)
    }
    
    # Add peers (fully connected for demo)
    for node_id, engine in nodes.items():
        for peer_id in nodes:
            if peer_id != node_id:
                engine.add_peer(peer_id)
    
    print(f"Created cluster with {len(nodes)} nodes")
    print()
    
    # Node-0 broadcasts a rumor
    print("node-0 broadcasts: {'leader': 'node-0'}")
    nodes["node-0"].broadcast({"leader": "node-0"})
    
    # Simulate gossip rounds
    print("\nSimulating gossip rounds...")
    
    for round_num in range(5):
        print(f"\n--- Round {round_num + 1} ---")
        
        # Each node does a gossip round
        all_messages = []
        for node_id, engine in nodes.items():
            messages = engine.gossip_round()
            all_messages.extend(messages)
        
        # Deliver messages
        for target_id, msg in all_messages:
            if target_id in nodes:
                nodes[target_id].receive(msg)
        
        # Check which nodes have the rumor
        have_rumor = sum(
            1 for e in nodes.values() 
            if len(e._seen_rumors) > 0
        )
        print(f"Nodes with rumor: {have_rumor}/{len(nodes)}")
    
    print()
    print("=" * 60)
    print("✅ Gossip Protocol operational!")
