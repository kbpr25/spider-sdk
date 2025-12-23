"""
S.P.I.D.E.R. Multi-Agent Communication Protocol.

Defines message types and data structures for distributed consensus
among coding agents using a Raft-inspired protocol.

All structures are pickle-able for multiprocessing compatibility.
"""

import hashlib
import pickle
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union


# =============================================================================
# ENUMS
# =============================================================================

class NodeState(Enum):
    """
    State of a node in the distributed consensus protocol.
    
    Follows Raft-style state machine:
    - FOLLOWER: Default state, follows the leader.
    - CANDIDATE: Seeking votes to become leader.
    - LEADER: Coordinates proposals and commits.
    - DEAD: Node has crashed or been removed from the cluster.
    """
    FOLLOWER = auto()
    CANDIDATE = auto()
    LEADER = auto()
    DEAD = auto()

    def __str__(self) -> str:
        return self.name


class MessageType(Enum):
    """
    Types of messages in the consensus protocol.
    
    - PROPOSAL: Leader proposes a code change.
    - VOTE: Response to a vote request (election or proposal).
    - HEARTBEAT: Leader's periodic liveness signal.
    - COMMIT: Leader confirms a proposal is committed.
    - REJECT: A proposal or vote request is rejected.
    """
    PROPOSAL = auto()
    VOTE = auto()
    HEARTBEAT = auto()
    COMMIT = auto()
    REJECT = auto()

    # Extended message types for richer protocol
    VOTE_REQUEST = auto()      # Request votes during election
    APPEND_ENTRIES = auto()    # Log replication (Raft terminology)
    ACK = auto()               # Generic acknowledgment
    NACK = auto()              # Negative acknowledgment
    SYNC_REQUEST = auto()      # Request state synchronization
    SYNC_RESPONSE = auto()     # State synchronization data

    def __str__(self) -> str:
        return self.name


class ProposalStatus(Enum):
    """Status of a proposal in the consensus pipeline."""
    PENDING = auto()       # Awaiting votes
    APPROVED = auto()      # Received majority approval
    REJECTED = auto()      # Did not receive majority
    COMMITTED = auto()     # Successfully applied
    ROLLED_BACK = auto()   # Applied but then reverted

    def __str__(self) -> str:
        return self.name


class VoteDecision(Enum):
    """Vote decision for proposals."""
    APPROVE = auto()
    REJECT = auto()
    ABSTAIN = auto()

    def __str__(self) -> str:
        return self.name


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Proposal:
    """
    A proposal for a code change submitted by an agent.
    
    Attributes:
        code_diff: The unified diff or code change content.
        merkle_root_hash: Hash of the codebase state this proposal is based on.
        reasoning_chain: List of reasoning steps that led to this proposal.
        proposal_id: Unique identifier for this proposal.
        author_id: ID of the agent that created this proposal.
        timestamp: Unix timestamp of proposal creation.
        target_files: List of files affected by this proposal.
        priority: Priority level (higher = more urgent).
        dependencies: List of proposal IDs this depends on.
        metadata: Additional context or metadata.
    """
    code_diff: str
    merkle_root_hash: str
    reasoning_chain: List[str] = field(default_factory=list)
    proposal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    author_id: str = ""
    timestamp: float = field(default_factory=time.time)
    target_files: List[str] = field(default_factory=list)
    priority: int = 0
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of the proposal content."""
        content = f"{self.code_diff}{self.merkle_root_hash}{self.author_id}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'proposal_id': self.proposal_id,
            'code_diff': self.code_diff,
            'merkle_root_hash': self.merkle_root_hash,
            'reasoning_chain': self.reasoning_chain,
            'author_id': self.author_id,
            'timestamp': self.timestamp,
            'target_files': self.target_files,
            'priority': self.priority,
            'dependencies': self.dependencies,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Proposal':
        """Reconstruct from dictionary."""
        return cls(
            code_diff=data['code_diff'],
            merkle_root_hash=data['merkle_root_hash'],
            reasoning_chain=data.get('reasoning_chain', []),
            proposal_id=data.get('proposal_id', str(uuid.uuid4())),
            author_id=data.get('author_id', ''),
            timestamp=data.get('timestamp', time.time()),
            target_files=data.get('target_files', []),
            priority=data.get('priority', 0),
            dependencies=data.get('dependencies', []),
            metadata=data.get('metadata', {}),
        )


@dataclass
class Vote:
    """
    A vote cast by an agent on a proposal.
    
    Attributes:
        voter_id: ID of the voting agent.
        proposal_id: ID of the proposal being voted on.
        decision: The vote decision (approve/reject/abstain).
        term: The logical clock term when vote was cast.
        reasoning: Explanation for the vote decision.
        timestamp: Unix timestamp of the vote.
        confidence: Confidence level (0.0 to 1.0).
    """
    voter_id: str
    proposal_id: str
    decision: VoteDecision
    term: int
    reasoning: str = ""
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'voter_id': self.voter_id,
            'proposal_id': self.proposal_id,
            'decision': self.decision.name,
            'term': self.term,
            'reasoning': self.reasoning,
            'timestamp': self.timestamp,
            'confidence': self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Vote':
        """Reconstruct from dictionary."""
        return cls(
            voter_id=data['voter_id'],
            proposal_id=data['proposal_id'],
            decision=VoteDecision[data['decision']],
            term=data['term'],
            reasoning=data.get('reasoning', ''),
            timestamp=data.get('timestamp', time.time()),
            confidence=data.get('confidence', 1.0),
        )


@dataclass
class Message:
    """
    A message in the S.P.I.D.E.R. consensus protocol.
    
    Attributes:
        sender_id: ID of the sending agent.
        term: Logical clock (Raft term) for consistency.
        type: The message type (proposal, vote, heartbeat, etc.).
        payload: The actual content being transmitted.
        signature: SHA-256 hash of the payload for integrity verification.
        message_id: Unique message identifier.
        timestamp: Unix timestamp of message creation.
        recipient_id: Target recipient (None for broadcast).
        ttl: Time-to-live in seconds (0 = no expiry).
        priority: Message priority (higher = more urgent).
    """
    sender_id: str
    term: int
    type: MessageType
    payload: Dict[str, Any] = field(default_factory=dict)
    signature: str = ""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    recipient_id: Optional[str] = None
    ttl: float = 0
    priority: int = 0

    def __post_init__(self):
        """Compute signature if not provided."""
        if not self.signature:
            self.signature = self._compute_signature()

    def _compute_signature(self) -> str:
        """Compute SHA-256 signature of the payload."""
        # Serialize payload deterministically
        content = f"{self.sender_id}{self.term}{self.type.name}"
        content += str(sorted(self.payload.items()))
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def verify_signature(self) -> bool:
        """Verify the message signature is valid."""
        return self.signature == self._compute_signature()

    def is_expired(self) -> bool:
        """Check if the message has expired based on TTL."""
        if self.ttl <= 0:
            return False
        return time.time() > (self.timestamp + self.ttl)

    def is_broadcast(self) -> bool:
        """Check if this is a broadcast message."""
        return self.recipient_id is None

    def to_bytes(self) -> bytes:
        """Serialize to bytes for transmission."""
        return pickle.dumps(self)

    @classmethod
    def from_bytes(cls, data: bytes) -> 'Message':
        """Deserialize from bytes."""
        return pickle.loads(data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'message_id': self.message_id,
            'sender_id': self.sender_id,
            'recipient_id': self.recipient_id,
            'term': self.term,
            'type': self.type.name,
            'payload': self.payload,
            'signature': self.signature,
            'timestamp': self.timestamp,
            'ttl': self.ttl,
            'priority': self.priority,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Reconstruct from dictionary."""
        msg = cls(
            sender_id=data['sender_id'],
            term=data['term'],
            type=MessageType[data['type']],
            payload=data.get('payload', {}),
            signature=data.get('signature', ''),
            message_id=data.get('message_id', str(uuid.uuid4())),
            timestamp=data.get('timestamp', time.time()),
            recipient_id=data.get('recipient_id'),
            ttl=data.get('ttl', 0),
            priority=data.get('priority', 0),
        )
        return msg

    def __repr__(self) -> str:
        recipient = self.recipient_id or "BROADCAST"
        return (
            f"Message(id={self.message_id[:8]}..., "
            f"from={self.sender_id}, to={recipient}, "
            f"term={self.term}, type={self.type.name})"
        )


# =============================================================================
# MESSAGE FACTORY
# =============================================================================

class MessageFactory:
    """Factory for creating common message types."""

    def __init__(self, node_id: str):
        """
        Initialize the factory.
        
        Args:
            node_id: The ID of the node creating messages.
        """
        self.node_id = node_id
        self.current_term = 0

    def set_term(self, term: int) -> None:
        """Update the current term."""
        self.current_term = term

    def create_proposal(
        self,
        proposal: Proposal,
        recipient_id: Optional[str] = None
    ) -> Message:
        """Create a PROPOSAL message."""
        return Message(
            sender_id=self.node_id,
            term=self.current_term,
            type=MessageType.PROPOSAL,
            payload={'proposal': proposal.to_dict()},
            recipient_id=recipient_id,
        )

    def create_vote(
        self,
        vote: Vote,
        recipient_id: str
    ) -> Message:
        """Create a VOTE message."""
        return Message(
            sender_id=self.node_id,
            term=self.current_term,
            type=MessageType.VOTE,
            payload={'vote': vote.to_dict()},
            recipient_id=recipient_id,
        )

    def create_heartbeat(
        self,
        leader_commit_index: int = 0,
        recipient_id: Optional[str] = None
    ) -> Message:
        """Create a HEARTBEAT message."""
        return Message(
            sender_id=self.node_id,
            term=self.current_term,
            type=MessageType.HEARTBEAT,
            payload={
                'leader_id': self.node_id,
                'commit_index': leader_commit_index,
            },
            recipient_id=recipient_id,
            ttl=5.0,  # Heartbeats expire quickly
        )

    def create_commit(
        self,
        proposal_id: str,
        commit_index: int,
        recipient_id: Optional[str] = None
    ) -> Message:
        """Create a COMMIT message."""
        return Message(
            sender_id=self.node_id,
            term=self.current_term,
            type=MessageType.COMMIT,
            payload={
                'proposal_id': proposal_id,
                'commit_index': commit_index,
            },
            recipient_id=recipient_id,
        )

    def create_reject(
        self,
        proposal_id: str,
        reason: str,
        recipient_id: str
    ) -> Message:
        """Create a REJECT message."""
        return Message(
            sender_id=self.node_id,
            term=self.current_term,
            type=MessageType.REJECT,
            payload={
                'proposal_id': proposal_id,
                'reason': reason,
            },
            recipient_id=recipient_id,
        )

    def create_vote_request(
        self,
        last_log_index: int = 0,
        last_log_term: int = 0,
        recipient_id: Optional[str] = None
    ) -> Message:
        """Create a VOTE_REQUEST message for leader election."""
        return Message(
            sender_id=self.node_id,
            term=self.current_term,
            type=MessageType.VOTE_REQUEST,
            payload={
                'candidate_id': self.node_id,
                'last_log_index': last_log_index,
                'last_log_term': last_log_term,
            },
            recipient_id=recipient_id,
        )

    def create_ack(
        self,
        original_message_id: str,
        recipient_id: str
    ) -> Message:
        """Create an ACK message."""
        return Message(
            sender_id=self.node_id,
            term=self.current_term,
            type=MessageType.ACK,
            payload={'ack_for': original_message_id},
            recipient_id=recipient_id,
        )

    def create_nack(
        self,
        original_message_id: str,
        reason: str,
        recipient_id: str
    ) -> Message:
        """Create a NACK message."""
        return Message(
            sender_id=self.node_id,
            term=self.current_term,
            type=MessageType.NACK,
            payload={
                'nack_for': original_message_id,
                'reason': reason,
            },
            recipient_id=recipient_id,
        )


# =============================================================================
# LOG ENTRY
# =============================================================================

@dataclass
class LogEntry:
    """
    An entry in the consensus log.
    
    Attributes:
        index: Position in the log (1-indexed).
        term: Term when entry was created.
        proposal: The proposal this entry represents.
        status: Current status of the entry.
        votes: Votes received for this entry.
        committed: Whether the entry has been committed.
    """
    index: int
    term: int
    proposal: Proposal
    status: ProposalStatus = ProposalStatus.PENDING
    votes: List[Vote] = field(default_factory=list)
    committed: bool = False

    def vote_count(self, decision: VoteDecision) -> int:
        """Count votes of a specific decision type."""
        return sum(1 for v in self.votes if v.decision == decision)

    def has_quorum(self, cluster_size: int) -> bool:
        """Check if we have majority approval."""
        required = (cluster_size // 2) + 1
        return self.vote_count(VoteDecision.APPROVE) >= required

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'index': self.index,
            'term': self.term,
            'proposal': self.proposal.to_dict(),
            'status': self.status.name,
            'votes': [v.to_dict() for v in self.votes],
            'committed': self.committed,
        }


# =============================================================================
# NODE DESCRIPTOR
# =============================================================================

@dataclass
class NodeDescriptor:
    """
    Describes a node in the cluster.
    
    Attributes:
        node_id: Unique node identifier.
        address: Network address (host:port).
        state: Current node state.
        capabilities: List of capabilities this node has.
        last_seen: Last heartbeat timestamp.
        metadata: Additional node metadata.
    """
    node_id: str
    address: str = ""
    state: NodeState = NodeState.FOLLOWER
    capabilities: List[str] = field(default_factory=list)
    last_seen: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_alive(self, timeout: float = 10.0) -> bool:
        """Check if node has been seen recently."""
        return (time.time() - self.last_seen) < timeout

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'node_id': self.node_id,
            'address': self.address,
            'state': self.state.name,
            'capabilities': self.capabilities,
            'last_seen': self.last_seen,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NodeDescriptor':
        """Reconstruct from dictionary."""
        return cls(
            node_id=data['node_id'],
            address=data.get('address', ''),
            state=NodeState[data.get('state', 'FOLLOWER')],
            capabilities=data.get('capabilities', []),
            last_seen=data.get('last_seen', time.time()),
            metadata=data.get('metadata', {}),
        )


# =============================================================================
# CLUSTER STATE
# =============================================================================

@dataclass
class ClusterState:
    """
    Represents the state of the entire cluster.
    
    Attributes:
        nodes: Dictionary of node_id -> NodeDescriptor.
        leader_id: Current leader's node ID.
        current_term: Current consensus term.
        commit_index: Highest log index known to be committed.
        last_applied: Highest log index applied to state machine.
    """
    nodes: Dict[str, NodeDescriptor] = field(default_factory=dict)
    leader_id: Optional[str] = None
    current_term: int = 0
    commit_index: int = 0
    last_applied: int = 0

    def add_node(self, node: NodeDescriptor) -> None:
        """Add or update a node in the cluster."""
        self.nodes[node.node_id] = node

    def remove_node(self, node_id: str) -> None:
        """Remove a node from the cluster."""
        self.nodes.pop(node_id, None)

    def get_alive_nodes(self, timeout: float = 10.0) -> List[NodeDescriptor]:
        """Get list of nodes that are still alive."""
        return [n for n in self.nodes.values() if n.is_alive(timeout)]

    def get_followers(self) -> List[NodeDescriptor]:
        """Get list of follower nodes."""
        return [n for n in self.nodes.values() if n.state == NodeState.FOLLOWER]

    @property
    def cluster_size(self) -> int:
        """Return the number of nodes in the cluster."""
        return len(self.nodes)

    @property
    def quorum_size(self) -> int:
        """Return the quorum size (majority)."""
        return (self.cluster_size // 2) + 1


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def serialize_message(msg: Message) -> bytes:
    """Serialize a message for network transmission."""
    return msg.to_bytes()


def deserialize_message(data: bytes) -> Message:
    """Deserialize a message from network data."""
    return Message.from_bytes(data)


def create_proposal_from_diff(
    diff: str,
    merkle_hash: str,
    reasoning: List[str],
    author_id: str,
    target_files: Optional[List[str]] = None
) -> Proposal:
    """Convenience function to create a proposal."""
    return Proposal(
        code_diff=diff,
        merkle_root_hash=merkle_hash,
        reasoning_chain=reasoning,
        author_id=author_id,
        target_files=target_files or [],
    )


# =============================================================================
# DEMO
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("S.P.I.D.E.R. Communication Protocol Demo")
    print("=" * 60)

    # Create a proposal
    proposal = Proposal(
        code_diff="+ def new_function(): pass",
        merkle_root_hash="abc123def456",
        reasoning_chain=[
            "Identified missing utility function",
            "Analyzed existing patterns",
            "Generated minimal implementation"
        ],
        author_id="agent-001"
    )
    print(f"\n[PROPOSAL] {proposal.proposal_id[:8]}...")
    print(f"  Author: {proposal.author_id}")
    print(f"  Hash: {proposal.compute_hash()[:16]}...")

    # Create message factory
    factory = MessageFactory("agent-001")
    factory.set_term(1)

    # Create and serialize a proposal message
    msg = factory.create_proposal(proposal)
    print(f"\n[MESSAGE] {msg}")
    print(f"  Signature valid: {msg.verify_signature()}")

    # Serialize/deserialize roundtrip
    serialized = msg.to_bytes()
    deserialized = Message.from_bytes(serialized)
    print(f"  Pickle roundtrip: {deserialized.signature == msg.signature}")

    # Create a vote
    vote = Vote(
        voter_id="agent-002",
        proposal_id=proposal.proposal_id,
        decision=VoteDecision.APPROVE,
        term=1,
        reasoning="Code looks correct and follows patterns"
    )
    vote_msg = factory.create_vote(vote, recipient_id="agent-001")
    print(f"\n[VOTE] {vote.decision} from {vote.voter_id}")

    # Create cluster state
    cluster = ClusterState()
    cluster.add_node(NodeDescriptor(node_id="agent-001", state=NodeState.LEADER))
    cluster.add_node(NodeDescriptor(node_id="agent-002", state=NodeState.FOLLOWER))
    cluster.add_node(NodeDescriptor(node_id="agent-003", state=NodeState.FOLLOWER))
    cluster.leader_id = "agent-001"

    print(f"\n[CLUSTER] Size: {cluster.cluster_size}, Quorum: {cluster.quorum_size}")
    for node in cluster.nodes.values():
        print(f"  - {node.node_id}: {node.state}")

    print("\n" + "=" * 60)
