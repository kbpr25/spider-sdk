"""
S.P.I.D.E.R. Distributed Node Implementation.

SpiderNode is a multiprocessing-based agent node that participates
in the distributed consensus protocol for collaborative code changes.

Integrates with ReasoningCore for AI-powered proposal analysis.
"""

import hashlib
import multiprocessing
import queue
import random
import signal
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from multiprocessing import Process, Queue
from typing import Dict, List, Optional, Set

from spider.core.distributed.protocol import (
    Message,
    MessageFactory,
    MessageType,
    NodeState,
    Proposal,
    ProposalStatus,
    Vote,
    VoteDecision,
)
from spider.core.agent.reasoning import ReasoningCore, ReasoningConfig, CheckStage
from spider.core.sre.failure_detector import PhiFailureDetector


# =============================================================================
# COLORS FOR TERMINAL OUTPUT
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    # Node state colors
    LEADER = "\033[95m"      # Magenta
    FOLLOWER = "\033[94m"    # Blue
    CANDIDATE = "\033[93m"   # Yellow
    DEAD = "\033[91m"        # Red
    
    # Message type colors
    HEARTBEAT = "\033[92m"   # Green
    PROPOSAL = "\033[96m"    # Cyan
    VOTE = "\033[93m"        # Yellow
    COMMIT = "\033[95m"      # Magenta
    REJECT = "\033[91m"      # Red


def colorize(text: str, color: str) -> str:
    """Apply ANSI color to text."""
    return f"{color}{text}{Colors.RESET}"


# =============================================================================
# PROPOSAL TRACKER
# =============================================================================

@dataclass
class ProposalTracker:
    """Tracks votes for a proposal."""
    proposal: Proposal
    votes: Dict[str, VoteDecision] = field(default_factory=dict)
    status: ProposalStatus = ProposalStatus.PENDING
    created_at: float = field(default_factory=time.time)

    def add_vote(self, voter_id: str, decision: VoteDecision) -> None:
        """Record a vote from a node."""
        self.votes[voter_id] = decision

    def count_votes(self, decision: VoteDecision) -> int:
        """Count votes of a specific type."""
        return sum(1 for d in self.votes.values() if d == decision)

    def has_quorum(self, cluster_size: int) -> bool:
        """Check if proposal has majority approval."""
        required = (cluster_size // 2) + 1
        return self.count_votes(VoteDecision.APPROVE) >= required

    def is_rejected(self, cluster_size: int) -> bool:
        """Check if proposal is definitively rejected."""
        required = (cluster_size // 2) + 1
        return self.count_votes(VoteDecision.REJECT) >= required


# =============================================================================
# SPIDER NODE
# =============================================================================

class SpiderNode(Process):
    """
    A distributed agent node in the S.P.I.D.E.R. system.
    
    Inherits from multiprocessing.Process for true parallel execution.
    Participates in Raft-inspired consensus for code proposals.
    
    Attributes:
        node_id: Unique identifier for this node.
        state: Current node state (FOLLOWER, CANDIDATE, LEADER, DEAD).
        term: Current consensus term (logical clock).
        message_queue: Queue for receiving messages.
        cluster_queues: Dict mapping node_id -> Queue for other nodes.
    """

    def __init__(
        self,
        node_id: str,
        message_queue: Queue,
        cluster_queues: Dict[str, Queue],
        election_timeout: float = 2.0,
        heartbeat_interval: float = 0.5,
        verbose: bool = True,
        codebase_path: Optional[str] = None,
        simulation_mode: bool = True,
    ):
        """
        Initialize the SpiderNode.
        
        Args:
            node_id: Unique identifier for this node.
            message_queue: Queue for receiving messages.
            cluster_queues: Dictionary of node_id -> Queue for other nodes.
            election_timeout: Seconds before starting election if no heartbeat.
            heartbeat_interval: Seconds between leader heartbeats.
            verbose: Enable detailed logging.
            codebase_path: Path to codebase for indexing (None for simulation).
            simulation_mode: Use simulated AI if True (no Ollama required).
        """
        super().__init__(name=f"SpiderNode-{node_id}")
        
        self.node_id = node_id
        self.message_queue = message_queue
        self.cluster_queues = cluster_queues
        self.election_timeout = election_timeout
        self.heartbeat_interval = heartbeat_interval
        self.verbose = verbose
        self.codebase_path = codebase_path
        self.simulation_mode = simulation_mode
        
        # State (will be initialized in run() for process safety)
        self._state = NodeState.FOLLOWER
        self._term = 0
        self._leader_id: Optional[str] = None
        self._voted_for: Optional[str] = None
        self._last_heartbeat = time.time()
        
        # Proposal tracking
        self._proposals: Dict[str, ProposalTracker] = {}
        self._pending_votes: Dict[str, Set[str]] = defaultdict(set)
        
        # Control flags
        self._running = multiprocessing.Event()
        self._shutdown = multiprocessing.Event()
        
        # ReasoningCore (brain) - initialized lazily in run() for process safety
        self._brain: Optional[ReasoningCore] = None
        
        # Phi Accrual Failure Detector - peer health monitoring
        self._peer_health: Dict[str, PhiFailureDetector] = {}
        self._health_check_interval = 1.0  # seconds between health checks
        self._last_health_check = 0.0
        self._dead_nodes: Set[str] = set()  # Track nodes we've already marked as dead

    @property
    def state(self) -> NodeState:
        """Get current node state."""
        return self._state

    @state.setter
    def state(self, new_state: NodeState) -> None:
        """Set node state with logging."""
        if self._state != new_state:
            self._log(f"State transition: {self._state} -> {new_state}", level="STATE")
            self._state = new_state

    @property
    def term(self) -> int:
        """Get current term."""
        return self._term

    @term.setter
    def term(self, new_term: int) -> None:
        """Set term with logging."""
        if self._term != new_term:
            self._log(f"Term update: {self._term} -> {new_term}", level="TERM")
            self._term = new_term
            self._voted_for = None  # Reset vote on term change

    # =========================================================================
    # LOGGING
    # =========================================================================

    def _log(self, message: str, level: str = "INFO") -> None:
        """Log a formatted message."""
        if not self.verbose:
            return

        timestamp = time.strftime("%H:%M:%S")
        
        # Color based on state
        state_color = {
            NodeState.LEADER: Colors.LEADER,
            NodeState.FOLLOWER: Colors.FOLLOWER,
            NodeState.CANDIDATE: Colors.CANDIDATE,
            NodeState.DEAD: Colors.DEAD,
        }.get(self._state, Colors.RESET)

        node_label = colorize(f"[Node {self.node_id}]", state_color)
        state_label = colorize(f"[{self._state.name}]", state_color)
        
        print(f"{timestamp} {node_label} {state_label} {message}")
        sys.stdout.flush()

    def _log_message(self, action: str, msg: Message) -> None:
        """Log a message event."""
        msg_color = {
            MessageType.HEARTBEAT: Colors.HEARTBEAT,
            MessageType.PROPOSAL: Colors.PROPOSAL,
            MessageType.VOTE: Colors.VOTE,
            MessageType.COMMIT: Colors.COMMIT,
            MessageType.REJECT: Colors.REJECT,
        }.get(msg.type, Colors.RESET)

        msg_type = colorize(msg.type.name, msg_color)
        self._log(f"{action} {msg_type} from Node {msg.sender_id} (term={msg.term})")

    # =========================================================================
    # MESSAGE FACTORY
    # =========================================================================

    def _create_factory(self) -> MessageFactory:
        """Create a message factory for this node."""
        factory = MessageFactory(self.node_id)
        factory.set_term(self._term)
        return factory

    # =========================================================================
    # MAIN LOOP
    # =========================================================================

    def run(self) -> None:
        """
        Main process loop.
        
        Continuously processes messages and handles timeouts.
        """
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        self._running.set()
        self._last_heartbeat = time.time()
        
        # Initialize ReasoningCore (brain) in the child process
        self._initialize_brain()

        self._log("Node started", level="INIT")

        try:
            while self._running.is_set() and not self._shutdown.is_set():
                # Process incoming messages
                self._process_messages()

                # Check for heartbeat timeout (followers only)
                if self._state == NodeState.FOLLOWER:
                    self._check_election_timeout()

                # Send heartbeats (leaders only)
                if self._state == NodeState.LEADER:
                    self._send_heartbeats()

                # Periodic health check using Phi Failure Detector
                self._check_peer_health()

                # Small sleep to prevent busy-waiting
                time.sleep(0.01)

        except Exception as e:
            self._log(f"Error in main loop: {e}", level="ERROR")
        finally:
            self._state = NodeState.DEAD
            self._log("Node shutdown complete", level="SHUTDOWN")

    def _initialize_brain(self) -> None:
        """Initialize the ReasoningCore in the child process."""
        config = ReasoningConfig(
            simulation_mode=self.simulation_mode,
            log_level="WARNING",  # Reduce noise from ReasoningCore
        )
        
        if self.codebase_path:
            try:
                self._brain = ReasoningCore.from_codebase(self.codebase_path, config=config)
                self._log("🧠 Brain initialized with codebase index", level="INIT")
            except Exception as e:
                self._log(f"⚠️ Failed to load codebase, using simulation: {e}", level="WARN")
                self._brain = ReasoningCore(config=config)
        else:
            self._brain = ReasoningCore(config=config)
            mode = "simulation" if self.simulation_mode else "live"
            self._log(f"🧠 Brain initialized in {mode} mode", level="INIT")

    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals."""
        self._log(f"Received signal {signum}, initiating shutdown...", level="SIGNAL")
        self.shutdown()

    def _process_messages(self) -> None:
        """Process all pending messages in the queue."""
        try:
            while True:
                msg = self.message_queue.get_nowait()
                self.handle_message(msg)
        except queue.Empty:
            pass

    def _check_election_timeout(self) -> None:
        """Check if election timeout has expired."""
        elapsed = time.time() - self._last_heartbeat
        if elapsed > self.election_timeout:
            self._log("Election timeout - starting election", level="ELECTION")
            self._start_election()

    def _send_heartbeats(self) -> None:
        """Send periodic heartbeats to all followers."""
        # Only send heartbeats at configured interval
        if not hasattr(self, '_last_heartbeat_sent'):
            self._last_heartbeat_sent = 0

        if time.time() - self._last_heartbeat_sent >= self.heartbeat_interval:
            factory = self._create_factory()
            heartbeat = factory.create_heartbeat()
            self.broadcast(heartbeat)
            self._last_heartbeat_sent = time.time()

    # =========================================================================
    # MESSAGE HANDLING
    # =========================================================================

    def handle_message(self, msg: Message) -> None:
        """
        Handle an incoming message based on its type.
        
        Args:
            msg: The received message.
        """
        # Verify message signature
        if not msg.verify_signature():
            self._log(f"REJECTED message from {msg.sender_id}: invalid signature", level="SECURITY")
            return

        # Update term if sender has higher term
        if msg.term > self._term:
            self.term = msg.term
            self.state = NodeState.FOLLOWER
            self._leader_id = None

        # Log the message
        self._log_message("Received", msg)

        # Simulate "thinking time"
        time.sleep(random.uniform(0.05, 0.2))

        # Dispatch to appropriate handler
        handlers = {
            MessageType.HEARTBEAT: self._handle_heartbeat,
            MessageType.PROPOSAL: self._handle_proposal,
            MessageType.VOTE: self._handle_vote,
            MessageType.COMMIT: self._handle_commit,
            MessageType.REJECT: self._handle_reject,
            MessageType.VOTE_REQUEST: self._handle_vote_request,
            MessageType.ACK: self._handle_ack,
        }

        handler = handlers.get(msg.type)
        if handler:
            handler(msg)
        else:
            self._log(f"Unknown message type: {msg.type}", level="WARN")

    def _handle_heartbeat(self, msg: Message) -> None:
        """Handle HEARTBEAT message with Phi Failure Detector tracking."""
        self._last_heartbeat = time.time()
        self._leader_id = msg.sender_id
        
        # Update Phi Failure Detector for this peer
        sender_id = msg.sender_id
        if sender_id not in self._peer_health:
            self._peer_health[sender_id] = PhiFailureDetector(
                threshold=8.0,
                min_std_deviation=0.1,
            )
        
        detector = self._peer_health[sender_id]
        detector.heartbeat()
        
        # Log if node is suspicious (Phi > 2.0)
        phi = detector.phi()
        if phi > 2.0:
            self._log(
                colorize(f"[Watchdog] Node {sender_id} phi={phi:.2f}", Colors.YELLOW),
                level="WATCHDOG"
            )

        if self._state == NodeState.CANDIDATE:
            self.state = NodeState.FOLLOWER

    def _check_peer_health(self) -> None:
        """
        Periodic health check using Phi Failure Detector.
        
        Runs every _health_check_interval seconds.
        Detects and removes dead nodes from the cluster.
        """
        now = time.time()
        if now - self._last_health_check < self._health_check_interval:
            return
        
        self._last_health_check = now
        
        # Check all peer health detectors
        nodes_to_remove = []
        for node_id, detector in self._peer_health.items():
            phi = detector.phi()
            
            # Skip nodes we've already marked as dead
            if node_id in self._dead_nodes:
                continue
            
            # Check if node has exceeded the threshold
            if phi > detector.threshold:
                self._log(
                    colorize(
                        f"💀 [KILL] Detecting Node {node_id} as DEAD (Phi={phi:.2f}). "
                        f"Removing from cluster map.",
                        Colors.RED
                    ),
                    level="FAILURE"
                )
                nodes_to_remove.append(node_id)
                self._dead_nodes.add(node_id)
        
        # Remove dead nodes from cluster queues (stop trying to communicate)
        for node_id in nodes_to_remove:
            if node_id in self.cluster_queues:
                # Don't actually delete, just mark as dead
                # In production, you might want to remove from routing
                self._log(f"Stopped routing to dead node: {node_id}", level="ROUTE")

    def _handle_proposal(self, msg: Message) -> None:
        """Handle PROPOSAL message - analyze with ReasoningCore and vote."""
        proposal_data = msg.payload.get('proposal', {})
        proposal = Proposal.from_dict(proposal_data)

        # Log thinking state
        self._log(colorize("🧠 Thinking...", Colors.PROPOSAL), level="ANALYSIS")

        # Analyze proposal using ReasoningCore (AI-powered)
        result = self._analyze_proposal_with_brain(proposal)

        # Create vote based on analysis
        factory = self._create_factory()
        vote = Vote(
            voter_id=self.node_id,
            proposal_id=proposal.proposal_id,
            decision=result['decision'],
            term=self._term,
            reasoning=result['reasoning'],
            confidence=result['confidence'],
        )

        # Send vote back to proposer
        vote_msg = factory.create_vote(vote, recipient_id=msg.sender_id)
        self._send_to(msg.sender_id, vote_msg)

        # Log decision with details
        if result['decision'] == VoteDecision.APPROVE:
            decision_str = colorize("✅ APPROVE", Colors.HEARTBEAT)
        else:
            decision_str = colorize("❌ REJECT", Colors.REJECT)
        
        self._log(f"{decision_str} vote for proposal {proposal.proposal_id[:8]}... ({result['reasoning'][:50]}...)")

    def _analyze_proposal_with_brain(self, proposal: Proposal) -> Dict:
        """
        Analyze a proposal using ReasoningCore.
        
        Uses three-stage verification:
        1. Fast Check (Bloom Filter) - instant file existence check
        2. Integrity Check (Merkle Tree) - state verification
        3. Deep Check (LLM/Simulation) - semantic code review
        
        Returns:
            Dict with 'decision', 'reasoning', 'confidence', 'stage'.
        """
        if not self._brain:
            # Fallback if brain not initialized
            return {
                'decision': VoteDecision.APPROVE,
                'reasoning': 'No brain configured, auto-approve',
                'confidence': 0.5,
                'stage': 'none',
            }

        # Run full analysis
        result = self._brain.analyze_proposal_detailed(proposal)

        # Log stage-specific information
        if result.stage_reached == CheckStage.FAST_CHECK:
            self._log(colorize("🛡️ Bloom Filter Reject - unknown files!", Colors.REJECT), level="FILTER")
        elif result.stage_reached == CheckStage.INTEGRITY_CHECK:
            self._log(colorize("🔗 Merkle Integrity Reject - state mismatch!", Colors.REJECT), level="INTEGRITY")
        elif result.stage_reached == CheckStage.DEEP_CHECK:
            if result.llm_review:
                risk_color = Colors.REJECT if result.llm_review.risk_score > 5 else Colors.HEARTBEAT
                self._log(colorize(f"📊 Risk Score: {result.llm_review.risk_score}/10", risk_color), level="RISK")

        return {
            'decision': result.decision,
            'reasoning': result.reasoning,
            'confidence': result.confidence,
            'stage': result.stage_reached.name,
        }

    def _handle_vote(self, msg: Message) -> None:
        """Handle VOTE message - count votes and commit if quorum reached."""
        vote_data = msg.payload.get('vote', {})
        vote = Vote.from_dict(vote_data)

        proposal_id = vote.proposal_id

        # Track the vote
        if proposal_id not in self._proposals:
            self._log(f"Received vote for unknown proposal {proposal_id[:8]}...", level="WARN")
            return

        tracker = self._proposals[proposal_id]
        tracker.add_vote(vote.voter_id, vote.decision)

        cluster_size = len(self.cluster_queues) + 1  # Include self
        approve_count = tracker.count_votes(VoteDecision.APPROVE)
        reject_count = tracker.count_votes(VoteDecision.REJECT)

        self._log(
            f"Vote tally for {proposal_id[:8]}...: "
            f"APPROVE={approve_count}, REJECT={reject_count}, "
            f"needed={(cluster_size // 2) + 1}"
        )

        # Check for quorum
        if tracker.has_quorum(cluster_size):
            self._commit_proposal(proposal_id)
        elif tracker.is_rejected(cluster_size):
            self._reject_proposal(proposal_id)

    def _commit_proposal(self, proposal_id: str) -> None:
        """Commit a proposal that has reached quorum."""
        tracker = self._proposals.get(proposal_id)
        if not tracker:
            return

        tracker.status = ProposalStatus.COMMITTED
        self._log(
            colorize(f"COMMIT proposal {proposal_id[:8]}... - Quorum reached!", Colors.COMMIT),
            level="CONSENSUS"
        )

        # Broadcast commit to all nodes
        factory = self._create_factory()
        commit_msg = factory.create_commit(proposal_id, commit_index=len(self._proposals))
        self.broadcast(commit_msg)

    def _reject_proposal(self, proposal_id: str) -> None:
        """Reject a proposal that failed to get quorum."""
        tracker = self._proposals.get(proposal_id)
        if not tracker:
            return

        tracker.status = ProposalStatus.REJECTED
        self._log(
            colorize(f"REJECTED proposal {proposal_id[:8]}... - No quorum", Colors.REJECT),
            level="CONSENSUS"
        )

    def _handle_commit(self, msg: Message) -> None:
        """Handle COMMIT message - apply the committed proposal."""
        proposal_id = msg.payload.get('proposal_id', '')
        self._log(f"Applied committed proposal {proposal_id[:8]}...", level="APPLY")

    def _handle_reject(self, msg: Message) -> None:
        """Handle REJECT message."""
        proposal_id = msg.payload.get('proposal_id', '')
        reason = msg.payload.get('reason', 'Unknown')
        self._log(f"Proposal {proposal_id[:8]}... rejected: {reason}", level="REJECT")

    def _handle_vote_request(self, msg: Message) -> None:
        """Handle VOTE_REQUEST for leader election."""
        candidate_id = msg.payload.get('candidate_id', msg.sender_id)

        # Grant vote if we haven't voted this term
        grant_vote = (self._voted_for is None or self._voted_for == candidate_id)

        if grant_vote:
            self._voted_for = candidate_id
            self._log(f"Granting vote to candidate {candidate_id}")
        else:
            self._log(f"Denying vote to {candidate_id} (already voted for {self._voted_for})")

        # Send vote response
        factory = self._create_factory()
        vote = Vote(
            voter_id=self.node_id,
            proposal_id=f"election-{self._term}",
            decision=VoteDecision.APPROVE if grant_vote else VoteDecision.REJECT,
            term=self._term,
        )
        vote_msg = factory.create_vote(vote, recipient_id=candidate_id)
        self._send_to(candidate_id, vote_msg)

    def _handle_ack(self, msg: Message) -> None:
        """Handle ACK message."""
        ack_for = msg.payload.get('ack_for', '')
        self._log(f"Received ACK for message {ack_for[:8]}...")

    # =========================================================================
    # ELECTION
    # =========================================================================

    def _start_election(self) -> None:
        """Start a leader election."""
        self.state = NodeState.CANDIDATE
        self.term += 1
        self._voted_for = self.node_id
        self._election_votes = {self.node_id: VoteDecision.APPROVE}

        self._log(f"Starting election for term {self._term}", level="ELECTION")

        # Request votes from all nodes
        factory = self._create_factory()
        vote_request = factory.create_vote_request()
        self.broadcast(vote_request)

        # Set election timeout
        self._last_heartbeat = time.time()

    # =========================================================================
    # PROPOSAL SUBMISSION
    # =========================================================================

    def submit_proposal(self, proposal: Proposal) -> str:
        """
        Submit a new proposal (leader only).
        
        Args:
            proposal: The proposal to submit.
            
        Returns:
            The proposal ID.
        """
        if self._state != NodeState.LEADER:
            self._log("Cannot submit proposal - not leader", level="WARN")
            return ""

        proposal.author_id = self.node_id
        tracker = ProposalTracker(proposal=proposal)
        tracker.add_vote(self.node_id, VoteDecision.APPROVE)  # Vote for own proposal
        self._proposals[proposal.proposal_id] = tracker

        self._log(f"Submitting proposal {proposal.proposal_id[:8]}...", level="PROPOSAL")

        # Broadcast proposal to all nodes
        factory = self._create_factory()
        msg = factory.create_proposal(proposal)
        self.broadcast(msg)

        return proposal.proposal_id

    # =========================================================================
    # COMMUNICATION
    # =========================================================================

    def broadcast(self, msg: Message) -> None:
        """
        Send a message to all other nodes in the cluster.
        
        Args:
            msg: The message to broadcast.
        """
        for node_id, node_queue in self.cluster_queues.items():
            if node_id != self.node_id:
                self._send_to_queue(node_queue, msg)

    def _send_to(self, node_id: str, msg: Message) -> bool:
        """
        Send a message to a specific node.
        
        Args:
            node_id: Target node ID.
            msg: The message to send.
            
        Returns:
            True if sent successfully.
        """
        if node_id not in self.cluster_queues:
            self._log(f"Unknown node: {node_id}", level="WARN")
            return False

        return self._send_to_queue(self.cluster_queues[node_id], msg)

    def _send_to_queue(self, target_queue: Queue, msg: Message) -> bool:
        """Send a message to a queue."""
        try:
            target_queue.put_nowait(msg)
            return True
        except queue.Full:
            self._log("Queue full, message dropped", level="WARN")
            return False

    # =========================================================================
    # CONTROL
    # =========================================================================

    def shutdown(self) -> None:
        """Initiate graceful shutdown."""
        self._log("Initiating shutdown...", level="SHUTDOWN")
        self._shutdown.set()
        self._running.clear()

    def promote_to_leader(self) -> None:
        """Force promotion to leader (for testing)."""
        self.state = NodeState.LEADER
        self._leader_id = self.node_id
        self._log("Promoted to LEADER", level="PROMOTE")


# =============================================================================
# CLUSTER MANAGER
# =============================================================================

class SpiderCluster:
    """
    Manages a cluster of SpiderNodes.
    
    Provides convenient methods for creating, starting, and stopping
    a cluster of nodes for testing and simulation.
    """

    def __init__(self, node_count: int = 3, verbose: bool = True):
        """
        Initialize the cluster.
        
        Args:
            node_count: Number of nodes to create.
            verbose: Enable verbose logging.
        """
        self.node_count = node_count
        self.verbose = verbose
        self.nodes: Dict[str, SpiderNode] = {}
        self.queues: Dict[str, Queue] = {}

    def setup(self) -> 'SpiderCluster':
        """
        Set up the cluster nodes and queues.
        
        Returns:
            self for method chaining.
        """
        # Create queues for each node
        for i in range(self.node_count):
            node_id = f"node-{i}"
            self.queues[node_id] = Queue()

        # Create nodes with references to all queues
        for i in range(self.node_count):
            node_id = f"node-{i}"
            self.nodes[node_id] = SpiderNode(
                node_id=node_id,
                message_queue=self.queues[node_id],
                cluster_queues=self.queues,
                verbose=self.verbose,
            )

        return self

    def start(self) -> None:
        """Start all nodes in the cluster."""
        print(f"\n{'='*60}")
        print(f"Starting S.P.I.D.E.R. Cluster with {self.node_count} nodes")
        print(f"{'='*60}\n")

        for node in self.nodes.values():
            node.start()

        # Small delay to let nodes initialize
        time.sleep(0.5)

    def stop(self) -> None:
        """Stop all nodes gracefully."""
        print(f"\n{'='*60}")
        print("Shutting down cluster...")
        print(f"{'='*60}\n")

        for node in self.nodes.values():
            node.shutdown()

        for node in self.nodes.values():
            node.join(timeout=2.0)
            if node.is_alive():
                node.terminate()

    def get_leader(self) -> Optional[SpiderNode]:
        """Get the current leader node."""
        for node in self.nodes.values():
            if node.state == NodeState.LEADER:
                return node
        return None

    def elect_leader(self, node_id: str) -> None:
        """Force a specific node to become leader."""
        if node_id in self.nodes:
            self.nodes[node_id].promote_to_leader()


# =============================================================================
# DEMO
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("S.P.I.D.E.R. Distributed Node Demo")
    print("=" * 60)

    # Create a small cluster
    cluster = SpiderCluster(node_count=3, verbose=True)
    cluster.setup()

    try:
        # Start the cluster
        cluster.start()

        # Wait for nodes to initialize
        time.sleep(1)

        # Elect a leader
        cluster.elect_leader("node-0")
        time.sleep(0.5)

        # Submit a proposal from the leader
        leader = cluster.nodes["node-0"]
        proposal = Proposal(
            code_diff="+ def new_feature(): return True",
            merkle_root_hash="abc123",
            reasoning_chain=[
                "Analyzed requirements",
                "Designed solution",
                "Implemented feature"
            ]
        )

        # Put proposal message in leader's queue to trigger processing
        factory = MessageFactory("external")
        factory.set_term(1)
        msg = factory.create_proposal(proposal)

        # Send proposal to all nodes
        for node_id, q in cluster.queues.items():
            q.put(msg)

        # Let the system process
        time.sleep(3)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cluster.stop()

    print("\n" + "=" * 60)
    print("Demo complete")
    print("=" * 60)
