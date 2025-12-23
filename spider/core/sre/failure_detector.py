"""
S.P.I.D.E.R. Phi Accrual Failure Detector
==========================================

Implementation of the Phi Accrual Failure Detector algorithm, as used in
Apache Cassandra, Akka, and other production distributed systems.

Instead of a binary "Up/Down" decision, this calculates a continuous
suspicion level (Phi) based on the statistical distribution of heartbeat
arrival times.

The Math:
    φ = -log₁₀(1 - F(t))

Where F(t) is the CDF of the normal distribution of heartbeat intervals.

Reference:
    Hayashibara, N., et al. "The φ Accrual Failure Detector" (2004)
    https://www.cs.cornell.edu/projects/Quicksilver/public_pdfs/SRDS04.pdf
"""

import math
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple


# =============================================================================
# MATHEMATICAL UTILITIES
# =============================================================================

def normal_cdf(x: float, mean: float, std_dev: float) -> float:
    """
    Calculate the Cumulative Distribution Function (CDF) of the normal distribution.
    
    Uses the error function (erf) approximation for standard normal CDF:
    Φ(z) = 0.5 * (1 + erf(z / √2))
    
    Args:
        x: The value to evaluate.
        mean: Mean of the distribution.
        std_dev: Standard deviation of the distribution.
        
    Returns:
        Probability that a random variable is less than or equal to x.
    """
    if std_dev <= 0:
        # Degenerate case: all values are the same
        return 1.0 if x >= mean else 0.0
    
    # Standardize x to z-score
    z = (x - mean) / std_dev
    
    # CDF using error function
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def exponential_cdf(x: float, rate: float) -> float:
    """
    Calculate the CDF of the exponential distribution.
    
    F(x) = 1 - e^(-λx)
    
    Alternative model for heartbeat intervals (memoryless).
    
    Args:
        x: The value to evaluate.
        rate: Rate parameter λ = 1/mean.
        
    Returns:
        Probability that a random variable is less than or equal to x.
    """
    if rate <= 0 or x < 0:
        return 0.0
    return 1.0 - math.exp(-rate * x)


# =============================================================================
# PHI ACCRUAL FAILURE DETECTOR
# =============================================================================

class PhiFailureDetector:
    """
    Phi Accrual Failure Detector for probabilistic node liveness detection.
    
    Instead of binary up/down status, calculates a continuous suspicion level
    based on the statistical distribution of heartbeat arrival times.
    
    Higher Phi values indicate higher probability of failure:
    - Phi 1: ~10% chance of being down
    - Phi 2: ~1% chance of being down
    - Phi 3: ~0.1% chance of being down
    - Phi 8: ~0.000001% chance of being down (default threshold)
    
    Attributes:
        threshold: Phi value above which a node is considered unavailable.
        min_std_deviation: Minimum std dev to prevent spikes.
        max_sample_size: Maximum heartbeat intervals to track.
        acceptable_heartbeat_pause: Tolerance for missed heartbeats.
    """

    def __init__(
        self,
        threshold: float = 8.0,
        max_sample_size: int = 100,
        min_std_deviation: float = 0.1,
        acceptable_heartbeat_pause: float = 0.0,
        first_heartbeat_estimate: float = 1.0,
    ):
        """
        Initialize the Phi Failure Detector.
        
        Args:
            threshold: Phi value for failure declaration (default 8.0 = 99.999999% sure).
            max_sample_size: Sliding window size for heartbeat history.
            min_std_deviation: Minimum std deviation in seconds (prevents over-sensitivity).
            acceptable_heartbeat_pause: Extra grace period in seconds.
            first_heartbeat_estimate: Assumed interval before first heartbeat.
        """
        self.threshold = threshold
        self.max_sample_size = max_sample_size
        self.min_std_deviation = min_std_deviation
        self.acceptable_heartbeat_pause = acceptable_heartbeat_pause
        self.first_heartbeat_estimate = first_heartbeat_estimate
        
        # Heartbeat interval history (sliding window)
        self._history: Deque[float] = deque(maxlen=max_sample_size)
        
        # Timestamps
        self._last_heartbeat: Optional[float] = None
        self._creation_time: float = time.time()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics cache (updated on each heartbeat)
        self._cached_mean: float = first_heartbeat_estimate
        self._cached_std: float = first_heartbeat_estimate / 4

    def heartbeat(self) -> None:
        """
        Record the receipt of a heartbeat.
        
        Updates the interval history and recalculates statistics.
        """
        with self._lock:
            now = time.time()
            
            if self._last_heartbeat is not None:
                # Calculate interval since last heartbeat
                interval = now - self._last_heartbeat
                self._history.append(interval)
                
                # Update cached statistics
                self._update_statistics()
            
            self._last_heartbeat = now

    def _update_statistics(self) -> None:
        """Update cached mean and standard deviation."""
        if len(self._history) == 0:
            self._cached_mean = self.first_heartbeat_estimate
            self._cached_std = self.first_heartbeat_estimate / 4
            return
        
        # Calculate mean
        self._cached_mean = statistics.mean(self._history)
        
        # Calculate standard deviation with minimum floor
        if len(self._history) >= 2:
            self._cached_std = max(
                statistics.stdev(self._history),
                self.min_std_deviation
            )
        else:
            self._cached_std = max(
                self._cached_mean / 4,
                self.min_std_deviation
            )

    @property
    def mean_interval(self) -> float:
        """Get the mean heartbeat interval."""
        with self._lock:
            return self._cached_mean

    @property
    def std_deviation(self) -> float:
        """Get the standard deviation of heartbeat intervals."""
        with self._lock:
            return self._cached_std

    def phi(self) -> float:
        """
        Calculate the current Phi (suspicion level).
        
        Phi = -log₁₀(1 - F(t))
        
        Where:
        - t = time since last heartbeat
        - F(t) = CDF of normal distribution with historical mean/std
        
        Returns:
            Phi value (higher = more suspicious).
            0.0 if no heartbeat history.
        """
        with self._lock:
            if self._last_heartbeat is None:
                # No heartbeats yet - assume healthy if recently created
                age = time.time() - self._creation_time
                if age < self.first_heartbeat_estimate * 2:
                    return 0.0
                # Otherwise, calculate based on creation time
                elapsed = age
            else:
                elapsed = time.time() - self._last_heartbeat
            
            # Subtract acceptable pause
            elapsed = max(0.0, elapsed - self.acceptable_heartbeat_pause)
            
            return self._calculate_phi(elapsed)

    def _calculate_phi(self, elapsed: float) -> float:
        """
        Calculate Phi for a given elapsed time.
        
        Args:
            elapsed: Seconds since last heartbeat.
            
        Returns:
            Phi suspicion level.
        """
        # Probability that we should have received a heartbeat by now
        # (using normal distribution CDF)
        p_late = normal_cdf(elapsed, self._cached_mean, self._cached_std)
        
        # Clamp to prevent log(0)
        p_late = min(p_late, 0.9999999999)
        
        # Probability of being on time (complement)
        p_on_time = 1.0 - p_late
        
        if p_on_time <= 0:
            # Effectively certain the node is down
            return float('inf')
        
        # Phi = -log₁₀(probability of being on time)
        phi = -math.log10(p_on_time)
        
        return max(0.0, phi)

    def is_available(self) -> bool:
        """
        Check if the monitored node is considered available.
        
        Returns:
            True if Phi < threshold, False otherwise.
        """
        return self.phi() < self.threshold

    def availability_probability(self) -> float:
        """
        Get the probability that the node is available.
        
        Returns:
            Probability between 0.0 and 1.0.
        """
        current_phi = self.phi()
        if current_phi == float('inf'):
            return 0.0
        # Convert Phi back to probability: P = 10^(-phi)
        return 10.0 ** (-current_phi)

    def time_until_suspicious(self, target_phi: Optional[float] = None) -> float:
        """
        Calculate time until the node becomes suspicious.
        
        Args:
            target_phi: Target suspicion level (default: threshold).
            
        Returns:
            Seconds until the target Phi is reached.
            Negative if already past that point.
        """
        if target_phi is None:
            target_phi = self.threshold
        
        with self._lock:
            if self._last_heartbeat is None:
                return 0.0
            
            elapsed = time.time() - self._last_heartbeat
            
            # Target probability: 10^(-target_phi)
            target_prob = 10.0 ** (-target_phi)
            
            # Solve for t in: 1 - F(t) = target_prob
            # F(t) = 1 - target_prob
            # t = mean + std * Φ⁻¹(1 - target_prob)
            
            target_cdf = 1.0 - target_prob
            
            # Inverse CDF (quantile function) approximation
            if target_cdf <= 0:
                target_t = 0
            elif target_cdf >= 1:
                target_t = float('inf')
            else:
                # Use inverse error function approximation
                z = math.sqrt(2) * self._inverse_erf(2 * target_cdf - 1)
                target_t = self._cached_mean + self._cached_std * z
            
            return target_t + self.acceptable_heartbeat_pause - elapsed

    @staticmethod
    def _inverse_erf(x: float) -> float:
        """
        Approximate inverse error function.
        
        Uses Abramowitz and Stegun approximation.
        """
        if x < -1 or x > 1:
            return float('nan')
        if x == -1:
            return float('-inf')
        if x == 1:
            return float('inf')
        
        # Abramowitz and Stegun approximation
        a = 0.147
        ln_term = math.log(1 - x * x)
        
        term1 = (2 / (math.pi * a)) + (ln_term / 2)
        term2 = ln_term / a
        
        return math.copysign(
            math.sqrt(math.sqrt(term1 * term1 - term2) - term1),
            x
        )

    def reset(self) -> None:
        """Reset the detector to initial state."""
        with self._lock:
            self._history.clear()
            self._last_heartbeat = None
            self._creation_time = time.time()
            self._cached_mean = self.first_heartbeat_estimate
            self._cached_std = self.first_heartbeat_estimate / 4

    @property
    def sample_count(self) -> int:
        """Get the number of samples in history."""
        with self._lock:
            return len(self._history)

    @property
    def last_heartbeat_time(self) -> Optional[float]:
        """Get the timestamp of the last heartbeat."""
        with self._lock:
            return self._last_heartbeat

    def to_dict(self) -> Dict:
        """Export current state as dictionary."""
        with self._lock:
            return {
                'phi': self.phi(),
                'is_available': self.is_available(),
                'availability_probability': self.availability_probability(),
                'mean_interval': self._cached_mean,
                'std_deviation': self._cached_std,
                'sample_count': len(self._history),
                'last_heartbeat': self._last_heartbeat,
                'threshold': self.threshold,
            }

    def __repr__(self) -> str:
        phi = self.phi()
        status = "UP" if self.is_available() else "DOWN"
        return (
            f"PhiFailureDetector(phi={phi:.2f}, status={status}, "
            f"samples={self.sample_count}, threshold={self.threshold})"
        )


# =============================================================================
# NODE MONITOR
# =============================================================================

@dataclass
class NodeHealth:
    """Health status of a monitored node."""
    node_id: str
    phi: float
    is_available: bool
    probability: float
    last_seen: Optional[float]
    sample_count: int
    mean_interval: float


class ClusterHealthMonitor:
    """
    Monitors health of all nodes in a cluster using Phi Failure Detectors.
    
    Provides centralized health tracking with configurable thresholds.
    """

    def __init__(
        self,
        threshold: float = 8.0,
        min_std_deviation: float = 0.1,
        max_sample_size: int = 100,
    ):
        """
        Initialize the cluster health monitor.
        
        Args:
            threshold: Default Phi threshold for all nodes.
            min_std_deviation: Minimum std deviation for detectors.
            max_sample_size: History size for each detector.
        """
        self.threshold = threshold
        self.min_std_deviation = min_std_deviation
        self.max_sample_size = max_sample_size
        
        self._detectors: Dict[str, PhiFailureDetector] = {}
        self._lock = threading.RLock()

    def register_node(self, node_id: str) -> None:
        """Register a new node for monitoring."""
        with self._lock:
            if node_id not in self._detectors:
                self._detectors[node_id] = PhiFailureDetector(
                    threshold=self.threshold,
                    min_std_deviation=self.min_std_deviation,
                    max_sample_size=self.max_sample_size,
                )

    def unregister_node(self, node_id: str) -> None:
        """Remove a node from monitoring."""
        with self._lock:
            self._detectors.pop(node_id, None)

    def heartbeat(self, node_id: str) -> None:
        """
        Record a heartbeat from a node.
        
        Automatically registers the node if not already tracked.
        """
        with self._lock:
            if node_id not in self._detectors:
                self.register_node(node_id)
            self._detectors[node_id].heartbeat()

    def phi(self, node_id: str) -> float:
        """Get the Phi value for a specific node."""
        with self._lock:
            if node_id not in self._detectors:
                return float('inf')  # Unknown node is suspicious
            return self._detectors[node_id].phi()

    def is_available(self, node_id: str) -> bool:
        """Check if a specific node is available."""
        with self._lock:
            if node_id not in self._detectors:
                return False
            return self._detectors[node_id].is_available()

    def get_node_health(self, node_id: str) -> Optional[NodeHealth]:
        """Get detailed health status for a node."""
        with self._lock:
            detector = self._detectors.get(node_id)
            if detector is None:
                return None
            
            return NodeHealth(
                node_id=node_id,
                phi=detector.phi(),
                is_available=detector.is_available(),
                probability=detector.availability_probability(),
                last_seen=detector.last_heartbeat_time,
                sample_count=detector.sample_count,
                mean_interval=detector.mean_interval,
            )

    def get_all_health(self) -> List[NodeHealth]:
        """Get health status for all monitored nodes."""
        with self._lock:
            return [
                self.get_node_health(node_id)
                for node_id in self._detectors
            ]

    def get_available_nodes(self) -> List[str]:
        """Get list of all available node IDs."""
        with self._lock:
            return [
                node_id
                for node_id, detector in self._detectors.items()
                if detector.is_available()
            ]

    def get_unavailable_nodes(self) -> List[str]:
        """Get list of all unavailable node IDs."""
        with self._lock:
            return [
                node_id
                for node_id, detector in self._detectors.items()
                if not detector.is_available()
            ]

    def get_suspicious_nodes(self, phi_threshold: float = 3.0) -> List[Tuple[str, float]]:
        """
        Get nodes with Phi above the suspicion threshold.
        
        Returns:
            List of (node_id, phi) tuples sorted by phi descending.
        """
        with self._lock:
            suspicious = [
                (node_id, detector.phi())
                for node_id, detector in self._detectors.items()
                if detector.phi() >= phi_threshold
            ]
            return sorted(suspicious, key=lambda x: x[1], reverse=True)

    @property
    def node_count(self) -> int:
        """Get total number of monitored nodes."""
        with self._lock:
            return len(self._detectors)

    @property
    def available_count(self) -> int:
        """Get count of available nodes."""
        return len(self.get_available_nodes())

    @property
    def cluster_health_ratio(self) -> float:
        """Get ratio of available nodes (0.0 to 1.0)."""
        total = self.node_count
        if total == 0:
            return 1.0
        return self.available_count / total

    def print_status(self) -> None:
        """Print a formatted status report."""
        print("\n" + "=" * 60)
        print("CLUSTER HEALTH STATUS (Phi Accrual Failure Detector)")
        print("=" * 60)
        
        for health in self.get_all_health():
            status = "✓ UP" if health.is_available else "✗ DOWN"
            status_color = "\033[92m" if health.is_available else "\033[91m"
            reset = "\033[0m"
            
            print(f"\n  {status_color}{status}{reset} {health.node_id}")
            print(f"      Phi:         {health.phi:.4f}")
            print(f"      Probability: {health.probability:.6f}")
            print(f"      Samples:     {health.sample_count}")
            print(f"      Mean HB:     {health.mean_interval:.4f}s")
        
        print(f"\n  Summary: {self.available_count}/{self.node_count} nodes available")
        print(f"  Health:  {self.cluster_health_ratio:.1%}")
        print("=" * 60 + "\n")


# =============================================================================
# DEMO
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Phi Accrual Failure Detector Demo")
    print("=" * 60)
    
    # Create a detector
    detector = PhiFailureDetector(threshold=8.0)
    
    print("\n--- Simulating Regular Heartbeats ---")
    for i in range(10):
        detector.heartbeat()
        phi = detector.phi()
        status = "UP" if detector.is_available() else "DOWN"
        print(f"  Heartbeat {i+1:2d}: Phi={phi:.4f} [{status}]")
        time.sleep(0.2)
    
    print(f"\n  Mean interval: {detector.mean_interval:.4f}s")
    print(f"  Std deviation: {detector.std_deviation:.4f}s")
    
    print("\n--- Simulating Delayed Heartbeat ---")
    for sec in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        time.sleep(0.5)
        phi = detector.phi()
        prob = detector.availability_probability()
        status = "UP" if detector.is_available() else "DOWN"
        print(f"  After {sec:.1f}s: Phi={phi:.4f} Prob={prob:.6f} [{status}]")
    
    # Cluster monitor demo
    print("\n--- Cluster Health Monitor ---")
    monitor = ClusterHealthMonitor(threshold=8.0)
    
    # Register nodes
    for i in range(5):
        node_id = f"node-{i}"
        monitor.register_node(node_id)
        monitor.heartbeat(node_id)
    
    # Simulate some heartbeats
    for _ in range(5):
        for i in range(5):
            if i != 2:  # Node-2 stops sending heartbeats
                monitor.heartbeat(f"node-{i}")
        time.sleep(0.3)
    
    # Wait a bit more so node-2 becomes suspicious
    time.sleep(1.5)
    
    monitor.print_status()
    
    print("\n" + "=" * 60)
    print("Demo complete")
    print("=" * 60)
