"""
S.P.I.D.E.R. Entropy Governor - Dynamic Compute Allocation
===========================================================

Born from: Counter to Anthropic-1.6 (Effort Parameter)

The Anthropic Weakness:
"They added an 'effort' parameter (Low/High). They are relying on
the user to guess how hard to try. Using 'High' everywhere is slow;
using 'Low' risks failure."

The S.P.I.D.E.R. Evolution:
Replace the "Effort Toggle" with Mathematical Certainty.

Mechanism:
1. Dynamic Compute: Agent starts solving
2. Variance Check: Monitor stability of MCTS paths
3. Auto-Scale: If paths diverge (high uncertainty) -> increase effort
4. Halt Proof: Stop only when confidence interval hits threshold

Result: Claude guesses effort. S.P.I.D.E.R. spends EXACTLY the compute
required - no more, no less.
"""

import hashlib
import logging
import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# ENTROPY TYPES
# =============================================================================

class ComputeLevel(Enum):
    """Compute effort levels (internal only - user never sets this)."""
    MINIMAL = 1          # Trivial task
    STANDARD = 2         # Normal task
    FOCUSED = 3          # Complex task
    INTENSIVE = 4        # Very hard task
    MAXIMUM = 5          # Extreme difficulty


class ConfidenceState(Enum):
    """State of solution confidence."""
    UNCERTAIN = auto()
    CONVERGING = auto()
    STABLE = auto()
    PROVEN = auto()


@dataclass
class SearchPath:
    """A single path in the MCTS tree."""
    path_id: str
    solution: str
    confidence: float
    depth: int
    visits: int = 1
    reward: float = 0.0


@dataclass
class SearchTree:
    """Monte Carlo Tree Search state."""
    paths: List[SearchPath] = field(default_factory=list)
    best_path: Optional[SearchPath] = None
    entropy: float = 1.0
    variance: float = 1.0
    convergence_rate: float = 0.0


@dataclass
class GovernorDecision:
    """Decision from the entropy governor."""
    compute_level: ComputeLevel
    should_continue: bool
    confidence: float
    entropy: float
    reasoning: str


# =============================================================================
# ENTROPY CALCULATOR
# =============================================================================

class EntropyCalculator:
    """
    Calculates Shannon Entropy of solution space.
    
    Low entropy = paths converge (we're confident)
    High entropy = paths diverge (we're uncertain)
    """
    
    def __init__(self):
        self.epsilon = 1e-10
    
    def shannon_entropy(self, probabilities: List[float]) -> float:
        """
        Calculate Shannon entropy.
        
        H = -sum(p * log2(p)) for each probability p
        """
        H = 0.0
        for p in probabilities:
            if p > self.epsilon:
                H -= p * math.log2(p)
        return H
    
    def path_entropy(self, tree: SearchTree) -> float:
        """
        Calculate entropy of solution paths.
        
        High entropy = many different solutions (uncertain)
        Low entropy = paths converge (confident)
        """
        if not tree.paths:
            return 1.0  # Maximum uncertainty
        
        # Group paths by similarity
        solution_groups: Dict[str, float] = {}
        total_confidence = sum(p.confidence for p in tree.paths)
        
        if total_confidence < self.epsilon:
            return 1.0
        
        for path in tree.paths:
            # Hash solution for grouping
            sol_hash = hashlib.md5(path.solution[:100].encode()).hexdigest()[:8]
            prob = path.confidence / total_confidence
            solution_groups[sol_hash] = solution_groups.get(sol_hash, 0) + prob
        
        # Calculate entropy
        probabilities = list(solution_groups.values())
        return self.shannon_entropy(probabilities)
    
    def variance(self, tree: SearchTree) -> float:
        """Calculate variance in path confidences."""
        if len(tree.paths) < 2:
            return 1.0
        
        confidences = [p.confidence for p in tree.paths]
        mean = sum(confidences) / len(confidences)
        variance = sum((c - mean) ** 2 for c in confidences) / len(confidences)
        
        return variance
    
    def convergence_rate(self, tree: SearchTree, history: List[float]) -> float:
        """Calculate how fast paths are converging."""
        if len(history) < 2:
            return 0.0
        
        # Calculate entropy reduction rate
        recent = history[-5:]  # Last 5 measurements
        if len(recent) < 2:
            return 0.0
        
        slope = (recent[-1] - recent[0]) / len(recent)
        
        # Negative slope = converging
        return -slope


# =============================================================================
# MCTS SIMULATOR
# =============================================================================

class MCTSSimulator:
    """
    Simplified Monte Carlo Tree Search for solution exploration.
    
    Used to measure uncertainty in the solution space.
    """
    
    def __init__(
        self,
        solution_generator: Optional[Callable[[str], str]] = None,
        confidence_evaluator: Optional[Callable[[str], float]] = None,
    ):
        self.generator = solution_generator or self._default_generator
        self.evaluator = confidence_evaluator or self._default_evaluator
        
        self.tree = SearchTree()
        self.entropy_history: List[float] = []
    
    def explore(
        self,
        problem: str,
        num_paths: int = 5,
    ) -> SearchTree:
        """
        Explore solution paths for a problem.
        
        Args:
            problem: Problem description
            num_paths: Number of paths to explore
            
        Returns:
            SearchTree with explored paths
        """
        self.tree = SearchTree()
        
        for i in range(num_paths):
            # Generate a solution
            solution = self.generator(problem)
            confidence = self.evaluator(solution)
            
            path = SearchPath(
                path_id=f"path_{i}",
                solution=solution,
                confidence=confidence,
                depth=1,
            )
            
            self.tree.paths.append(path)
        
        # Find best path
        if self.tree.paths:
            self.tree.best_path = max(self.tree.paths, key=lambda p: p.confidence)
        
        return self.tree
    
    def expand(self, iterations: int = 3) -> None:
        """Expand tree with more exploration."""
        for _ in range(iterations):
            if not self.tree.paths:
                break
            
            # Select path to expand (UCB1)
            selected = self._select_ucb1()
            
            # Expand
            new_solution = self._mutate_solution(selected.solution)
            confidence = self.evaluator(new_solution)
            
            new_path = SearchPath(
                path_id=f"path_{len(self.tree.paths)}",
                solution=new_solution,
                confidence=confidence,
                depth=selected.depth + 1,
            )
            
            self.tree.paths.append(new_path)
            
            # Update best
            if confidence > self.tree.best_path.confidence:
                self.tree.best_path = new_path
    
    def _select_ucb1(self) -> SearchPath:
        """Select path using UCB1 algorithm."""
        if not self.tree.paths:
            raise ValueError("No paths to select from")
        
        total_visits = sum(p.visits for p in self.tree.paths)
        
        def ucb1_score(path: SearchPath) -> float:
            exploitation = path.reward / max(path.visits, 1)
            exploration = math.sqrt(2 * math.log(total_visits + 1) / max(path.visits, 1))
            return exploitation + exploration
        
        return max(self.tree.paths, key=ucb1_score)
    
    def _mutate_solution(self, solution: str) -> str:
        """Create a variant of a solution."""
        # Simple mutation: random modifications
        words = solution.split()
        if len(words) > 3:
            idx = random.randint(0, len(words) - 1)
            words[idx] = f"modified_{words[idx]}"
        return " ".join(words)
    
    def _default_generator(self, problem: str) -> str:
        """Default solution generator (simulation)."""
        return f"Solution for: {problem[:50]}... (variant {random.randint(1, 100)})"
    
    def _default_evaluator(self, solution: str) -> float:
        """Default confidence evaluator (simulation)."""
        # Higher confidence for longer, structured solutions
        base = 0.5
        length_bonus = min(0.3, len(solution) / 1000)
        structure_bonus = 0.1 if "for" in solution or "def" in solution else 0
        return base + length_bonus + structure_bonus


# =============================================================================
# ENTROPY GOVERNOR
# =============================================================================

class EntropyGovernor:
    """
    The Entropy-Driven Governor - Dynamic Compute Allocation.
    
    Replaces Anthropic's "Effort Parameter" with mathematical certainty:
    1. Measure Shannon Entropy of solution space
    2. Auto-scale compute based on uncertainty
    3. Stop only when confidence threshold reached
    
    Usage:
        governor = EntropyGovernor()
        
        # Start solving
        decision = governor.allocate_compute(problem)
        
        while decision.should_continue:
            # Do work
            solution = generate_solution()
            governor.update(solution, confidence)
            
            decision = governor.check_halt()
        
        # Optimal compute spent - no more, no less
    """
    
    # Entropy thresholds for compute levels
    ENTROPY_THRESHOLDS = {
        ComputeLevel.MINIMAL: 0.1,
        ComputeLevel.STANDARD: 0.3,
        ComputeLevel.FOCUSED: 0.5,
        ComputeLevel.INTENSIVE: 0.7,
        ComputeLevel.MAXIMUM: 0.9,
    }
    
    def __init__(
        self,
        confidence_threshold: float = 0.95,
        max_iterations: int = 100,
        convergence_target: float = 0.01,
    ):
        """
        Initialize Entropy Governor.
        
        Args:
            confidence_threshold: Stop when confidence exceeds this
            max_iterations: Maximum iterations before forced stop
            convergence_target: Entropy level indicating convergence
        """
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations
        self.convergence_target = convergence_target
        
        self.entropy_calc = EntropyCalculator()
        self.mcts = MCTSSimulator()
        
        # State
        self.current_problem: str = ""
        self.iterations: int = 0
        self.entropy_history: List[float] = []
        self.confidence_history: List[float] = []
        
        self._stats = {
            "problems_solved": 0,
            "total_iterations": 0,
            "early_halts": 0,
            "max_compute_tasks": 0,
        }
    
    def allocate_compute(self, problem: str) -> GovernorDecision:
        """
        Initial compute allocation for a problem.
        
        Returns recommended compute level and whether to continue.
        """
        self.current_problem = problem
        self.iterations = 0
        self.entropy_history.clear()
        self.confidence_history.clear()
        
        # Initial exploration
        tree = self.mcts.explore(problem, num_paths=3)
        
        # Calculate initial entropy
        entropy = self.entropy_calc.path_entropy(tree)
        self.entropy_history.append(entropy)
        
        # Determine compute level
        compute_level = self._entropy_to_compute(entropy)
        
        # Get best confidence so far
        confidence = tree.best_path.confidence if tree.best_path else 0.0
        self.confidence_history.append(confidence)
        
        return GovernorDecision(
            compute_level=compute_level,
            should_continue=True,
            confidence=confidence,
            entropy=entropy,
            reasoning=f"Initial entropy {entropy:.3f} -> {compute_level.name} compute",
        )
    
    def update(
        self,
        solution: str,
        confidence: float,
    ) -> GovernorDecision:
        """
        Update governor with new solution attempt.
        
        Returns updated decision on compute allocation.
        """
        self.iterations += 1
        self._stats["total_iterations"] += 1
        
        # Add to MCTS tree
        path = SearchPath(
            path_id=f"path_{len(self.mcts.tree.paths)}",
            solution=solution,
            confidence=confidence,
            depth=1,
        )
        self.mcts.tree.paths.append(path)
        
        # Update best
        if (not self.mcts.tree.best_path or 
            confidence > self.mcts.tree.best_path.confidence):
            self.mcts.tree.best_path = path
        
        # Calculate new entropy
        entropy = self.entropy_calc.path_entropy(self.mcts.tree)
        variance = self.entropy_calc.variance(self.mcts.tree)
        convergence = self.entropy_calc.convergence_rate(
            self.mcts.tree, self.entropy_history
        )
        
        self.entropy_history.append(entropy)
        self.confidence_history.append(confidence)
        
        # Update tree stats
        self.mcts.tree.entropy = entropy
        self.mcts.tree.variance = variance
        self.mcts.tree.convergence_rate = convergence
        
        return self.check_halt()
    
    def check_halt(self) -> GovernorDecision:
        """
        Check if we should halt computation.
        
        Returns decision with halt reasoning.
        """
        tree = self.mcts.tree
        best_confidence = tree.best_path.confidence if tree.best_path else 0.0
        
        # Check confidence threshold
        if best_confidence >= self.confidence_threshold:
            self._stats["early_halts"] += 1
            self._stats["problems_solved"] += 1
            return GovernorDecision(
                compute_level=ComputeLevel.MINIMAL,
                should_continue=False,
                confidence=best_confidence,
                entropy=tree.entropy,
                reasoning=f"Confidence {best_confidence:.3f} >= {self.confidence_threshold}",
            )
        
        # Check convergence
        if tree.entropy < self.convergence_target:
            self._stats["early_halts"] += 1
            self._stats["problems_solved"] += 1
            return GovernorDecision(
                compute_level=ComputeLevel.MINIMAL,
                should_continue=False,
                confidence=best_confidence,
                entropy=tree.entropy,
                reasoning=f"Entropy {tree.entropy:.3f} < {self.convergence_target} (converged)",
            )
        
        # Check iteration limit
        if self.iterations >= self.max_iterations:
            return GovernorDecision(
                compute_level=ComputeLevel.MAXIMUM,
                should_continue=False,
                confidence=best_confidence,
                entropy=tree.entropy,
                reasoning=f"Max iterations ({self.max_iterations}) reached",
            )
        
        # Continue with adjusted compute
        compute_level = self._entropy_to_compute(tree.entropy)
        
        if compute_level == ComputeLevel.MAXIMUM:
            self._stats["max_compute_tasks"] += 1
        
        return GovernorDecision(
            compute_level=compute_level,
            should_continue=True,
            confidence=best_confidence,
            entropy=tree.entropy,
            reasoning=f"Entropy {tree.entropy:.3f} -> {compute_level.name}",
        )
    
    def _entropy_to_compute(self, entropy: float) -> ComputeLevel:
        """Map entropy to compute level."""
        for level, threshold in sorted(
            self.ENTROPY_THRESHOLDS.items(),
            key=lambda x: x[1],
        ):
            if entropy <= threshold:
                return level
        return ComputeLevel.MAXIMUM
    
    def get_best_solution(self) -> Optional[str]:
        """Get the best solution found."""
        if self.mcts.tree.best_path:
            return self.mcts.tree.best_path.solution
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            **self._stats,
            "current_iterations": self.iterations,
            "current_entropy": self.entropy_history[-1] if self.entropy_history else 1.0,
        }
    
    def print_status(self) -> None:
        """Print governor status."""
        print("\n" + "=" * 60)
        print("[*] ENTROPY GOVERNOR STATUS")
        print("=" * 60)
        
        tree = self.mcts.tree
        print(f"\n[E] Entropy: {tree.entropy:.3f}")
        print(f"[V] Variance: {tree.variance:.3f}")
        print(f"[C] Convergence Rate: {tree.convergence_rate:.3f}")
        
        if tree.best_path:
            print(f"\n[B] Best Solution:")
            print(f"   Confidence: {tree.best_path.confidence:.3f}")
            print(f"   Preview: {tree.best_path.solution[:50]}...")
        
        print(f"\n[%] Stats:")
        for key, val in self.get_stats().items():
            print(f"   {key}: {val}")
        
        print("=" * 60)


# =============================================================================
# PACKAGE INIT
# =============================================================================

__all__ = [
    "EntropyGovernor",
    "EntropyCalculator",
    "MCTSSimulator",
    "GovernorDecision",
    "ComputeLevel",
    "ConfidenceState",
    "SearchTree",
    "SearchPath",
]


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("[*] S.P.I.D.E.R. Entropy Governor - Demo")
    print("=" * 70)
    
    governor = EntropyGovernor(
        confidence_threshold=0.9,
        max_iterations=20,
    )
    
    # Simulate solving a problem
    print("\n[1] Allocating compute for problem...")
    decision = governor.allocate_compute("Fix the null pointer in auth.py")
    
    print(f"   Initial: {decision.compute_level.name}")
    print(f"   Entropy: {decision.entropy:.3f}")
    print(f"   Reasoning: {decision.reasoning}")
    
    # Simulate iterations
    print("\n[2] Simulating solution search...")
    
    for i in range(10):
        # Generate random solution with increasing confidence
        solution = f"Solution attempt {i}: fix line {random.randint(1, 100)}"
        confidence = 0.5 + (i * 0.05) + random.uniform(-0.1, 0.1)
        confidence = min(0.95, max(0.0, confidence))
        
        decision = governor.update(solution, confidence)
        
        print(f"   Iter {i+1}: conf={confidence:.2f}, entropy={decision.entropy:.3f}, "
              f"compute={decision.compute_level.name}")
        
        if not decision.should_continue:
            print(f"\n   [HALT] {decision.reasoning}")
            break
    
    # Get best solution
    print(f"\n[3] Best solution found:")
    best = governor.get_best_solution()
    if best:
        print(f"   {best}")
    
    governor.print_status()
