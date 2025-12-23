"""
S.P.I.D.E.R. Time Traveler - Monte Carlo Tree Search for Code Generation
==========================================================================

The same algorithm that AlphaGo used to beat Lee Sedol, applied to code.

Key Insight:
  Writing code is a "Game Tree." Every line opens branching possibilities.
  - Standard LLM: Greedy search (picks highest probability token)
  - Time Traveler: Explores 50 steps into the future
  
  If a path leads to a bug 10 steps later, we prune that branch NOW
  and choose a different path.

The Math: UCB1 (Upper Confidence Bound)
  
  UCB1 = X̄_j + C * sqrt(2 * ln(N) / n_j)
  
  Where:
  - X̄_j: Average success rate of this code pattern (Z3 Pass / Total Tries)
  - N: Total simulations run
  - n_j: Times this specific pattern was tried
  - C: Exploration constant (typically sqrt(2))

This balances EXPLOITATION (patterns that work) vs EXPLORATION (new patterns).
"""

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# CODE TEMPLATES - The "Move" Library
# =============================================================================

# These are the "moves" the agent can make at each step
# In a full implementation, these would come from the LLM
CODE_TEMPLATES = {
    'control_flow': [
        "if {var} is not None:",
        "if {var} > 0:",
        "if {var} == 0:",
        "if len({var}) > 0:",
        "for item in {var}:",
        "while {var}:",
        "try:",
        "except Exception as e:",
    ],
    'assignment': [
        "{var} = {var} + 1",
        "{var} = {var} - 1",
        "{var} = {var} * 2",
        "{var} = 0",
        "{var} = []",
        "{var} = None",
        "result = {var}",
    ],
    'return': [
        "return {var}",
        "return None",
        "return result",
        "return True",
        "return False",
        "return 0",
    ],
    'safety': [
        "    raise ValueError('Invalid input')",
        "    return None  # Safe exit",
        "    pass  # Placeholder",
    ],
}

# Variables we track in simulations
DEFAULT_VARS = ['x', 'y', 'n', 'result', 'data']


# =============================================================================
# VERIFICATION RESULT (Simplified)
# =============================================================================

class VerificationOutcome(Enum):
    """Outcome of Z3 verification."""
    SAFE = "safe"       # UNSAT - No bugs possible
    BUGGY = "buggy"     # SAT - Bug found
    UNKNOWN = "unknown" # Timeout or unsupported


# =============================================================================
# CODE NODE - A Node in the Game Tree
# =============================================================================

@dataclass
class CodeNode:
    """
    A node in the Monte Carlo Tree, representing a code state.
    
    Each node is a "game position" where:
    - code_snippet: The code written so far
    - children: Possible next moves (lines of code)
    - visits: How many times we've explored this subtree (n_j)
    - total_score: Sum of all simulation results (for calculating X̄_j)
    
    The game ends when we reach a "terminal" state:
    - return statement
    - maximum depth reached
    - syntax error
    """
    code_snippet: str
    parent: Optional['CodeNode'] = None
    children: List['CodeNode'] = field(default_factory=list)
    visits: int = 0
    total_score: float = 0.0
    untried_moves: List[str] = field(default_factory=list)
    depth: int = 0
    
    def __post_init__(self):
        if not self.untried_moves and self.depth < 10:
            self.untried_moves = self._generate_moves()
    
    def _generate_moves(self) -> List[str]:
        """Generate possible next moves (lines of code)."""
        moves = []
        var = random.choice(DEFAULT_VARS)
        
        # Add some control flow options
        for template in random.sample(CODE_TEMPLATES['control_flow'], min(2, len(CODE_TEMPLATES['control_flow']))):
            moves.append(template.format(var=var))
        
        # Add some assignments
        for template in random.sample(CODE_TEMPLATES['assignment'], min(2, len(CODE_TEMPLATES['assignment']))):
            moves.append(template.format(var=var))
        
        # Add return options (terminal moves)
        for template in random.sample(CODE_TEMPLATES['return'], min(2, len(CODE_TEMPLATES['return']))):
            moves.append(template.format(var=var))
        
        return moves
    
    @property
    def average_score(self) -> float:
        """Calculate X̄_j (average success rate)."""
        if self.visits == 0:
            return 0.0
        return self.total_score / self.visits
    
    def is_fully_expanded(self) -> bool:
        """Check if all possible moves have been tried."""
        return len(self.untried_moves) == 0
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal state (game over)."""
        code_lower = self.code_snippet.lower()
        return (
            'return' in code_lower or
            self.depth >= 10 or
            'raise' in code_lower
        )
    
    def expand(self) -> 'CodeNode':
        """
        Expand the tree by trying an untried move.
        
        Returns:
            A new child node representing the result of the move.
        """
        if not self.untried_moves:
            raise ValueError("No untried moves left!")
        
        # Pick a random untried move
        move = self.untried_moves.pop(random.randint(0, len(self.untried_moves) - 1))
        
        # Create new code by appending the move
        indent = "    " * (self.depth + 1)
        new_code = f"{self.code_snippet}\n{indent}{move}"
        
        # Create child node
        child = CodeNode(
            code_snippet=new_code,
            parent=self,
            depth=self.depth + 1,
        )
        self.children.append(child)
        
        return child
    
    def best_child(self, exploration_weight: float = 1.414) -> 'CodeNode':
        """
        Select the best child using UCB1 formula.
        
        UCB1 = X̄_j + C * sqrt(2 * ln(N) / n_j)
        
        Args:
            exploration_weight: The C constant (default: sqrt(2) ≈ 1.414)
            
        Returns:
            The child with the highest UCB1 score.
        """
        if not self.children:
            raise ValueError("No children to select from!")
        
        def ucb1(child: 'CodeNode') -> float:
            if child.visits == 0:
                return float('inf')  # Always try unvisited nodes
            
            exploitation = child.average_score
            exploration = exploration_weight * math.sqrt(
                2 * math.log(self.visits) / child.visits
            )
            return exploitation + exploration
        
        return max(self.children, key=ucb1)
    
    def best_move(self) -> 'CodeNode':
        """
        Select the best move (child with most visits).
        
        This is the "move" we would actually make - the most explored path.
        """
        if not self.children:
            raise ValueError("No children!")
        return max(self.children, key=lambda c: c.visits)
    
    def __repr__(self) -> str:
        return f"CodeNode(depth={self.depth}, visits={self.visits}, score={self.average_score:.2f})"


# =============================================================================
# TIME TRAVELER - The MCTS Engine
# =============================================================================

class TimeTraveler:
    """
    Monte Carlo Tree Search for Code Generation.
    
    Explores the "future" of code generation paths and picks the one
    most likely to pass Z3 verification.
    
    Usage:
        traveler = TimeTraveler("def divide(a, b):")
        best_code = traveler.search(iterations=100)
        print(best_code.code_snippet)
    
    The Algorithm:
    1. SELECTION: Traverse down using UCB1 until we find an unexpanded node
    2. EXPANSION: Generate possible next lines of code
    3. SIMULATION: Random rollout to completion
    4. BACKPROPAGATION: Update scores based on Z3 verification
    """
    
    def __init__(
        self,
        root_code: str,
        exploration_weight: float = 1.414,  # sqrt(2)
        max_depth: int = 10,
    ):
        """
        Initialize the Time Traveler.
        
        Args:
            root_code: The starting code (e.g., function signature)
            exploration_weight: UCB1 exploration constant C
            max_depth: Maximum depth of the tree
        """
        self.root = CodeNode(code_snippet=root_code)
        self.exploration_weight = exploration_weight
        self.max_depth = max_depth
        
        # Statistics
        self._stats = {
            'iterations': 0,
            'simulations': 0,
            'safe_outcomes': 0,
            'buggy_outcomes': 0,
            'unknown_outcomes': 0,
            'max_depth_reached': 0,
        }
    
    def search(self, iterations: int = 50) -> CodeNode:
        """
        Run MCTS for the specified number of iterations.
        
        Each iteration:
        1. SELECT a path down the tree using UCB1
        2. EXPAND by adding a new child node
        3. SIMULATE a random rollout to completion
        4. BACKPROPAGATE the result up the tree
        
        Args:
            iterations: Number of MCTS iterations to run
            
        Returns:
            The best CodeNode (most visited child of root)
        """
        for i in range(iterations):
            self._stats['iterations'] += 1
            
            # 1. SELECTION
            node = self._select(self.root)
            
            # 2. EXPANSION
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()
            
            # 3. SIMULATION
            outcome = self._simulate(node)
            
            # 4. BACKPROPAGATION
            self._backpropagate(node, outcome)
        
        # Return the most visited child (the "best move")
        if self.root.children:
            return self.root.best_move()
        return self.root
    
    def _select(self, node: CodeNode) -> CodeNode:
        """
        Selection phase: traverse down using UCB1.
        
        Keep going until we find a node that:
        - Is terminal (game over)
        - Has untried moves (needs expansion)
        """
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return node  # Found a node to expand
            elif node.children:
                node = node.best_child(self.exploration_weight)
            else:
                return node  # No children and fully expanded
        return node
    
    def _simulate(self, node: CodeNode) -> VerificationOutcome:
        """
        Simulation phase: random rollout to completion.
        
        From the current node, randomly pick moves until we reach
        a terminal state, then evaluate with Z3.
        """
        self._stats['simulations'] += 1
        
        # Build the complete code by random rollout
        code = node.code_snippet
        depth = node.depth
        
        while depth < self.max_depth:
            if 'return' in code.lower() or 'raise' in code.lower():
                break
            
            # Random rollout - pick a random move
            var = random.choice(DEFAULT_VARS)
            category = random.choice(list(CODE_TEMPLATES.keys()))
            template = random.choice(CODE_TEMPLATES[category])
            move = template.format(var=var)
            
            indent = "    " * (depth + 1)
            code = f"{code}\n{indent}{move}"
            depth += 1
        
        if depth >= self.max_depth:
            self._stats['max_depth_reached'] += 1
        
        # Evaluate the code with our mock Z3 verifier
        return self._evaluate(code)
    
    def _evaluate(self, code: str) -> VerificationOutcome:
        """
        Evaluate code using Z3 verification (simplified mock).
        
        In a real implementation, this would:
        1. Parse the code into AST
        2. Convert to SMT formulas
        3. Check for bugs (division by zero, null dereference, etc.)
        
        For this mock, we use pattern matching:
        - SAFE: Has safety checks (if != 0, if is not None, try/except)
        - BUGGY: Direct division, accessing without checks
        - UNKNOWN: Cannot determine
        """
        code_lower = code.lower()
        
        # Safety patterns that make code SAFE (UNSAT in Z3 terms)
        safety_patterns = [
            'if' in code_lower and 'none' in code_lower,
            'if' in code_lower and '!= 0' in code_lower,
            'if' in code_lower and '> 0' in code_lower,
            'try:' in code_lower,
            'raise' in code_lower,
            'len(' in code_lower and 'if' in code_lower,
        ]
        
        # Bug patterns that make code BUGGY (SAT in Z3 terms)
        bug_patterns = [
            '/ 0' in code_lower,
            'none' in code_lower and '.' in code_lower and 'if' not in code_lower,
            '[]' in code_lower and '[0]' in code_lower,
        ]
        
        # Check for bugs first
        if any(bug_patterns):
            self._stats['buggy_outcomes'] += 1
            return VerificationOutcome.BUGGY
        
        # Check for safety
        if any(safety_patterns):
            self._stats['safe_outcomes'] += 1
            return VerificationOutcome.SAFE
        
        # Unknown - couldn't determine
        self._stats['unknown_outcomes'] += 1
        return VerificationOutcome.UNKNOWN
    
    def _backpropagate(self, node: CodeNode, outcome: VerificationOutcome):
        """
        Backpropagation phase: update scores up to root.
        
        Score mapping:
        - SAFE: +1.0 (reward)
        - BUGGY: -1.0 (penalty)
        - UNKNOWN: 0.0 (neutral)
        """
        score = {
            VerificationOutcome.SAFE: 1.0,
            VerificationOutcome.BUGGY: -1.0,
            VerificationOutcome.UNKNOWN: 0.0,
        }[outcome]
        
        # Walk up to the root, updating each node
        current = node
        while current is not None:
            current.visits += 1
            current.total_score += score
            current = current.parent
    
    def print_tree(self, node: Optional[CodeNode] = None, indent: int = 0):
        """Print the tree structure for debugging."""
        if node is None:
            node = self.root
        
        prefix = "  " * indent
        score_str = f"{node.average_score:+.2f}" if node.visits > 0 else "N/A"
        print(f"{prefix}├── {node} [score={score_str}]")
        
        # Print snippet preview
        snippet = node.code_snippet.split('\n')[-1][:40]
        print(f"{prefix}│   '{snippet}...'")
        
        for child in node.children[:3]:  # Limit to first 3 children
            self.print_tree(child, indent + 1)
        
        if len(node.children) > 3:
            print(f"{prefix}│   ... and {len(node.children) - 3} more children")
    
    @property
    def stats(self) -> Dict[str, int]:
        """Get MCTS statistics."""
        return self._stats.copy()
    
    def print_stats(self):
        """Print MCTS statistics."""
        print("\n📊 Time Traveler Statistics:")
        print(f"   Iterations:     {self._stats['iterations']}")
        print(f"   Simulations:    {self._stats['simulations']}")
        print(f"   Safe outcomes:  {self._stats['safe_outcomes']}")
        print(f"   Buggy outcomes: {self._stats['buggy_outcomes']}")
        print(f"   Unknown:        {self._stats['unknown_outcomes']}")


# =============================================================================
# DEMO
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("🕰️ S.P.I.D.E.R. TIME TRAVELER - MCTS Demo")
    print("=" * 60)
    print("\nThe Algorithm:")
    print("  UCB1 = X̄_j + C * sqrt(2 * ln(N) / n_j)")
    print("  Where:")
    print("    X̄_j = Average success rate (Z3 Pass / Total)")
    print("    N   = Total simulations")
    print("    n_j = Times this pattern was tried")
    print()
    
    # Create a Time Traveler starting with a function signature
    root_code = "def safe_divide(a, b):"
    print(f"Starting Code: {root_code}")
    print()
    
    traveler = TimeTraveler(root_code, exploration_weight=1.414)
    
    # Run MCTS for 100 iterations
    print("Running MCTS with 100 iterations...")
    best = traveler.search(iterations=100)
    
    print("\n" + "=" * 60)
    print("BEST PATH FOUND:")
    print("=" * 60)
    print(best.code_snippet)
    print()
    
    print(f"Path Statistics:")
    print(f"  Depth:  {best.depth}")
    print(f"  Visits: {best.visits}")
    print(f"  Score:  {best.average_score:+.2f}")
    
    traveler.print_stats()
    
    print("\n" + "=" * 60)
    print("TREE STRUCTURE (first 3 levels):")
    print("=" * 60)
    traveler.print_tree()
