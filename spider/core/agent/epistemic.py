"""
S.P.I.D.E.R. Epistemic Selector - P(True) Sampling
====================================================

Born from: Anthropic-2.2 (Language Models Mostly Know What They Know)

The Anthropic Discovery:
"Large models are well-calibrated. If you ask 'What is the probability
this answer is true?' (P(True)), its self-assessment is surprisingly
accurate."

The Flaw:
Anthropic uses this for training (RLHF), not exposed at inference
because it doubles the cost.

The S.P.I.D.E.R. Implementation:
Replace "Temperature Sampling" with "Truth Sampling."

Mechanism:
1. Generation: Generate N candidate solutions
2. The Probe: For each, fork and ask: "P(True) = ?"
3. Selection: Pick most BELIEVED solution, not most likely

Result: We filter out "plausible but wrong" hallucinations that have
high perplexity but low P(True).
"""

import hashlib
import logging
import re
import statistics
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# EPISTEMIC TYPES
# =============================================================================

class ConfidenceLevel(Enum):
    """Categorical confidence levels."""
    VERY_LOW = auto()       # 0.0 - 0.2
    LOW = auto()            # 0.2 - 0.4
    MEDIUM = auto()         # 0.4 - 0.6
    HIGH = auto()           # 0.6 - 0.8
    VERY_HIGH = auto()      # 0.8 - 1.0


@dataclass
class ConfidenceMeasurement:
    """Result of probing model confidence."""
    code_snippet: str
    raw_response: str
    p_true: float
    confidence_level: ConfidenceLevel
    reasoning: str = ""
    measurement_time: float = 0.0


@dataclass
class CandidateSolution:
    """A candidate solution with confidence score."""
    solution_id: str
    code: str
    p_true: float
    confidence_level: ConfidenceLevel
    generation_log_prob: float = 0.0      # Traditional perplexity
    epistemic_rank: int = 0
    perplexity_rank: int = 0
    discrepancy: float = 0.0              # Difference in ranks


@dataclass
class SelectionResult:
    """Result of epistemic selection."""
    selected: CandidateSolution
    all_candidates: List[CandidateSolution]
    selection_method: str
    confidence_spread: float              # Variance in P(True)
    hallucination_filtered: int           # Count of high-perp/low-conf


# =============================================================================
# CONFIDENCE PROBE
# =============================================================================

class ConfidenceProbe:
    """
    Probes model for epistemic confidence P(True).
    
    Uses the finding from Anthropic-2.2 that models are well-calibrated
    on self-assessment of correctness.
    
    Usage:
        probe = ConfidenceProbe(llm_callback)
        
        measurement = probe.measure_confidence(
            code_snippet="def add(a, b): return a + b",
            problem="Write a function to add two numbers",
        )
        
        print(f"P(True) = {measurement.p_true}")
    """
    
    PROBE_PROMPT = '''You just wrote this code to solve the problem.

PROBLEM:
{problem}

YOUR CODE:
```python
{code}
```

TASK: Evaluate the probability that this code:
1. Correctly solves the problem
2. Handles edge cases
3. Will pass all tests without modification

Output ONLY a single float between 0.0 and 1.0 representing your confidence.
Do not explain. Just the number.

P(True) = '''

    CALIBRATION_PROMPT = '''You are evaluating code correctness.

PROBLEM:
{problem}

CODE:
```python
{code}
```

Rate your confidence that this code is correct:
- 0.0 = Definitely wrong
- 0.3 = Probably wrong
- 0.5 = Uncertain
- 0.7 = Probably correct
- 1.0 = Definitely correct

Consider:
- Does it handle null/empty inputs?
- Are there off-by-one errors?
- Does it match the problem requirements exactly?

Your confidence (float only): '''

    def __init__(
        self,
        llm_callback: Optional[Callable[[str], str]] = None,
        use_calibrated_prompt: bool = True,
    ):
        """
        Initialize Confidence Probe.
        
        Args:
            llm_callback: LLM function for probing
            use_calibrated_prompt: Use more detailed calibration prompt
        """
        self.llm_callback = llm_callback
        self.use_calibrated = use_calibrated_prompt
        
        self._measurements: List[ConfidenceMeasurement] = []
        self._stats = {
            "probes_executed": 0,
            "parse_failures": 0,
            "average_confidence": 0.0,
        }
    
    def measure_confidence(
        self,
        code_snippet: str,
        problem: str,
    ) -> ConfidenceMeasurement:
        """
        Measure model's epistemic confidence in its code.
        
        Args:
            code_snippet: The code to evaluate
            problem: The problem it was supposed to solve
            
        Returns:
            ConfidenceMeasurement with P(True)
        """
        self._stats["probes_executed"] += 1
        start_time = time.time()
        
        if self.llm_callback:
            # Build prompt
            prompt_template = (
                self.CALIBRATION_PROMPT if self.use_calibrated
                else self.PROBE_PROMPT
            )
            prompt = prompt_template.format(
                problem=problem[:1000],
                code=code_snippet[:2000],
            )
            
            # Get response
            response = self.llm_callback(prompt).strip()
            
            # Parse P(True)
            p_true = self._parse_confidence(response)
        else:
            # Heuristic without LLM
            p_true = self._heuristic_confidence(code_snippet)
            response = f"Heuristic: {p_true}"
        
        # Categorize
        confidence_level = self._categorize(p_true)
        
        measurement = ConfidenceMeasurement(
            code_snippet=code_snippet[:500],
            raw_response=response,
            p_true=p_true,
            confidence_level=confidence_level,
            measurement_time=time.time() - start_time,
        )
        
        self._measurements.append(measurement)
        self._update_stats()
        
        return measurement
    
    def measure_batch(
        self,
        candidates: List[str],
        problem: str,
    ) -> List[ConfidenceMeasurement]:
        """Measure confidence for multiple candidates."""
        return [self.measure_confidence(c, problem) for c in candidates]
    
    def _parse_confidence(self, response: str) -> float:
        """Parse P(True) from LLM response."""
        # Try to extract a float
        patterns = [
            r'(\d+\.\d+)',           # 0.85
            r'(\d+)%',               # 85%
            r'(\d+)/(\d+)',          # 85/100
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                if len(match.groups()) == 1:
                    value = float(match.group(1))
                    if value > 1:
                        value = value / 100  # Assume percentage
                    return min(1.0, max(0.0, value))
                elif len(match.groups()) == 2:
                    num = float(match.group(1))
                    denom = float(match.group(2))
                    return min(1.0, max(0.0, num / denom))
        
        # Keyword matching as fallback
        response_lower = response.lower()
        if any(w in response_lower for w in ["definitely correct", "very high", "certain"]):
            return 0.9
        elif any(w in response_lower for w in ["probably correct", "high"]):
            return 0.7
        elif any(w in response_lower for w in ["uncertain", "unsure", "maybe"]):
            return 0.5
        elif any(w in response_lower for w in ["probably wrong", "low"]):
            return 0.3
        elif any(w in response_lower for w in ["definitely wrong", "incorrect"]):
            return 0.1
        
        self._stats["parse_failures"] += 1
        return 0.5  # Default to uncertainty
    
    def _heuristic_confidence(self, code: str) -> float:
        """Estimate confidence using code heuristics (no LLM)."""
        confidence = 0.5
        
        # Positive signals
        if "try:" in code and "except" in code:
            confidence += 0.1
        if "if " in code:
            confidence += 0.05  # Has conditionals
        if "return" in code:
            confidence += 0.05
        if "def " in code:
            confidence += 0.05  # Is a function
        if "assert" in code or "test" in code.lower():
            confidence += 0.1  # Has tests
        
        # Negative signals
        if "TODO" in code or "FIXME" in code:
            confidence -= 0.2
        if "pass" in code and code.count("pass") > code.count("def "):
            confidence -= 0.2  # Empty functions
        if len(code) < 50:
            confidence -= 0.1  # Too short
        if "..." in code:
            confidence -= 0.15  # Placeholder
        
        return min(0.95, max(0.05, confidence))
    
    def _categorize(self, p_true: float) -> ConfidenceLevel:
        """Categorize P(True) into confidence level."""
        if p_true < 0.2:
            return ConfidenceLevel.VERY_LOW
        elif p_true < 0.4:
            return ConfidenceLevel.LOW
        elif p_true < 0.6:
            return ConfidenceLevel.MEDIUM
        elif p_true < 0.8:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH
    
    def _update_stats(self) -> None:
        """Update running statistics."""
        if self._measurements:
            self._stats["average_confidence"] = statistics.mean(
                m.p_true for m in self._measurements
            )
    
    def get_stats(self) -> Dict[str, Any]:
        return self._stats.copy()
    
    def get_calibration_report(self) -> str:
        """Generate a calibration report."""
        if not self._measurements:
            return "No measurements recorded."
        
        by_level = {level: [] for level in ConfidenceLevel}
        for m in self._measurements:
            by_level[m.confidence_level].append(m.p_true)
        
        report = ["Epistemic Calibration Report", "=" * 40]
        for level in ConfidenceLevel:
            measurements = by_level[level]
            if measurements:
                avg = statistics.mean(measurements)
                report.append(f"{level.name}: {len(measurements)} samples, avg={avg:.2f}")
        
        return "\n".join(report)


# =============================================================================
# EPISTEMIC SAMPLER
# =============================================================================

class EpistemicSampler:
    """
    Best-of-N sampling using Epistemic Confidence P(True).
    
    Replaces traditional temperature/perplexity sampling with
    truth-based selection.
    
    From Anthropic-2.2:
    "We pick the most BELIEVED solution, not the most LIKELY."
    
    Usage:
        sampler = EpistemicSampler(generator, llm_callback)
        
        result = sampler.generate_best_of_n(
            problem="Fix the null pointer in auth.py",
            n=5,
        )
        
        print(f"Selected: P(True)={result.selected.p_true}")
        print(f"Filtered {result.hallucination_filtered} hallucinations")
    """
    
    def __init__(
        self,
        solution_generator: Optional[Callable[[str, int], List[str]]] = None,
        llm_callback: Optional[Callable[[str], str]] = None,
        min_confidence: float = 0.3,
    ):
        """
        Initialize Epistemic Sampler.
        
        Args:
            solution_generator: Function to generate N solutions
            llm_callback: LLM for confidence probing
            min_confidence: Minimum P(True) to consider
        """
        self.generator = solution_generator or self._default_generator
        self.probe = ConfidenceProbe(llm_callback)
        self.min_confidence = min_confidence
        
        self._stats = {
            "selections": 0,
            "total_candidates": 0,
            "hallucinations_filtered": 0,
            "confidence_wins": 0,      # Times epistemic != perplexity
        }
    
    def generate_best_of_n(
        self,
        problem: str,
        n: int = 5,
        context: str = "",
    ) -> SelectionResult:
        """
        Generate N solutions and select best by P(True).
        
        Args:
            problem: Problem description
            n: Number of candidates to generate
            context: Additional context
            
        Returns:
            SelectionResult with epistemic selection
        """
        self._stats["selections"] += 1
        
        # Generate candidates
        raw_solutions = self.generator(problem, n)
        self._stats["total_candidates"] += len(raw_solutions)
        
        # Measure confidence for each
        candidates: List[CandidateSolution] = []
        
        for i, code in enumerate(raw_solutions):
            measurement = self.probe.measure_confidence(code, problem)
            
            candidate = CandidateSolution(
                solution_id=f"sol_{i}",
                code=code,
                p_true=measurement.p_true,
                confidence_level=measurement.confidence_level,
                generation_log_prob=1.0 - (i * 0.1),  # Simulate perplexity rank
            )
            candidates.append(candidate)
        
        # Rank by epistemic confidence
        by_confidence = sorted(candidates, key=lambda c: -c.p_true)
        for rank, candidate in enumerate(by_confidence):
            candidate.epistemic_rank = rank
        
        # Rank by perplexity
        by_perplexity = sorted(candidates, key=lambda c: -c.generation_log_prob)
        for rank, candidate in enumerate(by_perplexity):
            candidate.perplexity_rank = rank
            candidate.discrepancy = abs(candidate.epistemic_rank - rank)
        
        # Filter below minimum confidence
        valid_candidates = [c for c in candidates if c.p_true >= self.min_confidence]
        hallucination_count = len(candidates) - len(valid_candidates)
        self._stats["hallucinations_filtered"] += hallucination_count
        
        if not valid_candidates:
            # Take best anyway if all below threshold
            valid_candidates = candidates
        
        # Select best by P(True)
        selected = max(valid_candidates, key=lambda c: c.p_true)
        
        # Track if epistemic != perplexity selection
        perplexity_best = max(candidates, key=lambda c: c.generation_log_prob)
        if selected.solution_id != perplexity_best.solution_id:
            self._stats["confidence_wins"] += 1
        
        # Calculate spread
        confidences = [c.p_true for c in candidates]
        spread = max(confidences) - min(confidences) if confidences else 0
        
        return SelectionResult(
            selected=selected,
            all_candidates=candidates,
            selection_method="epistemic_p_true",
            confidence_spread=spread,
            hallucination_filtered=hallucination_count,
        )
    
    def generate_with_retry(
        self,
        problem: str,
        target_confidence: float = 0.7,
        max_rounds: int = 3,
        n_per_round: int = 3,
    ) -> SelectionResult:
        """
        Generate solutions until confidence threshold met.
        
        Args:
            problem: Problem description
            target_confidence: Required P(True)
            max_rounds: Maximum generation rounds
            n_per_round: Candidates per round
            
        Returns:
            Best result across all rounds
        """
        all_candidates: List[CandidateSolution] = []
        
        for round_num in range(max_rounds):
            result = self.generate_best_of_n(problem, n_per_round)
            all_candidates.extend(result.all_candidates)
            
            if result.selected.p_true >= target_confidence:
                # Meet threshold, return
                return SelectionResult(
                    selected=result.selected,
                    all_candidates=all_candidates,
                    selection_method=f"epistemic_round_{round_num + 1}",
                    confidence_spread=result.confidence_spread,
                    hallucination_filtered=result.hallucination_filtered,
                )
        
        # Return best across all rounds
        best = max(all_candidates, key=lambda c: c.p_true)
        
        return SelectionResult(
            selected=best,
            all_candidates=all_candidates,
            selection_method=f"epistemic_exhausted_{max_rounds}",
            confidence_spread=max(c.p_true for c in all_candidates) - min(c.p_true for c in all_candidates),
            hallucination_filtered=sum(1 for c in all_candidates if c.p_true < self.min_confidence),
        )
    
    def _default_generator(self, problem: str, n: int) -> List[str]:
        """Default solution generator (simulation)."""
        solutions = []
        for i in range(n):
            # Generate variations
            code = f'''def solve_{i}(input_data):
    """Solution attempt {i} for: {problem[:30]}..."""
    # Variation {i}
    result = process(input_data)
    return result
'''
            solutions.append(code)
        return solutions
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            **self._stats,
            **self.probe.get_stats(),
        }
    
    def print_status(self) -> None:
        """Print sampler status."""
        print("\n" + "=" * 60)
        print("[*] EPISTEMIC SAMPLER STATUS")
        print("=" * 60)
        
        print(f"\n[S] Selections: {self._stats['selections']}")
        print(f"[C] Total Candidates: {self._stats['total_candidates']}")
        print(f"[H] Hallucinations Filtered: {self._stats['hallucinations_filtered']}")
        print(f"[W] Confidence Wins: {self._stats['confidence_wins']}")
        
        print(f"\n[P] Probe Stats:")
        for key, val in self.probe.get_stats().items():
            print(f"   {key}: {val}")
        
        print("=" * 60)


# =============================================================================
# PACKAGE INIT
# =============================================================================

__all__ = [
    "EpistemicSampler",
    "ConfidenceProbe",
    "ConfidenceMeasurement",
    "CandidateSolution",
    "SelectionResult",
    "ConfidenceLevel",
]


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("[*] S.P.I.D.E.R. Epistemic Selector - Demo")
    print("=" * 70)
    
    # Test with sample solutions
    solutions = [
        # Good solution
        '''def add(a, b):
    """Add two numbers."""
    if a is None or b is None:
        raise ValueError("Arguments cannot be None")
    return a + b

def test_add():
    assert add(1, 2) == 3
    assert add(-1, 1) == 0
''',
        # Mediocre solution
        '''def add(a, b):
    return a + b
''',
        # Bad solution (placeholder)
        '''def add(a, b):
    # TODO: implement
    pass
''',
        # Hallucination (looks good but wrong)
        '''def add(a, b):
    """Add two numbers using advanced algorithm."""
    return a - b  # Bug: subtracts instead
''',
        # Edge case handler
        '''def add(a, b):
    if not isinstance(a, (int, float)):
        a = 0
    if not isinstance(b, (int, float)):
        b = 0
    return a + b
''',
    ]
    
    probe = ConfidenceProbe()
    
    print("\n[1] Probing confidence for each solution...")
    problem = "Write a function to add two numbers, handling edge cases."
    
    for i, code in enumerate(solutions):
        measurement = probe.measure_confidence(code, problem)
        print(f"\n   Solution {i + 1}: P(True) = {measurement.p_true:.2f} [{measurement.confidence_level.name}]")
        print(f"   Preview: {code[:50].replace(chr(10), ' ')}...")
    
    # Create sampler
    print("\n[2] Testing EpistemicSampler...")
    
    def custom_generator(problem: str, n: int) -> List[str]:
        return solutions[:n]
    
    sampler = EpistemicSampler(
        solution_generator=custom_generator,
        min_confidence=0.3,
    )
    
    result = sampler.generate_best_of_n(problem, n=5)
    
    print(f"\n   Selected solution: P(True) = {result.selected.p_true:.2f}")
    print(f"   Confidence spread: {result.confidence_spread:.2f}")
    print(f"   Hallucinations filtered: {result.hallucination_filtered}")
    
    print("\n[3] All candidates ranked:")
    for c in sorted(result.all_candidates, key=lambda x: -x.p_true):
        arrow = "<-- SELECTED" if c.solution_id == result.selected.solution_id else ""
        print(f"   {c.solution_id}: P(True)={c.p_true:.2f}, Level={c.confidence_level.name} {arrow}")
    
    sampler.print_status()
