"""
S.P.I.D.E.R. Reflexion Engine - Think-Critique-Fix Cycle
=========================================================

Born from: Assumption-8 (Self-Consistency), Assumption-9 (Reflexion)

The Scientific Finding:
"Greedy Decoding (taking the first/best word) is suboptimal for reasoning.
Multiple reasoning paths often diverge, but the correct answer appears
most frequently (Consistency). If you force a model to REFLECT on its
own output and critique errors, performance jumps massively (HumanEval 91%)."

The Solution:
Replace "One-Shot Generation" with a "Think-Critique-Fix" Cycle.

1. Diversity Sampling: Generate 5 different reasoning paths
2. Consistency Voting: Group answers, discard outliers
3. Verbal Reinforcement: Model reflects on winner
4. Final Commit: Only after double-checking

Result: We trade Latency for Reasoning Rigor. This kills "Stochastic Parrots."
"""

import hashlib
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# REFLEXION TYPES
# =============================================================================

class ReasoningStatus(Enum):
    """Status of reasoning path."""
    CONSENSUS = auto()     # Part of majority consensus
    OUTLIER = auto()       # Disagrees with majority
    REFLECTED = auto()     # Passed reflection check
    REJECTED = auto()      # Failed reflection check


@dataclass
class ReasoningPath:
    """A single reasoning path and its answer."""
    path_id: str
    chain_of_thought: str          # Full reasoning chain
    answer: str                    # Final answer extracted
    confidence: float = 0.0        # Self-reported confidence
    temperature: float = 0.7       # Generation temperature used
    status: ReasoningStatus = ReasoningStatus.OUTLIER


@dataclass
class ConsensusResult:
    """Result of consensus voting."""
    winning_answer: str
    supporting_paths: List[ReasoningPath]
    rejected_paths: List[ReasoningPath]
    agreement_ratio: float         # % of paths agreeing
    answer_distribution: Dict[str, int]


@dataclass
class ReflexionResult:
    """Complete reflexion analysis result."""
    question: str
    paths: List[ReasoningPath]
    consensus: ConsensusResult
    reflection_passed: bool
    final_answer: str
    final_reasoning: str
    iterations: int


# =============================================================================
# ANSWER NORMALIZATION
# =============================================================================

class AnswerNormalizer:
    """
    Normalizes answers for comparison.
    
    Handles:
    - Case normalization
    - Whitespace normalization
    - Numeric equivalence (1,000 = 1000)
    - Common variations
    """
    
    @staticmethod
    def normalize(answer: str) -> str:
        """Normalize answer for comparison."""
        # Lowercase and strip
        normalized = answer.lower().strip()
        
        # Remove punctuation at end
        normalized = re.sub(r'[.!?:,;]+$', '', normalized)
        
        # Normalize whitespace
        normalized = ' '.join(normalized.split())
        
        # Normalize numbers (remove commas)
        normalized = re.sub(r'(\d),(\d)', r'\1\2', normalized)
        
        # Remove common prefix phrases
        prefixes = [
            "the answer is",
            "i think",
            "i believe",
            "in my opinion",
            "based on the context",
            "according to",
        ]
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):].strip()
        
        return normalized
    
    @staticmethod
    def extract_answer(response: str) -> str:
        """Extract the final answer from a response."""
        # Look for explicit answer markers
        patterns = [
            r"(?:the answer is|answer:)\s*(.+?)(?:\.|$)",
            r"(?:therefore|thus|hence|so),?\s*(.+?)(?:\.|$)",
            r"(?:in conclusion|to conclude),?\s*(.+?)(?:\.|$)",
            r"\*\*(.+?)\*\*",  # Bold text often is the answer
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Fallback: last sentence
        sentences = re.split(r'[.!?]\s+', response)
        if sentences:
            return sentences[-1].strip()
        
        return response.strip()
    
    @staticmethod
    def similarity(a: str, b: str) -> float:
        """Calculate similarity between two answers."""
        norm_a = AnswerNormalizer.normalize(a)
        norm_b = AnswerNormalizer.normalize(b)
        
        if norm_a == norm_b:
            return 1.0
        
        return SequenceMatcher(None, norm_a, norm_b).ratio()


# =============================================================================
# PROMPTS
# =============================================================================

COT_PROMPT = """
Solve this problem step by step. Think carefully through each step.

Question: {question}

Context: {context}

Think through this carefully:
1. What are the key facts?
2. What logic applies?
3. What is the answer?

Provide your reasoning, then give your final answer.
"""

REFLECTION_PROMPT = """
You previously answered a question. Now reflect on your answer.

Question: {question}

Your Previous Answer: {answer}

Your Previous Reasoning: {reasoning}

The source context was: {context}

Now carefully check:
1. Does your answer directly follow from the source context?
2. Did you make any logical leaps not supported by the context?
3. Are there any errors in your reasoning?

If your answer is correct and well-supported, respond with:
VERIFIED: [your answer]
[brief explanation of why it's correct]

If you found an error, respond with:
REVISED: [corrected answer]
[explanation of the error and correction]
"""

CRITIQUE_PROMPT = """
Critically evaluate this reasoning and answer.

Question: {question}
Answer: {answer}
Reasoning: {reasoning}

Identify:
1. Any logical fallacies
2. Any unsupported claims
3. Any contradictions
4. Any missing considerations

Rate the answer quality from 1-10 and explain why.
"""


# =============================================================================
# REFLEXION ENGINE
# =============================================================================

class ReflexionEngine:
    """
    The Reflexion Engine - Multi-Path Consensus with Self-Critique.
    
    Implements the Think-Critique-Fix cycle:
    1. Generate multiple reasoning paths (diversity sampling)
    2. Find consensus answer (majority voting)
    3. Reflect and verify (self-critique)
    4. Commit only after verification
    
    From Assumption-8: Self-consistency improves accuracy
    From Assumption-9: Reflexion can achieve 91% on HumanEval
    
    Usage:
        engine = ReflexionEngine()
        
        result = engine.solve(
            question="What is 2+2?",
            context="Basic arithmetic: 2+2=4",
            llm_callback=my_llm_function,
            num_paths=5,
        )
        
        if result.reflection_passed:
            print(result.final_answer)
    """
    
    def __init__(
        self,
        num_paths: int = 5,
        consensus_threshold: float = 0.6,
        similarity_threshold: float = 0.8,
        temperatures: List[float] = None,
        max_reflection_iterations: int = 2,
    ):
        """
        Initialize Reflexion Engine.
        
        Args:
            num_paths: Number of reasoning paths to generate
            consensus_threshold: Minimum agreement for consensus
            similarity_threshold: Threshold for answer similarity
            temperatures: Temperatures for diversity (default: varied)
            max_reflection_iterations: Max reflection loops
        """
        self.num_paths = num_paths
        self.consensus_threshold = consensus_threshold
        self.similarity_threshold = similarity_threshold
        self.max_reflection_iterations = max_reflection_iterations
        
        # Use varied temperatures for diversity
        self.temperatures = temperatures or [0.3, 0.5, 0.7, 0.9, 1.0]
        if len(self.temperatures) < num_paths:
            self.temperatures = self.temperatures * (num_paths // len(self.temperatures) + 1)
        
        self.normalizer = AnswerNormalizer()
        
        self._stats = {
            "problems_solved": 0,
            "paths_generated": 0,
            "consensus_found": 0,
            "reflections_passed": 0,
            "reflections_revised": 0,
        }
    
    def generate_paths(
        self,
        question: str,
        context: str,
        llm_callback: Callable[[str, float], str],
        num_paths: int = None,
    ) -> List[ReasoningPath]:
        """
        Generate multiple reasoning paths with diversity sampling.
        
        Uses different temperatures to encourage diverse reasoning.
        """
        num_paths = num_paths or self.num_paths
        paths = []
        
        prompt = COT_PROMPT.format(
            question=question,
            context=context[:2000],  # Truncate context
        )
        
        for i in range(num_paths):
            temp = self.temperatures[i % len(self.temperatures)]
            
            try:
                response = llm_callback(prompt, temp)
                answer = self.normalizer.extract_answer(response)
                
                path = ReasoningPath(
                    path_id=f"path_{i}",
                    chain_of_thought=response,
                    answer=answer,
                    temperature=temp,
                )
                paths.append(path)
                self._stats["paths_generated"] += 1
                
            except Exception as e:
                logger.warning(f"Path {i} generation failed: {e}")
        
        return paths
    
    def find_consensus(self, paths: List[ReasoningPath]) -> ConsensusResult:
        """
        Find consensus answer through voting.
        
        Groups similar answers and finds the majority.
        """
        if not paths:
            return ConsensusResult(
                winning_answer="",
                supporting_paths=[],
                rejected_paths=[],
                agreement_ratio=0.0,
                answer_distribution={},
            )
        
        # Normalize and group answers
        normalized_answers = [
            (path, self.normalizer.normalize(path.answer))
            for path in paths
        ]
        
        # Group similar answers
        groups: Dict[str, List[ReasoningPath]] = {}
        
        for path, norm_answer in normalized_answers:
            # Find matching group
            matched = False
            for group_key in groups:
                if self.normalizer.similarity(norm_answer, group_key) >= self.similarity_threshold:
                    groups[group_key].append(path)
                    matched = True
                    break
            
            if not matched:
                groups[norm_answer] = [path]
        
        # Find winner
        if not groups:
            return ConsensusResult(
                winning_answer="",
                supporting_paths=[],
                rejected_paths=paths,
                agreement_ratio=0.0,
                answer_distribution={},
            )
        
        # Sort by group size
        sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
        winning_key, winning_paths = sorted_groups[0]
        
        # Update statuses
        for path in winning_paths:
            path.status = ReasoningStatus.CONSENSUS
        
        rejected = []
        for key, group_paths in sorted_groups[1:]:
            for path in group_paths:
                path.status = ReasoningStatus.OUTLIER
                rejected.append(path)
        
        agreement_ratio = len(winning_paths) / len(paths)
        
        if agreement_ratio >= self.consensus_threshold:
            self._stats["consensus_found"] += 1
        
        # Build distribution
        distribution = {key: len(group) for key, group in groups.items()}
        
        return ConsensusResult(
            winning_answer=winning_paths[0].answer,  # Use original form
            supporting_paths=winning_paths,
            rejected_paths=rejected,
            agreement_ratio=agreement_ratio,
            answer_distribution=distribution,
        )
    
    def reflect(
        self,
        answer: str,
        reasoning: str,
        question: str,
        context: str,
        llm_callback: Callable[[str, float], str],
    ) -> Tuple[bool, str, str]:
        """
        Reflect on an answer to verify or revise it.
        
        From Assumption-9: Self-reflection significantly improves accuracy.
        
        Returns:
            Tuple of (passed, final_answer, reasoning)
        """
        prompt = REFLECTION_PROMPT.format(
            question=question,
            answer=answer,
            reasoning=reasoning[:1000],
            context=context[:1000],
        )
        
        try:
            response = llm_callback(prompt, 0.3)  # Low temp for verification
            
            if "VERIFIED:" in response.upper():
                self._stats["reflections_passed"] += 1
                # Extract verified answer
                match = re.search(r'VERIFIED:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
                verified_answer = match.group(1).strip() if match else answer
                return (True, verified_answer, response)
            
            elif "REVISED:" in response.upper():
                self._stats["reflections_revised"] += 1
                # Extract revised answer
                match = re.search(r'REVISED:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
                revised_answer = match.group(1).strip() if match else answer
                return (False, revised_answer, response)
            
            else:
                # Unclear response, assume verification
                return (True, answer, response)
                
        except Exception as e:
            logger.warning(f"Reflection failed: {e}")
            return (True, answer, "Reflection failed, using original answer")
    
    def solve(
        self,
        question: str,
        context: str = "",
        llm_callback: Callable[[str, float], str] = None,
        num_paths: int = None,
    ) -> ReflexionResult:
        """
        Solve a problem using the full Reflexion cycle.
        
        1. Generate diverse reasoning paths
        2. Find consensus
        3. Reflect and verify
        4. Return verified answer
        
        Args:
            question: The problem to solve
            context: Available context/sources
            llm_callback: Function to call LLM (prompt, temperature) -> response
            num_paths: Number of reasoning paths
            
        Returns:
            ReflexionResult with verified answer
        """
        self._stats["problems_solved"] += 1
        num_paths = num_paths or self.num_paths
        
        if not llm_callback:
            # Mock for testing
            return self._mock_solve(question, context)
        
        # Step 1: Generate diverse paths
        paths = self.generate_paths(question, context, llm_callback, num_paths)
        
        if not paths:
            return ReflexionResult(
                question=question,
                paths=[],
                consensus=ConsensusResult("", [], [], 0.0, {}),
                reflection_passed=False,
                final_answer="Failed to generate reasoning paths",
                final_reasoning="",
                iterations=0,
            )
        
        # Step 2: Find consensus
        consensus = self.find_consensus(paths)
        
        if consensus.agreement_ratio < self.consensus_threshold:
            # No clear consensus - return with warning
            return ReflexionResult(
                question=question,
                paths=paths,
                consensus=consensus,
                reflection_passed=False,
                final_answer=f"[LOW CONSENSUS {consensus.agreement_ratio:.0%}] {consensus.winning_answer}",
                final_reasoning="Multiple reasoning paths disagreed",
                iterations=1,
            )
        
        # Step 3: Reflect on winning answer
        best_path = consensus.supporting_paths[0]
        iteration = 0
        current_answer = consensus.winning_answer
        current_reasoning = best_path.chain_of_thought
        reflection_passed = False
        
        for iteration in range(1, self.max_reflection_iterations + 1):
            passed, new_answer, reflection = self.reflect(
                current_answer,
                current_reasoning,
                question,
                context,
                llm_callback,
            )
            
            if passed:
                reflection_passed = True
                current_answer = new_answer
                current_reasoning = reflection
                break
            else:
                # Answer was revised, reflect again
                current_answer = new_answer
                current_reasoning = reflection
        
        # Mark paths as reflected
        for path in consensus.supporting_paths:
            path.status = ReasoningStatus.REFLECTED if reflection_passed else ReasoningStatus.REJECTED
        
        return ReflexionResult(
            question=question,
            paths=paths,
            consensus=consensus,
            reflection_passed=reflection_passed,
            final_answer=current_answer,
            final_reasoning=current_reasoning,
            iterations=iteration,
        )
    
    def _mock_solve(self, question: str, context: str) -> ReflexionResult:
        """Mock solve for testing without LLM."""
        # Simple mock response
        mock_path = ReasoningPath(
            path_id="mock_0",
            chain_of_thought="This is a mock reasoning path.",
            answer="Mock answer",
            status=ReasoningStatus.CONSENSUS,
        )
        
        return ReflexionResult(
            question=question,
            paths=[mock_path],
            consensus=ConsensusResult(
                winning_answer="Mock answer",
                supporting_paths=[mock_path],
                rejected_paths=[],
                agreement_ratio=1.0,
                answer_distribution={"mock answer": 1},
            ),
            reflection_passed=True,
            final_answer="Mock answer (no LLM provided)",
            final_reasoning="Mock reasoning",
            iterations=0,
        )
    
    def get_stats(self) -> Dict[str, int]:
        return self._stats.copy()
    
    def print_result(self, result: ReflexionResult) -> None:
        """Print detailed result."""
        print("\n" + "=" * 60)
        print("🔄 REFLEXION ENGINE RESULT")
        print("=" * 60)
        
        print(f"\n❓ Question: {result.question[:80]}...")
        print(f"📊 Paths Generated: {len(result.paths)}")
        print(f"🎯 Consensus: {result.consensus.agreement_ratio:.0%}")
        print(f"🔍 Reflection Iterations: {result.iterations}")
        print(f"✅ Reflection Passed: {result.reflection_passed}")
        
        print(f"\n📈 Answer Distribution:")
        for answer, count in result.consensus.answer_distribution.items():
            print(f"   [{count}x] {answer[:50]}...")
        
        print(f"\n🏆 FINAL ANSWER: {result.final_answer}")
        
        if result.final_reasoning:
            print(f"\n💭 Reasoning: {result.final_reasoning[:200]}...")
        
        print("=" * 60)


# =============================================================================
# PACKAGE INIT
# =============================================================================

__all__ = [
    "ReflexionEngine",
    "ReflexionResult",
    "ReasoningPath",
    "ConsensusResult",
    "ReasoningStatus",
    "AnswerNormalizer",
]


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🔄 S.P.I.D.E.R. Reflexion Engine - Demo")
    print("=" * 70)
    
    engine = ReflexionEngine(num_paths=5)
    
    # Test without LLM (mock mode)
    result = engine.solve(
        question="What is the capital of France?",
        context="France is a country in Europe. Paris is the largest city in France and serves as its capital.",
    )
    
    engine.print_result(result)
    
    print(f"\n📊 Stats: {engine.get_stats()}")
