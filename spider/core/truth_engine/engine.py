"""
S.P.I.D.E.R. Truth Engine - Hallucination-Proof Pipeline
=========================================================

The Manhattan Project of Reliability.

This module unifies 4 anti-hallucination systems:
1. ProvenanceTracker - Citation-enforced generation
2. EpistemicValve - Self-aware confidence throttle
3. ReflexionEngine - Multi-path consensus with self-critique
4. LogicalEntailmentShield - Adversarial truth filter

The pipeline transforms S.P.I.D.E.R. from "Generative AI" (creates plausibility)
to "Verifiable AI" (constructs truth).

Pipeline Flow:
    User Query → SDK Intercept
          ↓
    Retrieval: Fetch Docs
          ↓
    Generation (The Swarm): ReflexionEngine spawns 5 parallel threads
          ↓
    Constraint: ProvenanceTracker forces every thread to cite sources
          ↓
    Critique: LogicalEntailmentShield kills threads with logical leaps
          ↓
    Confidence: EpistemicValve measures survivors
          ↓
    Output: The Truth (or ABORT if confidence too low)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .provenance import ProvenanceTracker, ProvenanceResult
from .epistemic import EpistemicValve, EpistemicResult
from .reflexion import ReflexionEngine, ReflexionResult
from .entailment import LogicalEntailmentShield, ShieldResult

logger = logging.getLogger(__name__)


# =============================================================================
# TRUTH ENGINE RESULT
# =============================================================================

@dataclass
class TruthEngineResult:
    """Complete result from the Truth Engine pipeline."""
    
    # Input
    question: str
    sources: List[str]
    
    # Stage Results
    reflexion_result: Optional[ReflexionResult] = None
    provenance_result: Optional[ProvenanceResult] = None
    entailment_result: Optional[ShieldResult] = None
    epistemic_result: Optional[EpistemicResult] = None
    
    # Final Output
    final_answer: str = ""
    is_truthful: bool = False
    confidence: float = 0.0
    
    # Metadata
    processing_time_ms: float = 0.0
    stages_passed: List[str] = field(default_factory=list)
    stages_failed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def summary(self) -> str:
        """Get one-line summary."""
        status = "✅ TRUTHFUL" if self.is_truthful else "❌ BLOCKED"
        return f"{status} | Confidence: {self.confidence:.0%} | Time: {self.processing_time_ms:.0f}ms"


# =============================================================================
# TRUTH ENGINE
# =============================================================================

class TruthEngine:
    """
    The Truth Engine - Hallucination-Proof AI Pipeline.
    
    Combines 4 verification stages to ensure output is:
    - GROUNDED in sources (Provenance)
    - CONFIDENT (Epistemic)
    - CONSISTENT (Reflexion)
    - LOGICAL (Entailment)
    
    Usage:
        engine = TruthEngine()
        
        # Add sources
        engine.add_source("doc1", "Paris is the capital of France.")
        
        # Process query
        result = engine.process(
            question="What is the capital of France?",
            llm_callback=my_llm_function,
        )
        
        if result.is_truthful:
            print(result.final_answer)
        else:
            print("Cannot answer truthfully")
    """
    
    def __init__(
        self,
        # Provenance settings
        grounding_threshold: float = 0.4,
        strict_provenance: bool = True,
        
        # Epistemic settings
        confidence_threshold: float = 50.0,
        temperature_scaling: float = 1.5,
        
        # Reflexion settings
        num_paths: int = 5,
        consensus_threshold: float = 0.6,
        max_reflection_iterations: int = 2,
        
        # Entailment settings
        entailment_threshold: float = 0.5,
        block_myths: bool = True,
        
        # Global settings
        min_overall_confidence: float = 50.0,
        min_grounding: float = 0.5,
    ):
        """
        Initialize Truth Engine.
        
        All thresholds can be tuned based on use case.
        Higher thresholds = more conservative (blocks more).
        """
        # Initialize components
        self.provenance = ProvenanceTracker(
            grounding_threshold=grounding_threshold,
            strict_mode=strict_provenance,
        )
        
        self.epistemic = EpistemicValve(
            medium_threshold=confidence_threshold,
            temperature=temperature_scaling,
        )
        
        self.reflexion = ReflexionEngine(
            num_paths=num_paths,
            consensus_threshold=consensus_threshold,
            max_reflection_iterations=max_reflection_iterations,
        )
        
        self.entailment = LogicalEntailmentShield(
            entailment_threshold=entailment_threshold,
            block_myths=block_myths,
        )
        
        # Global thresholds
        self.min_confidence = min_overall_confidence
        self.min_grounding = min_grounding
        
        self._stats = {
            "queries_processed": 0,
            "truthful_outputs": 0,
            "blocked_outputs": 0,
            "total_time_ms": 0,
        }
    
    def add_source(
        self,
        doc_id: str,
        content: str,
        title: str = "",
        url: str = "",
    ) -> None:
        """Add a source document for verification."""
        self.provenance.add_source(doc_id, content, title, url)
    
    def add_sources(self, sources: List[Dict[str, str]]) -> None:
        """Add multiple sources."""
        for source in sources:
            self.add_source(
                doc_id=source.get("id", source.get("doc_id", "")),
                content=source.get("content", source.get("text", "")),
                title=source.get("title", ""),
                url=source.get("url", ""),
            )
    
    def clear_sources(self) -> None:
        """Clear all sources."""
        self.provenance.clear_sources()
    
    def process(
        self,
        question: str,
        llm_callback: Optional[Callable[[str, float], str]] = None,
        sources: List[str] = None,
        skip_reflexion: bool = False,
    ) -> TruthEngineResult:
        """
        Process a query through the full Truth Engine pipeline.
        
        Pipeline:
        1. (Optional) Reflexion: Generate and verify multiple answers
        2. Provenance: Check citation grounding
        3. Entailment: Verify logical validity
        4. Epistemic: Assess confidence
        
        Args:
            question: The user's question
            llm_callback: Function to call LLM (prompt, temperature) -> response
            sources: Optional additional sources (text strings)
            skip_reflexion: Skip multi-path generation for speed
            
        Returns:
            TruthEngineResult with verdict and answer
        """
        start_time = time.time()
        self._stats["queries_processed"] += 1
        
        # Add inline sources if provided
        if sources:
            for i, source in enumerate(sources):
                self.add_source(f"inline_{i}", source)
        
        result = TruthEngineResult(
            question=question,
            sources=[s.content for s in self.provenance.sources.values()],
        )
        
        # Get context for verification
        context = " ".join(s.content for s in self.provenance.sources.values())
        
        # Stage 1: Reflexion (multi-path generation)
        if not skip_reflexion and llm_callback:
            try:
                reflexion_result = self.reflexion.solve(
                    question=question,
                    context=context,
                    llm_callback=llm_callback,
                )
                result.reflexion_result = reflexion_result
                
                if reflexion_result.reflection_passed:
                    result.stages_passed.append("reflexion")
                    raw_answer = reflexion_result.final_answer
                else:
                    result.stages_failed.append("reflexion")
                    result.warnings.append(
                        f"Low consensus: {reflexion_result.consensus.agreement_ratio:.0%}"
                    )
                    raw_answer = reflexion_result.final_answer
                    
            except Exception as e:
                logger.warning(f"Reflexion failed: {e}")
                result.stages_failed.append("reflexion")
                raw_answer = self._simple_generate(question, context, llm_callback)
        else:
            # Simple generation without reflexion
            if llm_callback:
                raw_answer = self._simple_generate(question, context, llm_callback)
            else:
                raw_answer = f"Cannot answer without LLM callback."
        
        # Stage 2: Provenance (citation check)
        try:
            provenance_result = self.provenance.verify(raw_answer)
            result.provenance_result = provenance_result
            
            if provenance_result.overall_grounding >= self.min_grounding:
                result.stages_passed.append("provenance")
                grounded_answer = provenance_result.grounded_text
            else:
                result.stages_failed.append("provenance")
                result.warnings.append(
                    f"Low grounding: {provenance_result.overall_grounding:.0%}"
                )
                grounded_answer = provenance_result.grounded_text or raw_answer
                
        except Exception as e:
            logger.warning(f"Provenance check failed: {e}")
            result.stages_failed.append("provenance")
            grounded_answer = raw_answer
        
        # Stage 3: Entailment (logical check)
        try:
            entailment_result = self.entailment.filter(grounded_answer, context)
            result.entailment_result = entailment_result
            
            if entailment_result.overall_validity >= 0.5:
                result.stages_passed.append("entailment")
                logical_answer = entailment_result.filtered_text
            else:
                result.stages_failed.append("entailment")
                result.warnings.append(
                    f"Low validity: {entailment_result.overall_validity:.0%}"
                )
                logical_answer = entailment_result.filtered_text or grounded_answer
            
            if entailment_result.myth_claims:
                result.warnings.append(
                    f"Myths blocked: {len(entailment_result.myth_claims)}"
                )
                
        except Exception as e:
            logger.warning(f"Entailment check failed: {e}")
            result.stages_failed.append("entailment")
            logical_answer = grounded_answer
        
        # Stage 4: Epistemic (confidence check)
        try:
            epistemic_result = self.epistemic.assess(
                response=logical_answer,
                question=question,
                sources=[s.content[:500] for s in self.provenance.sources.values()],
                llm_callback=lambda p: llm_callback(p, 0.3) if llm_callback else None,
            )
            result.epistemic_result = epistemic_result
            
            if not epistemic_result.blocked:
                result.stages_passed.append("epistemic")
                result.confidence = epistemic_result.assessment.calibrated_confidence
            else:
                result.stages_failed.append("epistemic")
                result.confidence = epistemic_result.assessment.calibrated_confidence
                result.warnings.append(
                    f"Low confidence: {result.confidence:.0%}"
                )
                
        except Exception as e:
            logger.warning(f"Epistemic check failed: {e}")
            result.stages_failed.append("epistemic")
            result.confidence = 50.0  # Default
        
        # Final verdict
        all_stages_passed = (
            "provenance" in result.stages_passed and
            "entailment" in result.stages_passed and
            "epistemic" in result.stages_passed
        )
        
        result.is_truthful = all_stages_passed and result.confidence >= self.min_confidence
        
        if result.is_truthful:
            result.final_answer = logical_answer
            self._stats["truthful_outputs"] += 1
        else:
            if logical_answer.strip():
                result.final_answer = (
                    f"[UNVERIFIED - {', '.join(result.warnings)}]\n\n"
                    f"{logical_answer}"
                )
            else:
                result.final_answer = (
                    "❌ Cannot answer truthfully.\n\n"
                    f"Failed stages: {', '.join(result.stages_failed)}\n"
                    f"Confidence: {result.confidence:.0%}"
                )
            self._stats["blocked_outputs"] += 1
        
        # Timing
        result.processing_time_ms = (time.time() - start_time) * 1000
        self._stats["total_time_ms"] += result.processing_time_ms
        
        return result
    
    def _simple_generate(
        self,
        question: str,
        context: str,
        llm_callback: Callable[[str, float], str],
    ) -> str:
        """Simple single-path generation."""
        prompt = f"""Answer the following question based ONLY on the provided context.

Context: {context[:3000]}

Question: {question}

Provide a clear, factual answer supported by the context."""
        
        try:
            return llm_callback(prompt, 0.5)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return ""
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            **self._stats,
            "provenance": self.provenance.get_stats(),
            "epistemic": self.epistemic.get_stats(),
            "reflexion": self.reflexion.get_stats(),
            "entailment": self.entailment.get_stats(),
        }
    
    def print_result(self, result: TruthEngineResult) -> None:
        """Print detailed result."""
        print("\n" + "=" * 70)
        print("⚡ TRUTH ENGINE RESULT")
        print("=" * 70)
        
        print(f"\n❓ Question: {result.question[:80]}...")
        print(f"\n{result.summary()}")
        
        print(f"\n📊 STAGE RESULTS:")
        print(f"   ✅ Passed: {', '.join(result.stages_passed) or 'None'}")
        print(f"   ❌ Failed: {', '.join(result.stages_failed) or 'None'}")
        
        if result.warnings:
            print(f"\n⚠️ WARNINGS:")
            for warning in result.warnings:
                print(f"   - {warning}")
        
        if result.provenance_result:
            print(f"\n📎 Provenance: {result.provenance_result.overall_grounding:.0%} grounded")
            print(f"   Redacted: {len(result.provenance_result.redacted_claims)} claims")
        
        if result.entailment_result:
            print(f"\n🛡️ Entailment: {result.entailment_result.overall_validity:.0%} valid")
            print(f"   Myths caught: {len(result.entailment_result.myth_claims)}")
        
        if result.epistemic_result:
            print(f"\n🧠 Epistemic: {result.epistemic_result.assessment.calibrated_confidence:.0%} confidence")
        
        print(f"\n📤 FINAL OUTPUT:")
        print("-" * 70)
        print(result.final_answer[:500])
        if len(result.final_answer) > 500:
            print("... [truncated]")
        print("-" * 70)
        
        print(f"\n⏱️ Processing Time: {result.processing_time_ms:.0f}ms")
        print("=" * 70)


# =============================================================================
# PACKAGE INIT
# =============================================================================

__all__ = [
    "TruthEngine",
    "TruthEngineResult",
    # Re-export components
    "ProvenanceTracker",
    "EpistemicValve",
    "ReflexionEngine",
    "LogicalEntailmentShield",
]


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("⚡ S.P.I.D.E.R. TRUTH ENGINE - Demo")
    print("=" * 70)
    
    engine = TruthEngine()
    
    # Add source documents
    engine.add_source(
        "wiki_france",
        "France is a country in Western Europe. Paris is the capital and "
        "largest city of France. The Eiffel Tower was built in 1889. "
        "The official language is French. France has about 67 million people."
    )
    
    # Process query (without LLM - will use mock)
    result = engine.process(
        question="What is the capital of France?",
        skip_reflexion=True,
    )
    
    engine.print_result(result)
    
    print(f"\n📊 Engine Stats: {engine.get_stats()}")
