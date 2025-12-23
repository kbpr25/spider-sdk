"""
S.P.I.D.E.R. Truth Engine Package
=================================

The Manhattan Project of Reliability.

This package transforms S.P.I.D.E.R. from "Generative AI" to "Verifiable AI"
by implementing 4 anti-hallucination modules:

1. ProvenanceTracker - Citation-enforced generation
   Born from: Assumption-1 (RAG), Assumption-2 (REALM), Assumption-6 (Attribution)
   
2. EpistemicValve - Self-aware confidence throttle
   Born from: Assumption-3 (Calibration), Assumption-4 (Verbalized Uncertainty)
   
3. ReflexionEngine - Multi-path consensus with self-critique
   Born from: Assumption-8 (Self-Consistency), Assumption-9 (Reflexion)
   
4. LogicalEntailmentShield - Adversarial truth filter
   Born from: Assumption-5 (Faithful Reasoning), Assumption-10 (TruthfulQA)

Usage:
    from spider.core.truth_engine import TruthEngine
    
    engine = TruthEngine()
    engine.add_source("doc1", "Paris is the capital of France.")
    
    result = engine.process(
        question="What is the capital of France?",
        llm_callback=my_llm_function,
    )
    
    if result.is_truthful:
        print(result.final_answer)
"""

from .provenance import (
    ProvenanceTracker,
    ProvenanceResult,
    ProvenanceStatus,
    Citation,
    SourceDocument,
)

from .epistemic import (
    EpistemicValve,
    EpistemicResult,
    ConfidenceAssessment,
    ConfidenceLevel,
    TemperatureScaler,
)

from .reflexion import (
    ReflexionEngine,
    ReflexionResult,
    ReasoningPath,
    ConsensusResult,
    ReasoningStatus,
)

from .entailment import (
    LogicalEntailmentShield,
    ShieldResult,
    EntailmentResult,
    EntailmentStatus,
    AtomicClaim,
    ClaimType,
)

from .engine import (
    TruthEngine,
    TruthEngineResult,
)


__all__ = [
    # Main engine
    "TruthEngine",
    "TruthEngineResult",
    
    # Provenance
    "ProvenanceTracker",
    "ProvenanceResult",
    "ProvenanceStatus",
    "Citation",
    "SourceDocument",
    
    # Epistemic
    "EpistemicValve",
    "EpistemicResult",
    "ConfidenceAssessment",
    "ConfidenceLevel",
    "TemperatureScaler",
    
    # Reflexion
    "ReflexionEngine",
    "ReflexionResult",
    "ReasoningPath",
    "ConsensusResult",
    "ReasoningStatus",
    
    # Entailment
    "LogicalEntailmentShield",
    "ShieldResult",
    "EntailmentResult",
    "EntailmentStatus",
    "AtomicClaim",
    "ClaimType",
]
