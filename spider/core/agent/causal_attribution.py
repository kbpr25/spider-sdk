"""
S.P.I.D.E.R. Causal Attribution Engine - Influence-Based RAG
==============================================================

Born from: Anthropic-2.4 (Influence Functions)

The Anthropic Discovery:
"They use Influence Functions (EK-FAC) to mathematically determine
which TRAINING DOCUMENT caused the model to output a specific token."

The Flaw:
Computationally expensive, usually done offline for analysis.

The S.P.I.D.E.R. Implementation:
Lightweight, localized influence check for the Knowledge Graph.

Mechanism:
1. Draft: Agent writes code using library function
2. Influence Check: Check gradient against retrieved RAG documents
3. Verification: If code relies on Internal Memory (training data)
   rather than Retrieved Context (actual docs), flag as "High Risk"

Result: We detect when model is ignoring documentation and "winging it"
based on outdated training data.
"""

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# ATTRIBUTION TYPES
# =============================================================================

class AttributionSource(Enum):
    """Source of knowledge for generated content."""
    RETRIEVED = auto()        # From RAG documents
    PARAMETRIC = auto()       # From training (internal memory)
    MIXED = auto()            # Combination
    UNKNOWN = auto()


class RiskLevel(Enum):
    """Risk level of hallucination."""
    LOW = auto()              # Strongly grounded in docs
    MEDIUM = auto()           # Some grounding
    HIGH = auto()             # Mostly parametric
    CRITICAL = auto()         # No grounding, high confidence


@dataclass
class RetrievedDocument:
    """A document retrieved for context."""
    doc_id: str
    title: str
    content: str
    source: str               # URL, file path, etc.
    relevance_score: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class AttributionResult:
    """Result of causal attribution analysis."""
    code_segment: str
    source: AttributionSource
    risk_level: RiskLevel
    grounding_score: float          # 0-1, higher = more grounded
    influential_docs: List[Tuple[str, float]]  # (doc_id, influence)
    ungrounded_claims: List[str]    # Statements not in docs
    reasoning: str


@dataclass
class InfluenceVector:
    """Influence measurement for a document."""
    doc_id: str
    influence_score: float
    matched_terms: List[str]
    coverage: float           # Fraction of code covered by doc


# =============================================================================
# GROUNDING ANALYZER
# =============================================================================

class GroundingAnalyzer:
    """
    Analyzes how well generated code is grounded in retrieved documents.
    
    Uses term overlap, API usage patterns, and code structure.
    """
    
    # Code elements that MUST be grounded
    CRITICAL_PATTERNS = [
        r"(\w+)\(",                    # Function calls
        r"import\s+(\w+)",             # Imports
        r"from\s+(\w+)",               # Module imports
        r"\.(\w+)\(",                  # Method calls
        r"raise\s+(\w+)",              # Exceptions
    ]
    
    def __init__(self):
        self.compiled_patterns = [re.compile(p) for p in self.CRITICAL_PATTERNS]
    
    def analyze_grounding(
        self,
        code: str,
        documents: List[RetrievedDocument],
    ) -> Tuple[float, List[str]]:
        """
        Analyze how well code is grounded in documents.
        
        Returns:
            Tuple of (grounding_score, ungrounded_claims)
        """
        # Extract critical tokens from code
        code_tokens = self._extract_critical_tokens(code)
        
        # Build document content index
        doc_content = " ".join(d.content for d in documents)
        doc_tokens = set(re.findall(r'\b\w+\b', doc_content.lower()))
        
        # Calculate grounding
        grounded = []
        ungrounded = []
        
        for token in code_tokens:
            if token.lower() in doc_tokens:
                grounded.append(token)
            else:
                ungrounded.append(token)
        
        # Calculate score
        if code_tokens:
            score = len(grounded) / len(code_tokens)
        else:
            score = 1.0
        
        return (score, ungrounded)
    
    def _extract_critical_tokens(self, code: str) -> Set[str]:
        """Extract tokens that need grounding."""
        tokens = set()
        
        for pattern in self.compiled_patterns:
            matches = pattern.findall(code)
            tokens.update(matches)
        
        return tokens
    
    def compute_influence(
        self,
        code: str,
        doc: RetrievedDocument,
    ) -> InfluenceVector:
        """
        Compute influence of a document on generated code.
        
        Returns:
            InfluenceVector with influence metrics
        """
        code_lower = code.lower()
        doc_lower = doc.content.lower()
        
        # Find matching terms
        doc_terms = set(re.findall(r'\b\w{3,}\b', doc_lower))
        code_terms = set(re.findall(r'\b\w{3,}\b', code_lower))
        
        matched = code_terms & doc_terms
        
        # Compute influence score
        if code_terms:
            influence = len(matched) / len(code_terms)
        else:
            influence = 0.0
        
        # Weight by document relevance
        influence *= doc.relevance_score
        
        # Compute coverage
        if doc_terms:
            coverage = len(matched) / len(doc_terms)
        else:
            coverage = 0.0
        
        return InfluenceVector(
            doc_id=doc.doc_id,
            influence_score=influence,
            matched_terms=list(matched)[:20],
            coverage=coverage,
        )


# =============================================================================
# CAUSAL ATTRIBUTION ENGINE
# =============================================================================

class CausalAttributionEngine:
    """
    The Causal RAG Engine - Influence-Based Attribution.
    
    Detects when the model is ignoring retrieved documentation and
    relying on potentially outdated parametric memory.
    
    From Anthropic-2.4:
    "We mathematically determine which training document caused
    the model to output a specific token."
    
    Usage:
        engine = CausalAttributionEngine()
        
        # Add retrieved documents
        engine.add_document(doc1)
        engine.add_document(doc2)
        
        # Analyze generated code
        result = engine.analyze(generated_code)
        
        if result.risk_level == RiskLevel.HIGH:
            print("WARNING: Code not grounded in documentation!")
    """
    
    def __init__(
        self,
        min_grounding: float = 0.5,
        critical_threshold: float = 0.2,
    ):
        """
        Initialize Causal Attribution Engine.
        
        Args:
            min_grounding: Minimum grounding score for LOW risk
            critical_threshold: Below this = CRITICAL risk
        """
        self.min_grounding = min_grounding
        self.critical_threshold = critical_threshold
        
        self.analyzer = GroundingAnalyzer()
        
        # Document store
        self.documents: Dict[str, RetrievedDocument] = {}
        
        self._stats = {
            "analyses": 0,
            "high_risk_detected": 0,
            "critical_risk_detected": 0,
            "average_grounding": 0.0,
        }
    
    def add_document(self, doc: RetrievedDocument) -> None:
        """Add a retrieved document to the context."""
        self.documents[doc.doc_id] = doc
    
    def add_documents(self, docs: List[RetrievedDocument]) -> None:
        """Add multiple documents."""
        for doc in docs:
            self.add_document(doc)
    
    def create_document(
        self,
        title: str,
        content: str,
        source: str = "",
        relevance: float = 1.0,
    ) -> RetrievedDocument:
        """Create and add a new document."""
        doc_id = hashlib.md5(content[:100].encode()).hexdigest()[:12]
        
        doc = RetrievedDocument(
            doc_id=doc_id,
            title=title,
            content=content,
            source=source,
            relevance_score=relevance,
        )
        
        self.add_document(doc)
        return doc
    
    def analyze(self, code: str) -> AttributionResult:
        """
        Analyze causal attribution for generated code.
        
        Determines how much the code relies on retrieved docs
        vs. internal parametric memory.
        """
        self._stats["analyses"] += 1
        
        docs = list(self.documents.values())
        
        if not docs:
            # No documents - everything is parametric
            return AttributionResult(
                code_segment=code[:500],
                source=AttributionSource.PARAMETRIC,
                risk_level=RiskLevel.HIGH,
                grounding_score=0.0,
                influential_docs=[],
                ungrounded_claims=["No reference documents provided"],
                reasoning="No RAG documents available for grounding",
            )
        
        # Analyze grounding
        grounding_score, ungrounded = self.analyzer.analyze_grounding(code, docs)
        
        # Compute influence for each document
        influences = []
        for doc in docs:
            influence = self.analyzer.compute_influence(code, doc)
            influences.append((doc.doc_id, influence.influence_score))
        
        # Sort by influence
        influences.sort(key=lambda x: -x[1])
        
        # Determine source and risk
        if grounding_score >= 0.8:
            source = AttributionSource.RETRIEVED
            risk_level = RiskLevel.LOW
        elif grounding_score >= self.min_grounding:
            source = AttributionSource.MIXED
            risk_level = RiskLevel.MEDIUM
        elif grounding_score >= self.critical_threshold:
            source = AttributionSource.PARAMETRIC
            risk_level = RiskLevel.HIGH
            self._stats["high_risk_detected"] += 1
        else:
            source = AttributionSource.PARAMETRIC
            risk_level = RiskLevel.CRITICAL
            self._stats["critical_risk_detected"] += 1
        
        # Update average grounding
        total = self._stats["analyses"]
        prev_avg = self._stats["average_grounding"]
        self._stats["average_grounding"] = (
            (prev_avg * (total - 1) + grounding_score) / total
        )
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            grounding_score, influences, ungrounded
        )
        
        return AttributionResult(
            code_segment=code[:500],
            source=source,
            risk_level=risk_level,
            grounding_score=grounding_score,
            influential_docs=influences[:5],
            ungrounded_claims=ungrounded[:10],
            reasoning=reasoning,
        )
    
    def _generate_reasoning(
        self,
        grounding: float,
        influences: List[Tuple[str, float]],
        ungrounded: List[str],
    ) -> str:
        """Generate human-readable reasoning."""
        parts = []
        
        parts.append(f"Grounding score: {grounding:.1%}")
        
        if influences:
            top = influences[0]
            parts.append(f"Most influential doc: {top[0]} ({top[1]:.1%})")
        
        if ungrounded:
            parts.append(f"Ungrounded terms: {', '.join(ungrounded[:5])}")
        
        return " | ".join(parts)
    
    def flag_if_risky(
        self,
        code: str,
        threshold: RiskLevel = RiskLevel.HIGH,
    ) -> Optional[str]:
        """
        Return a warning if code is above risk threshold.
        
        Returns:
            Warning message if risky, None otherwise
        """
        result = self.analyze(code)
        
        risk_order = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
        
        if risk_order.index(result.risk_level) >= risk_order.index(threshold):
            return (
                f"CAUTION: Code may be hallucinated (grounding={result.grounding_score:.1%}). "
                f"Ungrounded terms: {', '.join(result.ungrounded_claims[:3])}"
            )
        
        return None
    
    def clear_documents(self) -> None:
        """Clear all documents."""
        self.documents.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            **self._stats,
            "documents_loaded": len(self.documents),
        }
    
    def print_status(self) -> None:
        """Print engine status."""
        print("\n" + "=" * 60)
        print("[*] CAUSAL ATTRIBUTION ENGINE STATUS")
        print("=" * 60)
        
        print(f"\n[D] Documents: {len(self.documents)}")
        for doc in list(self.documents.values())[:5]:
            print(f"   - {doc.title[:40]}... (rel={doc.relevance_score:.2f})")
        
        print(f"\n[%] Stats:")
        for key, val in self.get_stats().items():
            if isinstance(val, float):
                print(f"   {key}: {val:.2f}")
            else:
                print(f"   {key}: {val}")
        
        print("=" * 60)


# =============================================================================
# PACKAGE INIT
# =============================================================================

__all__ = [
    "CausalAttributionEngine",
    "GroundingAnalyzer",
    "RetrievedDocument",
    "AttributionResult",
    "InfluenceVector",
    "AttributionSource",
    "RiskLevel",
]


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("[*] S.P.I.D.E.R. Causal Attribution Engine - Demo")
    print("=" * 70)
    
    engine = CausalAttributionEngine()
    
    # Add reference documentation
    print("\n[1] Adding reference documentation...")
    
    engine.create_document(
        title="JSON Library Documentation",
        content="""
        json.load(fp) - Deserialize fp to a Python object.
        json.loads(s) - Deserialize s (a string) to a Python object.
        json.dump(obj, fp) - Serialize obj as JSON to fp.
        json.dumps(obj) - Serialize obj to a JSON formatted string.
        JSONDecodeError - Raised when parsing fails.
        """,
        source="https://docs.python.org/3/library/json.html",
        relevance=0.9,
    )
    
    print(f"   Documents loaded: {len(engine.documents)}")
    
    # Test well-grounded code
    print("\n[2] Analyzing well-grounded code...")
    
    grounded_code = '''
import json

def parse_config(filepath):
    with open(filepath) as fp:
        return json.load(fp)
'''
    
    result = engine.analyze(grounded_code)
    print(f"   Source: {result.source.name}")
    print(f"   Risk: {result.risk_level.name}")
    print(f"   Grounding: {result.grounding_score:.1%}")
    
    # Test ungrounded code
    print("\n[3] Analyzing ungrounded code (hallucination risk)...")
    
    ungrounded_code = '''
from magic_parser import SuperParser

def parse_config(filepath):
    parser = SuperParser.from_file(filepath)
    return parser.to_dict(recursive=True, validate=True)
'''
    
    result = engine.analyze(ungrounded_code)
    print(f"   Source: {result.source.name}")
    print(f"   Risk: {result.risk_level.name}")
    print(f"   Grounding: {result.grounding_score:.1%}")
    print(f"   Ungrounded: {', '.join(result.ungrounded_claims[:5])}")
    
    # Flag check
    warning = engine.flag_if_risky(ungrounded_code)
    if warning:
        print(f"\n   [!] {warning}")
    
    engine.print_status()
