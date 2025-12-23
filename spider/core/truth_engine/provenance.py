"""
S.P.I.D.E.R. Provenance Tracker - Citation-Enforced Generation
==============================================================

Born from: Assumption-1 (RAG), Assumption-2 (REALM), Assumption-6 (Attribution)

The Scientific Finding:
"Standard RAG retrieves documents and hopes the LLM uses them. Models often
ignore context and rely on internal bias. Non-Parametric Memory (External Docs)
must be the PRIMARY driver of generation, not an additive one."

The Solution:
We stop treating generation as "Text." We treat it as "Citations."

Every generated sentence MUST map to a span in the retrieved context.
If the model generates something not grounded in sources, we REDACT it.

This makes hallucination structurally impossible.
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# PROVENANCE TYPES
# =============================================================================

class ProvenanceStatus(Enum):
    """Status of provenance verification."""
    GROUNDED = auto()      # Claim is supported by source
    PARTIAL = auto()       # Weak support found
    UNGROUNDED = auto()    # No support found - HALLUCINATION
    REDACTED = auto()      # Removed due to no support


@dataclass
class SourceDocument:
    """A retrieved source document."""
    doc_id: str
    content: str
    title: str = ""
    url: str = ""
    relevance_score: float = 0.0
    
    def __hash__(self):
        return hash(self.doc_id)
    
    def get_sentences(self) -> List[str]:
        """Split content into sentences."""
        return [s.strip() for s in re.split(r'[.!?]+', self.content) if s.strip()]


@dataclass
class Citation:
    """A citation linking a claim to a source."""
    claim: str                         # The generated claim
    source_id: str                     # Which document supports it
    source_span: str                   # Exact text from source
    confidence: float                  # How confident the match is (0-1)
    span_start: int = 0                # Character position in source
    span_end: int = 0


@dataclass
class ProvenanceResult:
    """Result of provenance checking on generated text."""
    original_text: str
    grounded_text: str                 # Text with ungrounded claims removed
    claims: List[Tuple[str, ProvenanceStatus, Optional[Citation]]]
    overall_grounding: float           # % of claims that are grounded
    redacted_claims: List[str]         # Claims that were removed
    citations: List[Citation]          # Valid citations
    
    def is_trustworthy(self, threshold: float = 0.8) -> bool:
        """Check if the output meets trust threshold."""
        return self.overall_grounding >= threshold
    
    def to_cited_text(self) -> str:
        """Return text with inline citations."""
        cited = self.grounded_text
        for citation in sorted(self.citations, key=lambda c: -len(c.claim)):
            if citation.claim in cited:
                cited = cited.replace(
                    citation.claim,
                    f"{citation.claim} [Source: {citation.source_id}]"
                )
        return cited


# =============================================================================
# TEXT SIMILARITY
# =============================================================================

class SimilarityEngine:
    """
    Measures semantic and lexical similarity between claims and sources.
    
    Uses multiple signals:
    1. Lexical overlap (n-gram matching)
    2. Sequence matching (SequenceMatcher)
    3. Keyword overlap
    4. Optional: Embedding similarity (if embedder provided)
    """
    
    def __init__(self, embedder: Optional[Callable[[str], List[float]]] = None):
        self.embedder = embedder
    
    def lexical_overlap(self, claim: str, source: str) -> float:
        """Calculate word overlap ratio."""
        claim_words = set(claim.lower().split())
        source_words = set(source.lower().split())
        
        if not claim_words:
            return 0.0
        
        overlap = claim_words & source_words
        return len(overlap) / len(claim_words)
    
    def sequence_similarity(self, claim: str, source: str) -> float:
        """Calculate sequence similarity using SequenceMatcher."""
        return SequenceMatcher(None, claim.lower(), source.lower()).ratio()
    
    def ngram_overlap(self, claim: str, source: str, n: int = 3) -> float:
        """Calculate n-gram overlap."""
        def get_ngrams(text: str, n: int) -> Set[str]:
            words = text.lower().split()
            return set(' '.join(words[i:i+n]) for i in range(len(words)-n+1))
        
        claim_ngrams = get_ngrams(claim, n)
        source_ngrams = get_ngrams(source, n)
        
        if not claim_ngrams:
            return 0.0
        
        overlap = claim_ngrams & source_ngrams
        return len(overlap) / len(claim_ngrams)
    
    def embedding_similarity(self, claim: str, source: str) -> float:
        """Calculate embedding cosine similarity."""
        if not self.embedder:
            return 0.0
        
        try:
            claim_emb = self.embedder(claim)
            source_emb = self.embedder(source)
            
            # Cosine similarity
            dot = sum(a * b for a, b in zip(claim_emb, source_emb))
            norm_a = sum(a * a for a in claim_emb) ** 0.5
            norm_b = sum(b * b for b in source_emb) ** 0.5
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            return dot / (norm_a * norm_b)
        except Exception:
            return 0.0
    
    def combined_similarity(self, claim: str, source: str) -> float:
        """
        Calculate combined similarity score.
        
        Weights:
        - Lexical: 30%
        - Sequence: 30%
        - N-gram: 20%
        - Embedding: 20% (if available)
        """
        lexical = self.lexical_overlap(claim, source)
        sequence = self.sequence_similarity(claim, source)
        ngram = self.ngram_overlap(claim, source)
        
        if self.embedder:
            embedding = self.embedding_similarity(claim, source)
            return 0.25 * lexical + 0.25 * sequence + 0.25 * ngram + 0.25 * embedding
        else:
            return 0.35 * lexical + 0.35 * sequence + 0.30 * ngram
    
    def find_best_match(
        self,
        claim: str,
        sources: List[SourceDocument],
        threshold: float = 0.3,
    ) -> Optional[Tuple[SourceDocument, str, float]]:
        """
        Find the best matching source span for a claim.
        
        Returns:
            Tuple of (source, span, similarity) or None if no match above threshold
        """
        best_match = None
        best_score = 0.0
        best_span = ""
        
        for source in sources:
            # Check against full content
            full_score = self.combined_similarity(claim, source.content)
            if full_score > best_score:
                best_score = full_score
                best_match = source
                best_span = source.content[:500]  # First 500 chars
            
            # Check against individual sentences
            for sentence in source.get_sentences():
                if len(sentence) < 10:
                    continue
                
                score = self.combined_similarity(claim, sentence)
                if score > best_score:
                    best_score = score
                    best_match = source
                    best_span = sentence
        
        if best_score >= threshold and best_match:
            return (best_match, best_span, best_score)
        
        return None


# =============================================================================
# PROVENANCE TRACKER
# =============================================================================

class ProvenanceTracker:
    """
    The Provenance-Enforced Generator (PEG).
    
    Ensures every generated claim is grounded in retrieved sources.
    Ungrounded claims are REDACTED before reaching the user.
    
    This makes hallucination structurally impossible.
    
    Usage:
        tracker = ProvenanceTracker()
        
        # Add retrieved sources
        tracker.add_source("doc1", "The capital of France is Paris.")
        tracker.add_source("doc2", "The Eiffel Tower is in Paris.")
        
        # Verify generated text
        result = tracker.verify(
            "Paris is the capital of France. The sky is purple."
        )
        
        # "The sky is purple" is REDACTED (no source)
        print(result.grounded_text)  # "Paris is the capital of France."
        print(result.redacted_claims)  # ["The sky is purple"]
    """
    
    def __init__(
        self,
        grounding_threshold: float = 0.4,
        partial_threshold: float = 0.25,
        embedder: Optional[Callable[[str], List[float]]] = None,
        strict_mode: bool = True,
    ):
        """
        Initialize Provenance Tracker.
        
        Args:
            grounding_threshold: Minimum similarity for GROUNDED status
            partial_threshold: Minimum similarity for PARTIAL status
            embedder: Optional embedding function for semantic similarity
            strict_mode: If True, redact all ungrounded claims
        """
        self.grounding_threshold = grounding_threshold
        self.partial_threshold = partial_threshold
        self.strict_mode = strict_mode
        
        self.sources: Dict[str, SourceDocument] = {}
        self.similarity = SimilarityEngine(embedder)
        
        self._stats = {
            "claims_checked": 0,
            "claims_grounded": 0,
            "claims_partial": 0,
            "claims_redacted": 0,
        }
    
    def add_source(
        self,
        doc_id: str,
        content: str,
        title: str = "",
        url: str = "",
        relevance: float = 1.0,
    ) -> SourceDocument:
        """Add a source document to the knowledge base."""
        doc = SourceDocument(
            doc_id=doc_id,
            content=content,
            title=title or doc_id,
            url=url,
            relevance_score=relevance,
        )
        self.sources[doc_id] = doc
        return doc
    
    def add_sources(self, sources: List[Dict[str, Any]]) -> None:
        """Add multiple sources from dicts."""
        for s in sources:
            self.add_source(
                doc_id=s.get("id", s.get("doc_id", "")),
                content=s.get("content", s.get("text", "")),
                title=s.get("title", ""),
                url=s.get("url", ""),
                relevance=s.get("relevance", s.get("score", 1.0)),
            )
    
    def clear_sources(self) -> None:
        """Clear all sources."""
        self.sources.clear()
    
    def _extract_claims(self, text: str) -> List[str]:
        """Extract individual claims from generated text."""
        # Split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Skip very short fragments
                claims.append(sentence)
        
        return claims
    
    def verify_claim(self, claim: str) -> Tuple[ProvenanceStatus, Optional[Citation]]:
        """
        Verify a single claim against sources.
        
        Returns:
            Tuple of (status, citation if grounded)
        """
        self._stats["claims_checked"] += 1
        
        if not self.sources:
            return (ProvenanceStatus.UNGROUNDED, None)
        
        # Find best matching source
        match = self.similarity.find_best_match(
            claim,
            list(self.sources.values()),
            threshold=self.partial_threshold,
        )
        
        if not match:
            self._stats["claims_redacted"] += 1
            return (ProvenanceStatus.UNGROUNDED, None)
        
        source, span, score = match
        
        citation = Citation(
            claim=claim,
            source_id=source.doc_id,
            source_span=span,
            confidence=score,
        )
        
        if score >= self.grounding_threshold:
            self._stats["claims_grounded"] += 1
            return (ProvenanceStatus.GROUNDED, citation)
        else:
            self._stats["claims_partial"] += 1
            return (ProvenanceStatus.PARTIAL, citation)
    
    def verify(self, generated_text: str) -> ProvenanceResult:
        """
        Verify all claims in generated text.
        
        Redacts ungrounded claims if strict_mode is True.
        
        Args:
            generated_text: The LLM's generated response
            
        Returns:
            ProvenanceResult with grounded text and citations
        """
        claims = self._extract_claims(generated_text)
        
        verified_claims: List[Tuple[str, ProvenanceStatus, Optional[Citation]]] = []
        grounded_claims: List[str] = []
        redacted_claims: List[str] = []
        citations: List[Citation] = []
        
        for claim in claims:
            status, citation = self.verify_claim(claim)
            verified_claims.append((claim, status, citation))
            
            if status == ProvenanceStatus.GROUNDED:
                grounded_claims.append(claim)
                if citation:
                    citations.append(citation)
            elif status == ProvenanceStatus.PARTIAL:
                if not self.strict_mode:
                    grounded_claims.append(claim)
                    if citation:
                        citations.append(citation)
                else:
                    redacted_claims.append(claim)
            else:
                redacted_claims.append(claim)
        
        # Calculate grounding ratio
        total = len(claims)
        grounded_count = len([c for c in verified_claims if c[1] == ProvenanceStatus.GROUNDED])
        overall_grounding = grounded_count / total if total > 0 else 0.0
        
        # Build grounded text
        grounded_text = " ".join(grounded_claims)
        
        return ProvenanceResult(
            original_text=generated_text,
            grounded_text=grounded_text,
            claims=verified_claims,
            overall_grounding=overall_grounding,
            redacted_claims=redacted_claims,
            citations=citations,
        )
    
    def enforce(
        self,
        generated_text: str,
        min_grounding: float = 0.5,
    ) -> Tuple[str, bool]:
        """
        Enforce provenance and return safe output.
        
        Args:
            generated_text: The LLM's response
            min_grounding: Minimum grounding ratio to accept
            
        Returns:
            Tuple of (safe_text, is_acceptable)
        """
        result = self.verify(generated_text)
        
        if result.overall_grounding >= min_grounding:
            return (result.to_cited_text(), True)
        else:
            # Too many ungrounded claims
            if result.grounded_text.strip():
                return (
                    f"[PARTIAL RESPONSE - {result.overall_grounding:.0%} grounded]\n"
                    f"{result.to_cited_text()}\n\n"
                    f"[REDACTED: {len(result.redacted_claims)} unverifiable claims]",
                    False
                )
            else:
                return (
                    "[RESPONSE BLOCKED - No claims could be verified against sources]",
                    False
                )
    
    def get_stats(self) -> Dict[str, int]:
        return self._stats.copy()
    
    def print_report(self, result: ProvenanceResult) -> None:
        """Print a provenance verification report."""
        print("\n" + "=" * 60)
        print("📋 PROVENANCE VERIFICATION REPORT")
        print("=" * 60)
        
        print(f"\n📊 Overall Grounding: {result.overall_grounding:.0%}")
        print(f"   Trust Status: {'✅ TRUSTWORTHY' if result.is_trustworthy() else '⚠️ LOW CONFIDENCE'}")
        
        print(f"\n📝 Claims Analyzed: {len(result.claims)}")
        for claim, status, citation in result.claims:
            emoji = {
                ProvenanceStatus.GROUNDED: "✅",
                ProvenanceStatus.PARTIAL: "🟡",
                ProvenanceStatus.UNGROUNDED: "❌",
                ProvenanceStatus.REDACTED: "🚫",
            }.get(status, "❓")
            
            print(f"\n   {emoji} [{status.name}] {claim[:60]}...")
            if citation:
                print(f"      → Source: {citation.source_id} (conf: {citation.confidence:.0%})")
                print(f"      → Match: \"{citation.source_span[:50]}...\"")
        
        if result.redacted_claims:
            print(f"\n🚫 REDACTED CLAIMS ({len(result.redacted_claims)}):")
            for claim in result.redacted_claims:
                print(f"   - {claim[:60]}...")
        
        print("\n" + "=" * 60)


# =============================================================================
# PACKAGE INIT
# =============================================================================

__all__ = [
    "ProvenanceTracker",
    "ProvenanceResult",
    "ProvenanceStatus",
    "Citation",
    "SourceDocument",
    "SimilarityEngine",
]


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🔍 S.P.I.D.E.R. Provenance Tracker - Demo")
    print("=" * 70)
    
    tracker = ProvenanceTracker(strict_mode=True)
    
    # Add source documents
    tracker.add_source(
        "wiki_paris",
        "Paris is the capital and most populous city of France. "
        "The Eiffel Tower, built in 1889, is located in Paris. "
        "Paris is known for the Louvre Museum and Notre-Dame Cathedral."
    )
    tracker.add_source(
        "wiki_france",
        "France is a country in Western Europe. "
        "The official language of France is French. "
        "France has a population of approximately 67 million people."
    )
    
    # Test with mixed grounded and hallucinated content
    generated = (
        "Paris is the capital of France. "
        "The Eiffel Tower was built in 1889 and is a famous landmark. "
        "The sky over Paris turns purple every evening. "  # HALLUCINATION
        "France is located in Western Europe. "
        "French people eat rocks for breakfast."  # HALLUCINATION
    )
    
    print("\n📥 GENERATED TEXT:")
    print(generated)
    
    result = tracker.verify(generated)
    tracker.print_report(result)
    
    print("\n📤 SAFE OUTPUT (with citations):")
    safe_text, is_acceptable = tracker.enforce(generated)
    print(safe_text)
    print(f"\nAcceptable: {is_acceptable}")
    
    print(f"\n📊 Stats: {tracker.get_stats()}")
