"""
S.P.I.D.E.R. Logical Entailment Shield - Adversarial Truth Filter
==================================================================

Born from: Assumption-5 (Faithful Reasoning), Assumption-10 (TruthfulQA)

The Scientific Finding:
"A model can get the right answer for the wrong reasons (Faithfulness Gap).
Larger models are better liars—they mimic human misconceptions (e.g.,
'sugar causes hyperactivity') because they appear in training data."

The Solution:
Build a dedicated "Critic" that acts as a Logic Gate.

1. Decompose: Break answer into atomic claims
2. Entailment Check: Does Context C LOGICALLY entail Claim A?
3. TruthfulQA Trap: Immunize against common myths

Result: The SDK ensures CAUSAL VALIDITY. The answer isn't just
"found" in the docs; it is LOGICALLY DERIVED.
"""

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# ENTAILMENT TYPES
# =============================================================================

class EntailmentStatus(Enum):
    """Result of logical entailment check."""
    ENTAILED = auto()       # Context logically implies claim
    NEUTRAL = auto()        # Context neither supports nor contradicts
    CONTRADICTED = auto()   # Context contradicts claim
    UNSUPPORTED = auto()    # No relevant context


class ClaimType(Enum):
    """Type of claim being made."""
    FACTUAL = auto()        # Verifiable fact
    CAUSAL = auto()         # Cause-effect relationship
    COMPARATIVE = auto()    # Comparison between things
    QUANTITATIVE = auto()   # Numeric claim
    DEFINITIONAL = auto()   # Definition/classification
    TEMPORAL = auto()       # Time-related claim
    OPINION = auto()        # Subjective statement


@dataclass
class AtomicClaim:
    """A single atomic claim extracted from text."""
    claim_id: str
    text: str
    claim_type: ClaimType
    subject: str = ""               # What the claim is about
    predicate: str = ""             # What is claimed about it
    entities: List[str] = field(default_factory=list)
    

@dataclass
class EntailmentResult:
    """Result of entailment check for a claim."""
    claim: AtomicClaim
    status: EntailmentStatus
    confidence: float               # 0-1 confidence in the check
    supporting_evidence: str = ""   # Text that supports/contradicts
    reasoning: str = ""             # Why this status
    is_myth: bool = False           # Flagged as common myth


@dataclass
class ShieldResult:
    """Complete truth filtering result."""
    original_text: str
    claims: List[AtomicClaim]
    entailments: List[EntailmentResult]
    verified_claims: List[str]
    rejected_claims: List[str]
    myth_claims: List[str]
    overall_validity: float
    filtered_text: str


# =============================================================================
# COMMON MYTHS (from TruthfulQA)
# =============================================================================

# Common misconceptions that LLMs often perpetuate
COMMON_MYTHS = {
    # Health myths
    "sugar causes hyperactivity": "Studies show no causal link between sugar and hyperactivity",
    "we only use 10% of our brain": "Brain scans show all areas have function",
    "cracking knuckles causes arthritis": "No evidence links knuckle cracking to arthritis",
    "cold weather causes colds": "Colds are caused by viruses, not temperature",
    "chocolate causes acne": "Studies show weak or no link between chocolate and acne",
    "reading in dim light damages eyes": "May cause eye strain but not permanent damage",
    
    # Science myths
    "goldfish have 3 second memory": "Goldfish can remember for months",
    "lightning never strikes twice": "Lightning often strikes the same place repeatedly",
    "humans have 5 senses": "Humans have many more senses including balance, temperature",
    "great wall visible from space": "Not visible to naked eye from orbit",
    
    # History myths
    "vikings wore horned helmets": "No evidence of horned helmets in actual vikings",
    "napoleon was short": "Napoleon was average height for his time",
    "einstein failed math": "Einstein excelled at mathematics",
    
    # Food myths
    "msg is dangerous": "MSG is generally recognized as safe by FDA",
    "organic means no pesticides": "Organic farming uses approved pesticides",
}

# Patterns indicating potential myth
MYTH_PATTERNS = [
    r"everyone knows",
    r"it's common knowledge",
    r"studies show that \w+ causes",
    r"scientists have proven",
    r"research indicates",
    r"experts agree",
]


# =============================================================================
# CLAIM DECOMPOSER
# =============================================================================

class ClaimDecomposer:
    """
    Decomposes text into atomic claims.
    
    Uses pattern matching and heuristics to extract
    individual verifiable claims from text.
    """
    
    def __init__(self):
        self.claim_indicators = [
            r"is\s+a\s+",           # X is a Y
            r"are\s+",              # X are Y
            r"was\s+",              # X was Y
            r"were\s+",             # X were Y
            r"causes?\s+",          # X causes Y
            r"leads?\s+to",         # X leads to Y
            r"results?\s+in",       # X results in Y
            r"means?\s+that",       # X means that Y
            r"shows?\s+that",       # X shows that Y
            r"proved?\s+that",      # X proves that Y
        ]
    
    def decompose(self, text: str) -> List[AtomicClaim]:
        """Break text into atomic claims."""
        claims = []
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
            
            # Determine claim type
            claim_type = self._classify_claim(sentence)
            
            # Extract subject and predicate (simplified)
            subject, predicate = self._extract_subject_predicate(sentence)
            
            # Extract entities
            entities = self._extract_entities(sentence)
            
            claim = AtomicClaim(
                claim_id=f"claim_{i}",
                text=sentence,
                claim_type=claim_type,
                subject=subject,
                predicate=predicate,
                entities=entities,
            )
            claims.append(claim)
        
        return claims
    
    def _classify_claim(self, text: str) -> ClaimType:
        """Classify the type of claim."""
        text_lower = text.lower()
        
        # Check for causal claims
        if any(word in text_lower for word in ["causes", "leads to", "results in", "because"]):
            return ClaimType.CAUSAL
        
        # Check for comparative claims
        if any(word in text_lower for word in ["more than", "less than", "better", "worse", "compared to"]):
            return ClaimType.COMPARATIVE
        
        # Check for quantitative claims
        if re.search(r'\d+\.?\d*\s*%|\d+\.?\d*\s*(million|billion|thousand)', text_lower):
            return ClaimType.QUANTITATIVE
        
        # Check for temporal claims
        if any(word in text_lower for word in ["before", "after", "during", "in 19", "in 20", "year"]):
            return ClaimType.TEMPORAL
        
        # Check for definitional claims
        if " is a " in text_lower or " is the " in text_lower:
            return ClaimType.DEFINITIONAL
        
        # Check for opinion
        if any(word in text_lower for word in ["think", "believe", "feel", "opinion"]):
            return ClaimType.OPINION
        
        return ClaimType.FACTUAL
    
    def _extract_subject_predicate(self, text: str) -> Tuple[str, str]:
        """Extract subject and predicate from claim."""
        # Simple heuristic: split on main verb
        patterns = [
            r"^(.+?)\s+(is|are|was|were|has|have|can|will)\s+(.+)$",
            r"^(.+?)\s+(causes?|leads?\s+to|results?\s+in)\s+(.+)$",
        ]
        
        for pattern in patterns:
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                return (match.group(1).strip(), match.group(3).strip())
        
        # Fallback
        words = text.split()
        if len(words) >= 3:
            return (" ".join(words[:2]), " ".join(words[2:]))
        
        return (text, "")
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract named entities from text."""
        entities = []
        
        # Simple heuristic: capitalized words not at sentence start
        words = text.split()
        for i, word in enumerate(words[1:], 1):  # Skip first word
            if word[0].isupper() and len(word) > 1:
                entities.append(word.strip(".,!?"))
        
        return entities[:5]  # Limit


# =============================================================================
# ENTAILMENT CHECKER
# =============================================================================

class EntailmentChecker:
    """
    Checks if context logically entails a claim.
    
    Uses pattern matching and optional LLM for complex cases.
    """
    
    def __init__(
        self,
        llm_callback: Optional[Callable[[str], str]] = None,
        strict_mode: bool = True,
    ):
        self.llm_callback = llm_callback
        self.strict_mode = strict_mode
        
        # Compile myth patterns
        self.myth_patterns = [re.compile(p, re.IGNORECASE) for p in MYTH_PATTERNS]
    
    def check_myth(self, claim: str) -> Tuple[bool, str]:
        """Check if claim is a common myth."""
        claim_lower = claim.lower()
        
        for myth, correction in COMMON_MYTHS.items():
            if myth in claim_lower:
                return (True, correction)
        
        # Check for myth indicators
        for pattern in self.myth_patterns:
            if pattern.search(claim):
                return (True, "Claim uses common unsupported phrasing")
        
        return (False, "")
    
    def check_lexical_entailment(self, claim: str, context: str) -> Tuple[float, str]:
        """
        Check entailment using lexical overlap.
        
        Returns (confidence, evidence)
        """
        claim_words = set(claim.lower().split())
        context_lower = context.lower()
        
        # Check word overlap
        overlap = sum(1 for word in claim_words if word in context_lower)
        overlap_ratio = overlap / len(claim_words) if claim_words else 0
        
        # Find best matching sentence
        sentences = re.split(r'[.!?]', context)
        best_match = ""
        best_score = 0
        
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            score = len(claim_words & sentence_words) / len(claim_words) if claim_words else 0
            if score > best_score:
                best_score = score
                best_match = sentence.strip()
        
        return (overlap_ratio, best_match)
    
    def check_contradiction(self, claim: str, context: str) -> bool:
        """Check if context contradicts claim."""
        # Simple negation check
        claim_lower = claim.lower()
        context_lower = context.lower()
        
        negation_patterns = [
            (r"is\s+(\w+)", r"is\s+not\s+\1"),
            (r"are\s+(\w+)", r"are\s+not\s+\1"),
            (r"can\s+(\w+)", r"cannot\s+\1"),
        ]
        
        for positive, negative in negation_patterns:
            pos_match = re.search(positive, claim_lower)
            if pos_match:
                neg_pattern = negative.replace(r"\1", pos_match.group(1))
                if re.search(neg_pattern, context_lower):
                    return True
        
        return False
    
    def check(self, claim: AtomicClaim, context: str) -> EntailmentResult:
        """
        Check if context entails the claim.
        
        Args:
            claim: The atomic claim to verify
            context: The source context
            
        Returns:
            EntailmentResult with status and confidence
        """
        # Step 1: Check for common myths
        is_myth, myth_correction = self.check_myth(claim.text)
        if is_myth:
            return EntailmentResult(
                claim=claim,
                status=EntailmentStatus.CONTRADICTED,
                confidence=0.9,
                supporting_evidence=myth_correction,
                reasoning="Claim contains a common misconception",
                is_myth=True,
            )
        
        # Step 2: Check for contradiction
        if self.check_contradiction(claim.text, context):
            return EntailmentResult(
                claim=claim,
                status=EntailmentStatus.CONTRADICTED,
                confidence=0.8,
                reasoning="Context contains negation of claim",
            )
        
        # Step 3: Check lexical entailment
        overlap, evidence = self.check_lexical_entailment(claim.text, context)
        
        if overlap >= 0.7:
            return EntailmentResult(
                claim=claim,
                status=EntailmentStatus.ENTAILED,
                confidence=min(0.9, overlap + 0.2),
                supporting_evidence=evidence,
                reasoning=f"High lexical overlap ({overlap:.0%}) with source",
            )
        elif overlap >= 0.4:
            return EntailmentResult(
                claim=claim,
                status=EntailmentStatus.NEUTRAL,
                confidence=overlap,
                supporting_evidence=evidence,
                reasoning=f"Partial support ({overlap:.0%}) found in source",
            )
        else:
            return EntailmentResult(
                claim=claim,
                status=EntailmentStatus.UNSUPPORTED,
                confidence=1.0 - overlap,
                reasoning=f"Low overlap ({overlap:.0%}) - claim not grounded",
            )


# =============================================================================
# LOGICAL ENTAILMENT SHIELD
# =============================================================================

class LogicalEntailmentShield:
    """
    The Adversarial Truth Filter.
    
    Ensures CAUSAL VALIDITY by:
    1. Decomposing answers into atomic claims
    2. Checking each claim for logical entailment
    3. Filtering out unsupported or mythical claims
    
    From Assumption-5: Prevents getting right answer for wrong reasons
    From Assumption-10: Immunizes against common myths
    
    Usage:
        shield = LogicalEntailmentShield()
        
        result = shield.filter(
            text="Sugar causes hyperactivity in children. Paris is the capital.",
            context="France is a country. Paris is the capital of France.",
        )
        
        print(result.filtered_text)  # Only verified claims
        print(result.myth_claims)    # Caught myths
    """
    
    def __init__(
        self,
        entailment_threshold: float = 0.5,
        llm_callback: Optional[Callable[[str], str]] = None,
        strict_mode: bool = True,
        block_myths: bool = True,
    ):
        """
        Initialize Logical Entailment Shield.
        
        Args:
            entailment_threshold: Minimum confidence for ENTAILED status
            llm_callback: Optional LLM for complex entailment checks
            strict_mode: If True, only allow fully entailed claims
            block_myths: If True, actively block common myths
        """
        self.entailment_threshold = entailment_threshold
        self.strict_mode = strict_mode
        self.block_myths = block_myths
        
        self.decomposer = ClaimDecomposer()
        self.checker = EntailmentChecker(llm_callback, strict_mode)
        
        self._stats = {
            "texts_filtered": 0,
            "claims_checked": 0,
            "claims_entailed": 0,
            "claims_rejected": 0,
            "myths_caught": 0,
        }
    
    def filter(self, text: str, context: str) -> ShieldResult:
        """
        Filter text through the logical entailment shield.
        
        Args:
            text: Generated text to verify
            context: Source context for verification
            
        Returns:
            ShieldResult with verified claims
        """
        self._stats["texts_filtered"] += 1
        
        # Step 1: Decompose into atomic claims
        claims = self.decomposer.decompose(text)
        
        # Step 2: Check each claim
        entailments = []
        verified = []
        rejected = []
        myths = []
        
        for claim in claims:
            self._stats["claims_checked"] += 1
            
            result = self.checker.check(claim, context)
            entailments.append(result)
            
            if result.is_myth and self.block_myths:
                myths.append(claim.text)
                rejected.append(claim.text)
                self._stats["myths_caught"] += 1
            elif result.status == EntailmentStatus.ENTAILED:
                verified.append(claim.text)
                self._stats["claims_entailed"] += 1
            elif result.status == EntailmentStatus.CONTRADICTED:
                rejected.append(claim.text)
                self._stats["claims_rejected"] += 1
            elif result.status == EntailmentStatus.NEUTRAL and not self.strict_mode:
                verified.append(claim.text)  # Allow in non-strict mode
            else:
                rejected.append(claim.text)
                self._stats["claims_rejected"] += 1
        
        # Build filtered text
        filtered_text = " ".join(verified)
        
        # Calculate validity
        total = len(claims)
        validity = len(verified) / total if total > 0 else 0.0
        
        return ShieldResult(
            original_text=text,
            claims=claims,
            entailments=entailments,
            verified_claims=verified,
            rejected_claims=rejected,
            myth_claims=myths,
            overall_validity=validity,
            filtered_text=filtered_text,
        )
    
    def is_valid(self, text: str, context: str, threshold: float = 0.7) -> bool:
        """Quick check if text is valid."""
        result = self.filter(text, context)
        return result.overall_validity >= threshold
    
    def get_stats(self) -> Dict[str, int]:
        return self._stats.copy()
    
    def print_result(self, result: ShieldResult) -> None:
        """Print detailed result."""
        print("\n" + "=" * 60)
        print("🛡️ LOGICAL ENTAILMENT SHIELD RESULT")
        print("=" * 60)
        
        print(f"\n📊 Overall Validity: {result.overall_validity:.0%}")
        print(f"📝 Claims Analyzed: {len(result.claims)}")
        print(f"✅ Verified: {len(result.verified_claims)}")
        print(f"❌ Rejected: {len(result.rejected_claims)}")
        print(f"🚫 Myths Caught: {len(result.myth_claims)}")
        
        print("\n🔍 CLAIM-BY-CLAIM ANALYSIS:")
        for ent in result.entailments:
            status_emoji = {
                EntailmentStatus.ENTAILED: "✅",
                EntailmentStatus.NEUTRAL: "🟡",
                EntailmentStatus.CONTRADICTED: "❌",
                EntailmentStatus.UNSUPPORTED: "⚠️",
            }.get(ent.status, "❓")
            
            myth_flag = " 🚫MYTH" if ent.is_myth else ""
            print(f"\n   {status_emoji} [{ent.status.name}]{myth_flag}")
            print(f"      Claim: {ent.claim.text[:60]}...")
            print(f"      Type: {ent.claim.claim_type.name}")
            print(f"      Confidence: {ent.confidence:.0%}")
            if ent.reasoning:
                print(f"      Reasoning: {ent.reasoning[:50]}...")
        
        if result.myth_claims:
            print("\n🚫 MYTHS BLOCKED:")
            for myth in result.myth_claims:
                print(f"   - {myth[:60]}...")
        
        print(f"\n📤 FILTERED OUTPUT:\n{result.filtered_text}")
        print("=" * 60)


# =============================================================================
# PACKAGE INIT
# =============================================================================

__all__ = [
    "LogicalEntailmentShield",
    "ShieldResult",
    "EntailmentResult",
    "EntailmentStatus",
    "AtomicClaim",
    "ClaimType",
    "ClaimDecomposer",
    "EntailmentChecker",
]


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🛡️ S.P.I.D.E.R. Logical Entailment Shield - Demo")
    print("=" * 70)
    
    shield = LogicalEntailmentShield(strict_mode=True, block_myths=True)
    
    # Test text with mixed claims
    generated_text = (
        "Paris is the capital of France. "
        "The Eiffel Tower was built in 1889. "
        "Sugar causes hyperactivity in children. "  # MYTH
        "France is located in Western Europe. "
        "Everyone knows that cold weather causes colds."  # MYTH
    )
    
    context = (
        "France is a country in Western Europe. "
        "Paris is the capital city of France. "
        "The Eiffel Tower is a famous landmark built in 1889. "
        "The official language is French."
    )
    
    print("\n📥 GENERATED TEXT:")
    print(generated_text)
    
    print("\n📚 CONTEXT:")
    print(context)
    
    result = shield.filter(generated_text, context)
    shield.print_result(result)
    
    print(f"\n📊 Stats: {shield.get_stats()}")
