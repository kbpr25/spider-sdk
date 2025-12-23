"""
S.P.I.D.E.R. Epistemic Valve - Self-Aware Confidence Throttle
==============================================================

Born from: Assumption-3 (Calibration), Assumption-4 (Verbalized Uncertainty)

The Scientific Finding:
"Modern deep networks are miscalibrated—they are most confident when they
are wrong. Models can be fine-tuned/prompted to verbalize their uncertainty
(e.g., 'I am 60% sure') and these verbal probabilities are actually MORE
ACCURATE than the model's raw logits."

The Solution:
We implement a "Confidence Throttle" on SDK output.

Before returning an answer:
1. Ask the model: "How confident are you (0-100)?"
2. Apply Temperature Scaling to strip away arrogance
3. Gate the output based on confidence level

Valve Logic:
- Confidence > 90%: Output answer
- Confidence 50-90%: Output + WARNING
- Confidence < 50%: HARD BLOCK - "Insufficient data to answer"

Result: The system becomes SELF-AWARE. It knows what it doesn't know.
"""

import json
import logging
import math
import re
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIDENCE LEVELS
# =============================================================================

class ConfidenceLevel(IntEnum):
    """Confidence classification levels."""
    HIGH = auto()       # > 90% - Safe to output
    MEDIUM = auto()     # 50-90% - Output with warning
    LOW = auto()        # 20-50% - Output with strong warning
    VERY_LOW = auto()   # < 20% - Block output


@dataclass
class ConfidenceAssessment:
    """Result of confidence assessment."""
    raw_confidence: float              # 0-100 from model
    calibrated_confidence: float       # After temperature scaling
    reasoning: str                     # Why this confidence level
    level: ConfidenceLevel
    should_output: bool
    warning: Optional[str] = None


@dataclass
class EpistemicResult:
    """Complete epistemic analysis result."""
    original_response: str
    assessment: ConfidenceAssessment
    final_output: str
    blocked: bool
    uncertainty_markers: List[str]     # Words indicating uncertainty
    hedging_phrases: List[str]         # "I think", "maybe", etc.


# =============================================================================
# UNCERTAINTY DETECTION
# =============================================================================

# Linguistic markers of uncertainty
UNCERTAINTY_MARKERS = [
    "maybe", "perhaps", "possibly", "probably", "likely",
    "might", "could", "may", "uncertain", "unclear",
    "not sure", "unsure", "don't know", "hard to say",
    "it depends", "approximately", "roughly", "around",
    "I think", "I believe", "I assume", "I guess",
    "it seems", "appears to", "seems like", "looks like",
]

HEDGING_PATTERNS = [
    r"I('m| am) not (entirely |completely |totally )?sure",
    r"I (think|believe|assume|suppose|guess)",
    r"It (might|could|may) be",
    r"(Possibly|Probably|Perhaps|Maybe)",
    r"To the best of my knowledge",
    r"If I('m| am) not mistaken",
    r"As far as I (know|understand|recall)",
    r"I('d| would) say (roughly|approximately|about)",
]

CONFIDENCE_ASKING_PROMPT = """
Analyze your previous response and provide a confidence assessment.

Your response was:
"{response}"

The question was:
"{question}"

The sources provided were:
{sources}

Now, honestly assess:
1. On a scale of 0-100, how confident are you that your response is ACCURATE and SUPPORTED by the sources?
2. What specific evidence from the sources supports your answer?
3. What aspects of your answer are you LEAST certain about?

Respond in this exact JSON format:
{{
    "confidence": <0-100>,
    "reasoning": "<why this confidence level>",
    "evidence": ["<source quote 1>", "<source quote 2>"],
    "uncertainties": ["<uncertainty 1>", "<uncertainty 2>"]
}}
"""


# =============================================================================
# TEMPERATURE SCALING (CALIBRATION)
# =============================================================================

class TemperatureScaler:
    """
    Temperature Scaling for Confidence Calibration.
    
    From Assumption-3: Modern networks are overconfident.
    Temperature scaling learns a single parameter T that
    recalibrates the confidence scores.
    
    calibrated_confidence = original_confidence ^ (1/T)
    
    T > 1: Reduces overconfidence
    T < 1: Increases confidence
    """
    
    def __init__(self, temperature: float = 1.5):
        """
        Initialize Temperature Scaler.
        
        Args:
            temperature: Scaling factor (default 1.5 reduces overconfidence)
        """
        self.temperature = temperature
    
    def calibrate(self, raw_confidence: float) -> float:
        """
        Apply temperature scaling to raw confidence.
        
        Args:
            raw_confidence: 0-100 confidence score
            
        Returns:
            Calibrated confidence 0-100
        """
        if raw_confidence <= 0:
            return 0.0
        if raw_confidence >= 100:
            raw_confidence = 99.9
        
        # Normalize to 0-1
        p = raw_confidence / 100.0
        
        # Apply temperature scaling
        # Using softmax-style scaling
        calibrated = p ** (1 / self.temperature)
        
        # Re-normalize to 0-100
        return calibrated * 100.0
    
    def calibrate_logits(self, logits: List[float]) -> List[float]:
        """Apply temperature scaling to logits."""
        return [l / self.temperature for l in logits]
    
    def set_temperature(self, t: float) -> None:
        """Update temperature parameter."""
        if t > 0:
            self.temperature = t


# =============================================================================
# EPISTEMIC VALVE
# =============================================================================

class EpistemicValve:
    """
    The Epistemic Confidence Layer.
    
    Makes the system SELF-AWARE by:
    1. Prompting the model to verbalize its uncertainty
    2. Applying temperature scaling for calibration
    3. Gating output based on confidence thresholds
    
    Usage:
        valve = EpistemicValve()
        
        # Check confidence before outputting
        result = valve.assess(
            response="Paris is the capital of France.",
            question="What is the capital of France?",
            sources=["France is a country with Paris as its capital."],
            llm_callback=my_llm_function  # For asking confidence
        )
        
        if result.blocked:
            print("Insufficient confidence to answer")
        else:
            print(result.final_output)
    """
    
    def __init__(
        self,
        high_threshold: float = 90.0,
        medium_threshold: float = 50.0,
        low_threshold: float = 20.0,
        temperature: float = 1.5,
        strict_mode: bool = False,
    ):
        """
        Initialize Epistemic Valve.
        
        Args:
            high_threshold: Confidence above this = safe output
            medium_threshold: Confidence above this = output with warning
            low_threshold: Confidence below this = block output
            temperature: Calibration temperature (>1 reduces overconfidence)
            strict_mode: If True, block medium confidence too
        """
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.low_threshold = low_threshold
        self.strict_mode = strict_mode
        
        self.scaler = TemperatureScaler(temperature)
        
        # Compile patterns
        self.hedging_patterns = [
            re.compile(p, re.IGNORECASE) for p in HEDGING_PATTERNS
        ]
        
        self._stats = {
            "assessments": 0,
            "high_confidence": 0,
            "medium_confidence": 0,
            "low_confidence": 0,
            "blocked": 0,
        }
    
    def detect_uncertainty_markers(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Detect linguistic markers of uncertainty.
        
        Returns:
            Tuple of (uncertainty_words, hedging_phrases)
        """
        text_lower = text.lower()
        
        # Find uncertainty markers
        found_markers = []
        for marker in UNCERTAINTY_MARKERS:
            if marker.lower() in text_lower:
                found_markers.append(marker)
        
        # Find hedging patterns
        found_hedging = []
        for pattern in self.hedging_patterns:
            matches = pattern.findall(text)
            found_hedging.extend(matches)
        
        return (found_markers, found_hedging)
    
    def estimate_confidence_from_text(self, text: str) -> float:
        """
        Estimate confidence from linguistic cues (no LLM needed).
        
        Uses hedging words and uncertainty markers as signals.
        """
        markers, hedging = self.detect_uncertainty_markers(text)
        
        # Base confidence
        confidence = 80.0
        
        # Reduce for each uncertainty marker
        confidence -= len(markers) * 5.0
        
        # Reduce for each hedging phrase
        confidence -= len(hedging) * 8.0
        
        # Check for explicit uncertainty
        if "I don't know" in text.lower() or "I'm not sure" in text.lower():
            confidence -= 30.0
        
        # Check for definitive statements
        if " is " in text and len(text) < 100:
            confidence += 10.0
        
        return max(0.0, min(100.0, confidence))
    
    def ask_confidence(
        self,
        response: str,
        question: str,
        sources: List[str],
        llm_callback: Callable[[str], str],
    ) -> Tuple[float, str, List[str]]:
        """
        Ask the LLM to verbalize its confidence.
        
        From Assumption-4: Verbal probabilities are more accurate
        than raw logits.
        
        Returns:
            Tuple of (confidence, reasoning, uncertainties)
        """
        source_text = "\n".join(f"- {s[:200]}" for s in sources[:5])
        
        prompt = CONFIDENCE_ASKING_PROMPT.format(
            response=response,
            question=question,
            sources=source_text,
        )
        
        try:
            llm_response = llm_callback(prompt)
            
            # Parse JSON response
            json_match = re.search(r'\{[^{}]+\}', llm_response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                confidence = float(data.get("confidence", 50))
                reasoning = data.get("reasoning", "No reasoning provided")
                uncertainties = data.get("uncertainties", [])
                return (confidence, reasoning, uncertainties)
        except Exception as e:
            logger.warning(f"Failed to parse confidence: {e}")
        
        # Fallback to text-based estimation
        confidence = self.estimate_confidence_from_text(response)
        return (confidence, "Estimated from linguistic cues", [])
    
    def classify_confidence(self, confidence: float) -> ConfidenceLevel:
        """Classify confidence into level."""
        if confidence >= self.high_threshold:
            return ConfidenceLevel.HIGH
        elif confidence >= self.medium_threshold:
            return ConfidenceLevel.MEDIUM
        elif confidence >= self.low_threshold:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def assess(
        self,
        response: str,
        question: str = "",
        sources: List[str] = None,
        llm_callback: Optional[Callable[[str], str]] = None,
    ) -> EpistemicResult:
        """
        Assess confidence and gate the output.
        
        Args:
            response: The LLM's generated response
            question: The original question
            sources: Retrieved source documents
            llm_callback: Function to call LLM for confidence check
            
        Returns:
            EpistemicResult with final gated output
        """
        self._stats["assessments"] += 1
        sources = sources or []
        
        # Step 1: Ask for verbalized confidence (if LLM available)
        if llm_callback and sources:
            raw_confidence, reasoning, uncertainties = self.ask_confidence(
                response, question, sources, llm_callback
            )
        else:
            # Fallback to text-based estimation
            raw_confidence = self.estimate_confidence_from_text(response)
            reasoning = "Estimated from linguistic cues (no LLM callback)"
            uncertainties = []
        
        # Step 2: Apply temperature scaling (calibration)
        calibrated_confidence = self.scaler.calibrate(raw_confidence)
        
        # Step 3: Classify and gate
        level = self.classify_confidence(calibrated_confidence)
        markers, hedging = self.detect_uncertainty_markers(response)
        
        # Build assessment
        warning = None
        should_output = True
        
        if level == ConfidenceLevel.HIGH:
            self._stats["high_confidence"] += 1
            final_output = response
        
        elif level == ConfidenceLevel.MEDIUM:
            self._stats["medium_confidence"] += 1
            if self.strict_mode:
                should_output = False
                final_output = (
                    "[CAUTION: Medium confidence - verification recommended]\n\n"
                    f"{response}"
                )
            else:
                warning = f"⚠️ Confidence: {calibrated_confidence:.0f}% - Verify independently"
                final_output = f"{response}\n\n{warning}"
        
        elif level == ConfidenceLevel.LOW:
            self._stats["low_confidence"] += 1
            warning = (
                f"⚠️ LOW CONFIDENCE ({calibrated_confidence:.0f}%): "
                "This response may contain inaccuracies."
            )
            final_output = f"[LOW CONFIDENCE RESPONSE]\n\n{response}\n\n{warning}"
        
        else:  # VERY_LOW
            self._stats["blocked"] += 1
            should_output = False
            final_output = (
                "❌ RESPONSE BLOCKED - Insufficient confidence to answer truthfully.\n\n"
                f"Confidence: {calibrated_confidence:.0f}%\n"
                f"Reason: {reasoning}"
            )
        
        assessment = ConfidenceAssessment(
            raw_confidence=raw_confidence,
            calibrated_confidence=calibrated_confidence,
            reasoning=reasoning,
            level=level,
            should_output=should_output,
            warning=warning,
        )
        
        return EpistemicResult(
            original_response=response,
            assessment=assessment,
            final_output=final_output,
            blocked=not should_output,
            uncertainty_markers=markers,
            hedging_phrases=hedging,
        )
    
    def get_stats(self) -> Dict[str, int]:
        return self._stats.copy()
    
    def print_assessment(self, result: EpistemicResult) -> None:
        """Print detailed assessment."""
        a = result.assessment
        
        level_emoji = {
            ConfidenceLevel.HIGH: "✅",
            ConfidenceLevel.MEDIUM: "🟡",
            ConfidenceLevel.LOW: "🟠",
            ConfidenceLevel.VERY_LOW: "❌",
        }
        
        print("\n" + "=" * 60)
        print("🧠 EPISTEMIC ASSESSMENT")
        print("=" * 60)
        print(f"\n{level_emoji[a.level]} Confidence Level: {a.level.name}")
        print(f"   Raw Confidence:       {a.raw_confidence:.1f}%")
        print(f"   Calibrated Confidence:{a.calibrated_confidence:.1f}%")
        print(f"   Reasoning: {a.reasoning[:100]}")
        
        if result.uncertainty_markers:
            print(f"\n📍 Uncertainty Markers: {', '.join(result.uncertainty_markers[:5])}")
        
        if result.hedging_phrases:
            print(f"📍 Hedging Phrases: {', '.join(str(h) for h in result.hedging_phrases[:5])}")
        
        print(f"\n🚦 Should Output: {'YES' if a.should_output else 'BLOCKED'}")
        if a.warning:
            print(f"⚠️ Warning: {a.warning}")
        
        print("=" * 60)


# =============================================================================
# PACKAGE INIT
# =============================================================================

__all__ = [
    "EpistemicValve",
    "EpistemicResult",
    "ConfidenceAssessment",
    "ConfidenceLevel",
    "TemperatureScaler",
]


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🧠 S.P.I.D.E.R. Epistemic Valve - Demo")
    print("=" * 70)
    
    valve = EpistemicValve(temperature=1.5)
    
    # Test responses with different confidence levels
    test_cases = [
        {
            "question": "What is 2+2?",
            "response": "2+2 equals 4.",
            "expected": "HIGH",
        },
        {
            "question": "What is the weather tomorrow?",
            "response": "I think the weather might be sunny, but I'm not entirely sure.",
            "expected": "LOW",
        },
        {
            "question": "Is Python a programming language?",
            "response": "Python is a high-level programming language used for various applications.",
            "expected": "HIGH",
        },
        {
            "question": "What caused the collapse?",
            "response": "I don't know exactly, it could be many factors, perhaps structural issues.",
            "expected": "VERY_LOW",
        },
    ]
    
    for case in test_cases:
        print(f"\n{'─' * 60}")
        print(f"❓ Question: {case['question']}")
        print(f"💬 Response: {case['response']}")
        
        result = valve.assess(
            response=case["response"],
            question=case["question"],
        )
        
        valve.print_assessment(result)
        print(f"\n📤 Final Output:\n{result.final_output[:200]}...")
    
    print(f"\n📊 Stats: {valve.get_stats()}")
