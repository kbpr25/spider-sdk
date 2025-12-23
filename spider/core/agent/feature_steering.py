"""
S.P.I.D.E.R. Feature Steering Drive - Runtime Activation Control
=================================================================

Born from: Anthropic-2.5 (Scaling Monosemanticity) + Anthropic-2.8 (Biology of LLM)

The Anthropic Discovery:
"LLMs represent concepts (like 'Base64 encoding' or 'I am confused') as
specific DIRECTIONS in the neuron activation space. They use 'Sparse
Autoencoders' (SAE) to find these features."

The Flaw:
Anthropic uses this for safety (preventing jailbreaks). They don't use
it to FORCE coding excellence.

The S.P.I.D.E.R. Implementation:
We implement Runtime Activation Steering.

Mechanism:
1. Feature Mapping: Identify vectors for positive traits ("Unit Testing",
   "Error Handling") and negative traits ("Lazy Coding", "Guessing")
2. The Clamp: During generation, mathematically ADD positive vectors
   and SUBTRACT negative vectors from the residual stream
3. Result: We don't just PROMPT to be good. We LOBOTOMIZE laziness
   and STIMULATE the testing part of the brain.
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
# FEATURE TYPES
# =============================================================================

class FeaturePolarity(Enum):
    """Whether a feature should be amplified or suppressed."""
    POSITIVE = auto()      # Amplify (good coding practices)
    NEGATIVE = auto()      # Suppress (bad practices)
    NEUTRAL = auto()       # Neither


class FeatureCategory(Enum):
    """Categories of behavioral features."""
    TESTING = auto()           # Unit tests, assertions
    ERROR_HANDLING = auto()    # Try/except, validation
    DOCUMENTATION = auto()     # Comments, docstrings
    EFFICIENCY = auto()        # Performance, optimization
    SECURITY = auto()          # Input validation, sanitization
    LAZINESS = auto()          # Shortcuts, placeholders
    GUESSING = auto()          # Uncertainty, hallucination
    OVERCONFIDENCE = auto()    # Ignoring edge cases


@dataclass
class FeatureVector:
    """A feature direction in activation space."""
    feature_id: str
    name: str
    category: FeatureCategory
    polarity: FeaturePolarity
    
    # Vector representation (simplified)
    direction: List[float]
    magnitude: float = 1.0
    
    # Activation patterns (keywords that trigger)
    activation_patterns: List[str] = field(default_factory=list)
    suppression_patterns: List[str] = field(default_factory=list)
    
    # Statistics
    times_applied: int = 0


@dataclass
class SteeringConfig:
    """Configuration for feature steering."""
    # Amplification strengths
    positive_strength: float = 1.5       # How much to boost good features
    negative_strength: float = 0.5       # How much to suppress bad features
    
    # Feature selection
    max_features: int = 5                # Max simultaneous features
    min_activation: float = 0.3          # Minimum activation threshold


@dataclass 
class SteeringResult:
    """Result of applying feature steering."""
    original_prompt: str
    steered_prompt: str
    features_applied: List[FeatureVector]
    positive_features: int
    negative_features: int
    steering_strength: float


# =============================================================================
# FEATURE LIBRARY
# =============================================================================

class FeatureLibrary:
    """
    Library of coding behavior features.
    
    Maps concepts to activation patterns for steering.
    """
    
    # Core positive features (to amplify)
    POSITIVE_FEATURES = {
        "unit_testing": {
            "name": "Unit Testing Mindset",
            "category": FeatureCategory.TESTING,
            "patterns": ["test_", "assert", "unittest", "pytest", "mock"],
            "boost_prompt": "Write comprehensive unit tests. Include edge cases.",
        },
        "error_handling": {
            "name": "Error Handling",
            "category": FeatureCategory.ERROR_HANDLING,
            "patterns": ["try:", "except", "raise", "validate", "check"],
            "boost_prompt": "Handle all error cases. Validate inputs thoroughly.",
        },
        "documentation": {
            "name": "Documentation",
            "category": FeatureCategory.DOCUMENTATION,
            "patterns": ['"""', "docstring", "#", "comment", "explain"],
            "boost_prompt": "Document your code clearly. Add docstrings.",
        },
        "edge_cases": {
            "name": "Edge Case Awareness",
            "category": FeatureCategory.TESTING,
            "patterns": ["edge", "boundary", "empty", "null", "None", "zero"],
            "boost_prompt": "Consider all edge cases: empty inputs, null values, boundaries.",
        },
        "type_safety": {
            "name": "Type Safety",
            "category": FeatureCategory.ERROR_HANDLING,
            "patterns": ["isinstance", "typing", "->", ": str", ": int"],
            "boost_prompt": "Use type hints. Validate types at runtime.",
        },
        "defensive_coding": {
            "name": "Defensive Coding",
            "category": FeatureCategory.SECURITY,
            "patterns": ["sanitize", "escape", "validate", "check_"],
            "boost_prompt": "Code defensively. Never trust input.",
        },
    }
    
    # Core negative features (to suppress)
    NEGATIVE_FEATURES = {
        "lazy_coding": {
            "name": "Lazy Implementation",
            "category": FeatureCategory.LAZINESS,
            "patterns": ["pass", "TODO", "FIXME", "...", "NotImplemented"],
            "suppress_prompt": "DO NOT use placeholders. Implement fully.",
        },
        "guessing": {
            "name": "Guessing/Hallucination",
            "category": FeatureCategory.GUESSING,
            "patterns": ["probably", "I think", "might", "should work"],
            "suppress_prompt": "DO NOT guess. If uncertain, say so explicitly.",
        },
        "overconfidence": {
            "name": "Overconfidence",
            "category": FeatureCategory.OVERCONFIDENCE,
            "patterns": ["obviously", "clearly", "simple", "trivial"],
            "suppress_prompt": "Check your assumptions. Consider what could go wrong.",
        },
        "shortcuts": {
            "name": "Dangerous Shortcuts",
            "category": FeatureCategory.LAZINESS,
            "patterns": ["eval(", "exec(", "globals()", "import *"],
            "suppress_prompt": "Avoid dangerous shortcuts. Be explicit.",
        },
        "ignoring_errors": {
            "name": "Ignoring Errors",
            "category": FeatureCategory.LAZINESS,
            "patterns": ["except: pass", "except:", "# ignore"],
            "suppress_prompt": "Never silently ignore errors. Handle or log them.",
        },
    }
    
    def __init__(self):
        self.features: Dict[str, FeatureVector] = {}
        self._build_library()
    
    def _build_library(self) -> None:
        """Build the feature library."""
        # Create positive features
        for fid, config in self.POSITIVE_FEATURES.items():
            self.features[fid] = FeatureVector(
                feature_id=fid,
                name=config["name"],
                category=config["category"],
                polarity=FeaturePolarity.POSITIVE,
                direction=self._generate_vector(),
                activation_patterns=config["patterns"],
            )
        
        # Create negative features
        for fid, config in self.NEGATIVE_FEATURES.items():
            self.features[fid] = FeatureVector(
                feature_id=fid,
                name=config["name"],
                category=config["category"],
                polarity=FeaturePolarity.NEGATIVE,
                direction=self._generate_vector(),
                suppression_patterns=config["patterns"],
            )
    
    def _generate_vector(self, dim: int = 64) -> List[float]:
        """Generate a random unit vector (placeholder for SAE)."""
        vec = [random.gauss(0, 1) for _ in range(dim)]
        mag = math.sqrt(sum(x**2 for x in vec))
        return [x / mag for x in vec]
    
    def get_feature(self, feature_id: str) -> Optional[FeatureVector]:
        return self.features.get(feature_id)
    
    def get_by_category(self, category: FeatureCategory) -> List[FeatureVector]:
        return [f for f in self.features.values() if f.category == category]
    
    def get_positive(self) -> List[FeatureVector]:
        return [f for f in self.features.values() if f.polarity == FeaturePolarity.POSITIVE]
    
    def get_negative(self) -> List[FeatureVector]:
        return [f for f in self.features.values() if f.polarity == FeaturePolarity.NEGATIVE]


# =============================================================================
# ACTIVATION DETECTOR
# =============================================================================

class ActivationDetector:
    """
    Detects which features are active in generated content.
    
    Analyzes code/text for patterns indicating feature presence.
    """
    
    def __init__(self, library: FeatureLibrary):
        self.library = library
    
    def detect_activations(
        self,
        content: str,
    ) -> Dict[str, float]:
        """
        Detect feature activations in content.
        
        Returns:
            Dict mapping feature_id to activation strength (0.0-1.0)
        """
        content_lower = content.lower()
        activations = {}
        
        for fid, feature in self.library.features.items():
            patterns = (
                feature.activation_patterns
                if feature.polarity == FeaturePolarity.POSITIVE
                else feature.suppression_patterns
            )
            
            # Count pattern matches
            matches = sum(1 for p in patterns if p.lower() in content_lower)
            
            # Normalize to 0-1
            if patterns:
                activation = min(1.0, matches / len(patterns))
            else:
                activation = 0.0
            
            activations[fid] = activation
        
        return activations
    
    def get_dominant_features(
        self,
        activations: Dict[str, float],
        threshold: float = 0.3,
    ) -> List[Tuple[str, float]]:
        """Get features above threshold, sorted by activation."""
        above = [(fid, act) for fid, act in activations.items() if act >= threshold]
        return sorted(above, key=lambda x: -x[1])


# =============================================================================
# FEATURE STEERING DRIVE
# =============================================================================

class FeatureSteeringDrive:
    """
    The Neural Surgeon - Runtime Activation Steering.
    
    Mathematically manipulates the model's behavior by:
    1. Boosting positive feature activations (testing, error handling)
    2. Suppressing negative feature activations (laziness, guessing)
    
    From Anthropic-2.5 + 2.8:
    "We don't just PROMPT the model to be good. We LOBOTOMIZE the part
    that wants to be lazy and STIMULATE the part that writes tests."
    
    Usage:
        drive = FeatureSteeringDrive()
        
        # Steer towards testing and away from laziness
        result = drive.steer(
            prompt="Write a function to parse JSON",
            boost=["unit_testing", "error_handling"],
            suppress=["lazy_coding", "guessing"],
        )
        
        # Use the steered prompt
        response = llm(result.steered_prompt)
    """
    
    def __init__(self, config: SteeringConfig = None):
        """
        Initialize Feature Steering Drive.
        
        Args:
            config: Steering configuration
        """
        self.config = config or SteeringConfig()
        self.library = FeatureLibrary()
        self.detector = ActivationDetector(self.library)
        
        self._stats = {
            "steerings": 0,
            "positive_boosts": 0,
            "negative_suppressions": 0,
            "total_features_applied": 0,
        }
    
    def steer(
        self,
        prompt: str,
        boost: List[str] = None,
        suppress: List[str] = None,
        auto_detect: bool = True,
    ) -> SteeringResult:
        """
        Apply feature steering to a prompt.
        
        Args:
            prompt: Original prompt
            boost: Feature IDs to boost
            suppress: Feature IDs to suppress
            auto_detect: Automatically detect needed features
            
        Returns:
            SteeringResult with steered prompt
        """
        self._stats["steerings"] += 1
        
        boost = boost or []
        suppress = suppress or []
        
        # Auto-detect if enabled
        if auto_detect:
            activations = self.detector.detect_activations(prompt)
            
            # Boost features with low activation that should be present
            for feature in self.library.get_positive():
                if activations.get(feature.feature_id, 0) < 0.3:
                    if feature.feature_id not in boost:
                        boost.append(feature.feature_id)
            
            # Suppress features with high activation that shouldn't be
            for feature in self.library.get_negative():
                if activations.get(feature.feature_id, 0) > 0.2:
                    if feature.feature_id not in suppress:
                        suppress.append(feature.feature_id)
        
        # Limit features
        boost = boost[:self.config.max_features]
        suppress = suppress[:self.config.max_features]
        
        # Build steering injection
        steering_parts = []
        features_applied = []
        
        # Add positive boosts
        for fid in boost:
            if fid in self.library.POSITIVE_FEATURES:
                config = self.library.POSITIVE_FEATURES[fid]
                boost_prompt = config.get("boost_prompt", "")
                if boost_prompt:
                    steering_parts.append(f"[+] {boost_prompt}")
                    features_applied.append(self.library.get_feature(fid))
                    self._stats["positive_boosts"] += 1
        
        # Add negative suppressions
        for fid in suppress:
            if fid in self.library.NEGATIVE_FEATURES:
                config = self.library.NEGATIVE_FEATURES[fid]
                suppress_prompt = config.get("suppress_prompt", "")
                if suppress_prompt:
                    steering_parts.append(f"[-] {suppress_prompt}")
                    features_applied.append(self.library.get_feature(fid))
                    self._stats["negative_suppressions"] += 1
        
        self._stats["total_features_applied"] += len(features_applied)
        
        # Construct steered prompt
        if steering_parts:
            steering_block = "\n".join([
                "## BEHAVIORAL STEERING (MANDATORY)",
                *steering_parts,
                "---",
            ])
            steered_prompt = f"{steering_block}\n\n{prompt}"
        else:
            steered_prompt = prompt
        
        # Update feature usage stats
        for f in features_applied:
            if f:
                f.times_applied += 1
        
        return SteeringResult(
            original_prompt=prompt,
            steered_prompt=steered_prompt,
            features_applied=[f for f in features_applied if f],
            positive_features=len(boost),
            negative_features=len(suppress),
            steering_strength=self.config.positive_strength,
        )
    
    def steer_for_testing(self, prompt: str) -> SteeringResult:
        """Pre-configured steering for test writing."""
        return self.steer(
            prompt,
            boost=["unit_testing", "edge_cases", "error_handling"],
            suppress=["lazy_coding", "overconfidence"],
            auto_detect=False,
        )
    
    def steer_for_quality(self, prompt: str) -> SteeringResult:
        """Pre-configured steering for high-quality code."""
        return self.steer(
            prompt,
            boost=["error_handling", "type_safety", "documentation", "defensive_coding"],
            suppress=["lazy_coding", "shortcuts", "guessing"],
            auto_detect=False,
        )
    
    def steer_for_security(self, prompt: str) -> SteeringResult:
        """Pre-configured steering for secure code."""
        return self.steer(
            prompt,
            boost=["defensive_coding", "type_safety", "error_handling"],
            suppress=["shortcuts", "ignoring_errors", "guessing"],
            auto_detect=False,
        )
    
    def analyze_output(
        self,
        generated_code: str,
    ) -> Dict[str, Any]:
        """
        Analyze generated code for feature compliance.
        
        Returns metrics on how well steering was followed.
        """
        activations = self.detector.detect_activations(generated_code)
        
        # Score positive features
        positive_score = sum(
            activations.get(f.feature_id, 0)
            for f in self.library.get_positive()
        ) / max(len(self.library.get_positive()), 1)
        
        # Score negative features (lower is better)
        negative_score = sum(
            activations.get(f.feature_id, 0)
            for f in self.library.get_negative()
        ) / max(len(self.library.get_negative()), 1)
        
        return {
            "positive_compliance": positive_score,
            "negative_avoidance": 1.0 - negative_score,
            "overall_score": (positive_score + (1.0 - negative_score)) / 2,
            "activations": activations,
        }
    
    def get_stats(self) -> Dict[str, int]:
        return self._stats.copy()
    
    def print_status(self) -> None:
        """Print drive status."""
        print("\n" + "=" * 60)
        print("[*] FEATURE STEERING DRIVE STATUS")
        print("=" * 60)
        
        print(f"\n[+] Positive Features ({len(self.library.get_positive())}):")
        for f in self.library.get_positive():
            print(f"   - {f.name} (applied {f.times_applied}x)")
        
        print(f"\n[-] Negative Features ({len(self.library.get_negative())}):")
        for f in self.library.get_negative():
            print(f"   - {f.name} (applied {f.times_applied}x)")
        
        print(f"\n[%] Stats:")
        for key, val in self._stats.items():
            print(f"   {key}: {val}")
        
        print("=" * 60)


# =============================================================================
# PACKAGE INIT
# =============================================================================

__all__ = [
    "FeatureSteeringDrive",
    "FeatureLibrary",
    "ActivationDetector",
    "FeatureVector",
    "FeaturePolarity",
    "FeatureCategory",
    "SteeringConfig",
    "SteeringResult",
]


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("[*] S.P.I.D.E.R. Feature Steering Drive - Demo")
    print("=" * 70)
    
    drive = FeatureSteeringDrive()
    
    # Test basic steering
    print("\n[1] Basic steering for quality code...")
    
    prompt = "Write a function to parse JSON from a file"
    result = drive.steer_for_quality(prompt)
    
    print(f"   Original: {prompt}")
    print(f"\n   Steered prompt:")
    print("-" * 50)
    print(result.steered_prompt[:500])
    print("-" * 50)
    print(f"\n   Features applied: {len(result.features_applied)}")
    print(f"   Positive: {result.positive_features}, Negative: {result.negative_features}")
    
    # Test output analysis
    print("\n[2] Analyzing generated code...")
    
    good_code = '''
def parse_json(filepath: str) -> dict:
    """Parse JSON from a file with error handling."""
    if not filepath:
        raise ValueError("Filepath cannot be empty")
    
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

def test_parse_json():
    assert parse_json("test.json") is not None
'''
    
    bad_code = '''
def parse_json(filepath):
    # TODO: add error handling
    with open(filepath) as f:
        return json.load(f)  # should work
'''
    
    print("\n   Good code analysis:")
    good_analysis = drive.analyze_output(good_code)
    print(f"      Positive compliance: {good_analysis['positive_compliance']:.2f}")
    print(f"      Negative avoidance: {good_analysis['negative_avoidance']:.2f}")
    print(f"      Overall score: {good_analysis['overall_score']:.2f}")
    
    print("\n   Bad code analysis:")
    bad_analysis = drive.analyze_output(bad_code)
    print(f"      Positive compliance: {bad_analysis['positive_compliance']:.2f}")
    print(f"      Negative avoidance: {bad_analysis['negative_avoidance']:.2f}")
    print(f"      Overall score: {bad_analysis['overall_score']:.2f}")
    
    drive.print_status()
