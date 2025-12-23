"""
S.P.I.D.E.R. Chameleon Config - Unified Adaptive Configuration
===============================================================

From HPO-2 Paper:
"Amortized HPO via Latent Task Embeddings... predicts the optimal
region in the hyperparameter search space immediately."

The ChameleonConfig ties together:
1. TaskProjector output (fingerprint)
2. HyperTuner output (solver config)
3. Runtime adjustments (feedback-based)

This creates "Zero-Shot Configuration" - instant optimal setup
without trial and error.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .projector import TaskDomain, TaskFingerprint, TaskProjector
from .tuner import HyperTuner, SolverConfig, TuningHistory

logger = logging.getLogger(__name__)


# =============================================================================
# CHAMELEON CONFIG
# =============================================================================

@dataclass
class ChameleonConfig:
    """
    The unified adaptive configuration for a task.
    
    This is what the Chameleon Engine produces - a complete
    specification of how S.P.I.D.E.R. should behave for this
    specific task.
    """
    
    # Task Understanding
    fingerprint: TaskFingerprint
    task_id: str
    
    # Solver Configuration
    solver_config: SolverConfig
    
    # Runtime Hints
    recommended_tools: List[str] = field(default_factory=list)
    disabled_tools: List[str] = field(default_factory=list)
    
    # Prompt Strategy
    prompt_style: str = "chain_of_thought"    # cot, direct, few_shot
    persona: str = "senior_engineer"          # expert role
    
    # Verification Strategy
    verification_priority: List[str] = field(default_factory=lambda: ["test", "z3", "lint"])
    skip_verification: List[str] = field(default_factory=list)
    
    # Meta
    confidence: float = 0.5                   # How confident in this config
    generation_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "domain": self.fingerprint.primary_domain.name,
            "solver_config": self.solver_config.to_dict(),
            "prompt_style": self.prompt_style,
            "persona": self.persona,
            "recommended_tools": self.recommended_tools,
            "disabled_tools": self.disabled_tools,
            "verification_priority": self.verification_priority,
            "confidence": self.confidence,
        }
    
    def summary(self) -> str:
        """One-line summary of the config."""
        return (
            f"[{self.fingerprint.primary_domain.name}] "
            f"temp={self.solver_config.temperature:.1f} "
            f"mcts={self.solver_config.mcts_iterations} "
            f"z3={'ON' if self.solver_config.z3_enabled else 'OFF'} "
            f"conf={self.confidence:.0%}"
        )


# =============================================================================
# PERSONA MAPPINGS
# =============================================================================

DOMAIN_PERSONAS = {
    TaskDomain.FRONTEND: {
        "persona": "senior_frontend_engineer",
        "description": "Expert in CSS, HTML, JavaScript, and UI/UX best practices",
        "prompt_style": "direct",
    },
    TaskDomain.BACKEND: {
        "persona": "senior_backend_engineer",
        "description": "Expert in APIs, server architecture, and business logic",
        "prompt_style": "chain_of_thought",
    },
    TaskDomain.DATABASE: {
        "persona": "database_architect",
        "description": "Expert in SQL optimization, schema design, and data integrity",
        "prompt_style": "chain_of_thought",
    },
    TaskDomain.SYSTEMS: {
        "persona": "systems_engineer",
        "description": "Expert in threading, memory management, and low-level systems",
        "prompt_style": "metacognitive",
    },
    TaskDomain.ML_DATA: {
        "persona": "ml_engineer",
        "description": "Expert in NumPy, Pandas, and machine learning pipelines",
        "prompt_style": "chain_of_thought",
    },
    TaskDomain.TESTING: {
        "persona": "qa_engineer",
        "description": "Expert in test design, coverage, and quality assurance",
        "prompt_style": "direct",
    },
    TaskDomain.DEVOPS: {
        "persona": "devops_engineer",
        "description": "Expert in CI/CD, containers, and infrastructure",
        "prompt_style": "direct",
    },
    TaskDomain.UNKNOWN: {
        "persona": "senior_engineer",
        "description": "Experienced software engineer with broad expertise",
        "prompt_style": "chain_of_thought",
    },
}


DOMAIN_TOOLS = {
    TaskDomain.FRONTEND: {
        "recommended": ["read_file", "search_code", "apply_patch"],
        "disabled": ["run_tests"],  # Often no tests for CSS
    },
    TaskDomain.BACKEND: {
        "recommended": ["read_file", "search_code", "run_tests", "apply_patch"],
        "disabled": [],
    },
    TaskDomain.DATABASE: {
        "recommended": ["read_file", "search_code", "run_tests"],
        "disabled": [],
    },
    TaskDomain.SYSTEMS: {
        "recommended": ["read_file", "search_code", "run_tests", "apply_patch", "analyze_code"],
        "disabled": [],
    },
    TaskDomain.ML_DATA: {
        "recommended": ["read_file", "search_code", "run_tests"],
        "disabled": [],
    },
    TaskDomain.TESTING: {
        "recommended": ["read_file", "search_code", "run_tests"],
        "disabled": [],
    },
    TaskDomain.DEVOPS: {
        "recommended": ["read_file", "search_code"],
        "disabled": ["run_tests"],
    },
    TaskDomain.UNKNOWN: {
        "recommended": ["read_file", "search_code", "run_tests", "apply_patch"],
        "disabled": [],
    },
}


# =============================================================================
# CONFIG PREDICTOR
# =============================================================================

class ConfigPredictor:
    """
    Zero-Shot Configuration Predictor.
    
    From HPO-2 Paper:
    "We utilize the Task Embeddings from the Hypernetwork to
    Warm-Start the HPO process... The model predicts the optimal
    region in the hyperparameter search space immediately."
    
    This is the main interface for the Chameleon Engine.
    It takes a problem description and produces a complete
    adaptive configuration instantly.
    
    Usage:
        predictor = ConfigPredictor()
        
        config = predictor.predict(
            problem="Fix the mutex deadlock in worker.py",
            code_context="def worker(): lock.acquire()...",
        )
        
        print(config.summary())
        # [SYSTEMS] temp=0.1 mcts=200 z3=ON conf=85%
    """
    
    def __init__(
        self,
        history_path: Optional[str] = None,
        enable_learning: bool = True,
    ):
        """
        Initialize the Config Predictor.
        
        Args:
            history_path: Path to persist learning history
            enable_learning: Whether to learn from outcomes
        """
        self.projector = TaskProjector()
        self.tuner = HyperTuner(history_path=history_path)
        self.enable_learning = enable_learning
        
        # Recent predictions for feedback
        self.recent_predictions: Dict[str, ChameleonConfig] = {}
        
        self.stats = {
            "predictions": 0,
            "feedbacks": 0,
            "avg_confidence": 0.0,
        }
    
    def predict(
        self,
        problem: str,
        code_context: str = "",
        file_path: str = "",
        additional_context: Optional[Dict[str, Any]] = None,
        override: Optional[Dict[str, Any]] = None,
    ) -> ChameleonConfig:
        """
        Predict the optimal configuration for a task.
        
        This is the "Zero-Shot HPO" - instant configuration
        based on task embedding analysis.
        
        Args:
            problem: The problem statement or bug description
            code_context: Related code snippets
            file_path: Path to file being modified
            additional_context: Extra metadata
            override: Manual configuration overrides
            
        Returns:
            ChameleonConfig with complete solver setup
        """
        start_time = time.time()
        self.stats["predictions"] += 1
        
        # Step 1: Project task to fingerprint
        fingerprint = self.projector.project(
            problem_statement=problem,
            code_context=code_context,
            file_path=file_path,
            additional_context=additional_context,
        )
        
        # Step 2: Tune solver configuration
        solver_config = self.tuner.tune(fingerprint, override)
        
        # Step 3: Determine persona and tools
        domain = fingerprint.primary_domain
        persona_info = DOMAIN_PERSONAS.get(domain, DOMAIN_PERSONAS[TaskDomain.UNKNOWN])
        tool_info = DOMAIN_TOOLS.get(domain, DOMAIN_TOOLS[TaskDomain.UNKNOWN])
        
        # Step 4: Determine verification strategy
        verification = self._determine_verification(fingerprint, solver_config)
        
        # Step 5: Calculate confidence
        confidence = self._calculate_confidence(fingerprint)
        
        # Step 6: Generate task ID
        task_id = hashlib.md5(
            f"{problem[:100]}{time.time()}".encode()
        ).hexdigest()[:12]
        
        # Build config
        config = ChameleonConfig(
            fingerprint=fingerprint,
            task_id=task_id,
            solver_config=solver_config,
            recommended_tools=tool_info["recommended"],
            disabled_tools=tool_info["disabled"],
            prompt_style=persona_info["prompt_style"],
            persona=persona_info["persona"],
            verification_priority=verification["priority"],
            skip_verification=verification["skip"],
            confidence=confidence,
            generation_time_ms=(time.time() - start_time) * 1000,
        )
        
        # Cache for feedback
        self.recent_predictions[task_id] = config
        
        # Update stats
        n = self.stats["predictions"]
        self.stats["avg_confidence"] = (
            self.stats["avg_confidence"] * (n - 1) + confidence
        ) / n
        
        return config
    
    def _determine_verification(
        self,
        fingerprint: TaskFingerprint,
        config: SolverConfig,
    ) -> Dict[str, List[str]]:
        """Determine verification strategy based on task."""
        priority = []
        skip = []
        
        # Tests always first if enabled
        if config.test_enabled:
            priority.append("test")
        else:
            skip.append("test")
        
        # Z3 second if enabled and useful
        if config.z3_enabled:
            # Skip Z3 for domains where it doesn't help
            if fingerprint.primary_domain in [TaskDomain.FRONTEND, TaskDomain.DEVOPS]:
                skip.append("z3")
            else:
                priority.append("z3")
        else:
            skip.append("z3")
        
        # Lint always useful
        priority.append("lint")
        
        # Type check for typed code
        priority.append("type_check")
        
        return {"priority": priority, "skip": skip}
    
    def _calculate_confidence(self, fingerprint: TaskFingerprint) -> float:
        """
        Calculate confidence in the configuration.
        
        Higher confidence when:
        - Clear domain classification
        - Similar to past successful tasks
        - Low complexity
        """
        # Domain clarity (entropy of domain scores)
        scores = list(fingerprint.domain_scores.values())
        max_score = max(scores) if scores else 0
        domain_clarity = max_score  # Higher = clearer domain
        
        # Complexity penalty
        complexity_factor = 1.0 - (fingerprint.complexity_score * 0.3)
        
        # Risk penalty
        risk_factor = 1.0 - (fingerprint.risk_level * 0.2)
        
        # History bonus
        similar = self.projector.find_similar(fingerprint, threshold=0.8)
        history_bonus = min(0.2, len(similar) * 0.05)
        
        confidence = (
            domain_clarity * 0.4 +
            complexity_factor * 0.3 +
            risk_factor * 0.2 +
            history_bonus
        )
        
        return min(1.0, max(0.0, confidence))
    
    def feedback(
        self,
        task_id: str,
        success: bool,
        score: float = 0.0,
        duration_ms: float = 0.0,
        iterations_used: int = 0,
    ) -> None:
        """
        Provide feedback on a prediction outcome.
        
        This enables learning for future predictions.
        
        Args:
            task_id: The task_id from ChameleonConfig
            success: Whether the task was solved
            score: Quality score (0.0 to 1.0)
            duration_ms: Time taken
            iterations_used: Self-correction iterations used
        """
        if not self.enable_learning:
            return
        
        if task_id not in self.recent_predictions:
            logger.warning(f"Unknown task_id for feedback: {task_id}")
            return
        
        config = self.recent_predictions[task_id]
        
        # Record in tuner history
        self.tuner.record(
            fingerprint=config.fingerprint,
            config=config.solver_config,
            success=success,
            score=score,
            duration_ms=duration_ms,
            iterations_used=iterations_used,
        )
        
        self.stats["feedbacks"] += 1
        
        # Cleanup old predictions
        if len(self.recent_predictions) > 100:
            # Remove oldest
            oldest = min(self.recent_predictions.keys())
            del self.recent_predictions[oldest]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get predictor statistics."""
        return {
            **self.stats,
            "projector": self.projector.get_stats(),
            "tuner": self.tuner.get_stats(),
        }
    
    def print_config(self, config: ChameleonConfig) -> None:
        """Pretty-print a configuration."""
        print("\n" + "=" * 60)
        print(f"🦎 CHAMELEON CONFIG: {config.task_id}")
        print("=" * 60)
        print(f"\n📋 Task Analysis:")
        print(f"  Domain:        {config.fingerprint.primary_domain.name}")
        print(f"  Complexity:    {config.fingerprint.complexity_score:.2f}")
        print(f"  Risk Level:    {config.fingerprint.risk_level:.2f}")
        print(f"  Has Threading: {config.fingerprint.has_threading}")
        print(f"  Has Database:  {config.fingerprint.has_database}")
        
        print(f"\n⚙️ Solver Configuration:")
        print(f"  Temperature:   {config.solver_config.temperature}")
        print(f"  MCTS Iter:     {config.solver_config.mcts_iterations}")
        print(f"  Z3 Enabled:    {config.solver_config.z3_enabled}")
        print(f"  Test Enabled:  {config.solver_config.test_enabled}")
        print(f"  Restart:       {config.solver_config.restart_enabled}")
        print(f"  Max Iterations:{config.solver_config.max_iterations}")
        
        print(f"\n🎭 Prompt Strategy:")
        print(f"  Persona:       {config.persona}")
        print(f"  Style:         {config.prompt_style}")
        
        print(f"\n🔧 Tools:")
        print(f"  Recommended:   {', '.join(config.recommended_tools)}")
        print(f"  Disabled:      {', '.join(config.disabled_tools) or 'None'}")
        
        print(f"\n✅ Verification Priority: {' → '.join(config.verification_priority)}")
        print(f"📊 Confidence: {config.confidence:.0%}")
        print(f"⏱️ Generation Time: {config.generation_time_ms:.1f}ms")
        print("=" * 60)


# =============================================================================
# CHAMELEON ENGINE (UNIFIED INTERFACE)
# =============================================================================

class ChameleonEngine:
    """
    The Chameleon Engine - Self-Adaptive Meta-Tuning System.
    
    This is the unified interface combining:
    - TaskProjector: Fingerprint generation
    - HyperTuner: Configuration optimization
    - ConfigPredictor: Zero-shot prediction
    
    The Engine transforms S.P.I.D.E.R. from a static architecture
    to a fluid, self-modifying system.
    
    Usage:
        engine = ChameleonEngine()
        
        # Get adaptive config for any task
        config = engine.adapt("Fix the CSS padding bug...")
        
        # Access components
        print(config.solver_config.temperature)
        print(config.persona)
    """
    
    def __init__(self, history_path: Optional[str] = None):
        self.predictor = ConfigPredictor(
            history_path=history_path,
            enable_learning=True,
        )
    
    def adapt(
        self,
        problem: str,
        code: str = "",
        **kwargs,
    ) -> ChameleonConfig:
        """
        Adapt S.P.I.D.E.R. to a specific task.
        
        Returns a complete configuration optimized for this task.
        """
        return self.predictor.predict(
            problem=problem,
            code_context=code,
            **kwargs,
        )
    
    def learn(
        self,
        task_id: str,
        success: bool,
        **kwargs,
    ) -> None:
        """Learn from a task outcome."""
        self.predictor.feedback(task_id, success, **kwargs)
    
    @property
    def projector(self) -> TaskProjector:
        return self.predictor.projector
    
    @property
    def tuner(self) -> HyperTuner:
        return self.predictor.tuner
    
    def get_stats(self) -> Dict[str, Any]:
        return self.predictor.get_stats()


# =============================================================================
# MAIN (DEMO)
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🦎 S.P.I.D.E.R. CHAMELEON ENGINE - Full Demo")
    print("=" * 70)
    
    engine = ChameleonEngine()
    
    scenarios = [
        {
            "name": "Frontend: CSS Bug",
            "problem": "The button padding is 2px off in Safari. The color doesn't match.",
            "code": "def render_button(style): return f'<button style=\"{style}\">'",
        },
        {
            "name": "Systems: Race Condition",
            "problem": "Multiple threads accessing shared counter causes incorrect values. Need mutex.",
            "code": "import threading\nlock = threading.Lock()\ndef increment():\n    global counter\n    counter += 1",
        },
        {
            "name": "Database: SQL Injection",
            "problem": "User input not sanitized in query. SQL injection vulnerability.",
            "code": "cursor.execute(f'SELECT * FROM users WHERE id = {user_id}')",
        },
    ]
    
    for scenario in scenarios:
        print(f"\n{'='*70}")
        print(f"Scenario: {scenario['name']}")
        print("="*70)
        
        config = engine.adapt(
            problem=scenario["problem"],
            code=scenario["code"],
        )
        
        engine.predictor.print_config(config)
        
        # Simulate outcome
        engine.learn(config.task_id, success=True, score=0.95)
    
    print(f"\n{'='*70}")
    print("Engine Stats:")
    print(json.dumps(engine.get_stats(), indent=2, default=str))
