"""
S.P.I.D.E.R. Chameleon Engine - Adaptive Core
==============================================

This package implements the self-adaptive meta-tuning system based on:
- Hypernetworks (HYPERFORMER++) for dynamic weight generation
- Hyperparameter Optimization (HPO) for automated configuration
- Task embeddings for zero-shot configuration prediction

Components:
- projector.py: Task Projector - creates embeddings from code fingerprints
- tuner.py: Hyper-Tuner - warm-start HPO with rugged landscape navigation
- config.py: Dynamic configuration generation

The Chameleon Engine transforms S.P.I.D.E.R. from a static architecture
to a fluid, self-modifying system that adapts to each problem type.
"""

from .projector import TaskProjector, TaskFingerprint, TaskDomain
from .tuner import HyperTuner, SolverConfig, TuningHistory
from .config import ChameleonConfig, ConfigPredictor

__all__ = [
    "TaskProjector",
    "TaskFingerprint", 
    "TaskDomain",
    "HyperTuner",
    "SolverConfig",
    "TuningHistory",
    "ChameleonConfig",
    "ConfigPredictor",
]
