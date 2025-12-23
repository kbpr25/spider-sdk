"""
S.P.I.D.E.R. Debug Package
==========================

The Debugging Trinity for autonomous bug repair.

Components:
- RubberDuckEngine: Trace-based debugging with execution state capture
- ReflexionBuffer: Episodic learning memory for debugging lessons
- FaultLocalizer: SBFL-based fault localization

Born from: Debug-1 through Debug-5 research papers
"""

from .rubber_duck import (
    RubberDuckAgent,
    TraceCapture,
    ExecutionTrace,
    Diagnosis,
    TraceEntry,
    StateChange,
)

from .reflexion_buffer import (
    ReflexionBuffer,
    Lesson,
    LessonQuery,
    LessonStore,
)

from .fault_localizer import (
    FaultLocalizer,
    LocalizationResult,
    FaultLocation,
    LineCoverage,
    SBFLMetric,
    SBFLCalculator,
)


__all__ = [
    # Rubber Duck
    "RubberDuckAgent",
    "TraceCapture",
    "ExecutionTrace",
    "Diagnosis",
    "TraceEntry",
    "StateChange",
    
    # Reflexion Buffer
    "ReflexionBuffer",
    "Lesson",
    "LessonQuery",
    "LessonStore",
    
    # Fault Localizer
    "FaultLocalizer",
    "LocalizationResult",
    "FaultLocation",
    "LineCoverage",
    "SBFLMetric",
    "SBFLCalculator",
]
