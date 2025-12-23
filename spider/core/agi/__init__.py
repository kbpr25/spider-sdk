"""
S.P.I.D.E.R. AGI Package
========================

The Soul of S.P.I.D.E.R. - Moving from Narrow AI to AGI.

Components:
- ToolGenesis: Self-extending AI that invents its own tools
- DigitalTwin: Neuro-symbolic world model for codebase topology
- FractalPlanner: Recursive chain-of-thought with alignment

This is the Event Horizon.
"""

from .tool_genesis import (
    ToolFabricator,
    ToolSynthesizer,
    ToolSandbox,
    CapabilityAnalyzer,
    SynthesizedTool,
    ToolSpec,
    ToolStatus,
    CapabilityGap,
)

from .digital_twin import (
    DigitalTwin,
    WorldGraph,
    AttentionCalculator,
    Entity,
    Relationship,
    EntityType,
    RelationType,
    ImpactAnalysis,
    ConceptMapping,
)

from .fractal_planner import (
    FractalPlanner,
    TaskDecomposer,
    AlignmentManager,
    Task,
    PlanNode,
    TaskStatus,
    AlignmentViolation,
    AlignmentConfig,
    ExecutionResult,
)


__all__ = [
    # Tool Genesis
    "ToolFabricator",
    "ToolSynthesizer",
    "ToolSandbox",
    "CapabilityAnalyzer",
    "SynthesizedTool",
    "ToolSpec",
    "CapabilityGap",
    
    # Digital Twin
    "DigitalTwin",
    "WorldGraph",
    "AttentionCalculator",
    "Entity",
    "Relationship",
    "EntityType",
    "RelationType",
    "ImpactAnalysis",
    "ConceptMapping",
    
    # Fractal Planner
    "FractalPlanner",
    "TaskDecomposer",
    "AlignmentManager",
    "Task",
    "PlanNode",
    "TaskStatus",
    "AlignmentViolation",
    "AlignmentConfig",
    "ExecutionResult",
]
