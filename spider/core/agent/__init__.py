"""S.P.I.D.E.R. Agent Module - Full Agentic Capabilities."""

# Core reasoning
from spider.core.agent.reasoning import ReasoningCore, ReasoningConfig, AnalysisResult, CheckStage
from spider.core.agent.architect import Architect, ArchitectConfig

# LLM Gateway
from spider.core.agent.llm_client import LLMGateway, Message, MessageRole, LLMResponse

# Agentic Core (ReAct pattern)
from spider.core.agent.agentic import (
    Tool, ToolResult, ToolStatus, ToolRegistry,
    ReActAgent, AgentConfig, AgentStep,
)

# Multi-Agent System
from spider.core.agent.multiagent import (
    SpecializedAgent, PlannerAgent, CoderAgent, TesterAgent, ReviewerAgent,
    AgentOrchestrator, TeamConfig, ContextManager,
    AgentMessage, MessageType,
)

# Ultimate Solver
from spider.core.agent.ultimate import UltimateSolver, UltimateSolverConfig, SolverMode

# MCTS - import lazily due to heavy dependencies
# from spider.core.agent.mcts import TimeTraveler, CodeNode

# Anthropic Counter-Systems
from spider.core.agent.governor import (
    EntropyGovernor, EntropyCalculator, MCTSSimulator,
    GovernorDecision, ComputeLevel, ConfidenceState,
)
from spider.core.agent.state_handoff import (
    StateVectorHandoff, StateSerializer, StateVector,
    StateComponent, StateType, HandoffConfig,
)

__all__ = [
    # Reasoning
    'ReasoningCore', 'ReasoningConfig', 'AnalysisResult', 'CheckStage',
    'Architect', 'ArchitectConfig',
    
    # LLM
    'LLMGateway', 'Message', 'MessageRole', 'LLMResponse',
    
    # Agentic
    'Tool', 'ToolResult', 'ToolStatus', 'ToolRegistry',
    'ReActAgent', 'AgentConfig', 'AgentStep',
    
    # Multi-Agent
    'SpecializedAgent', 'PlannerAgent', 'CoderAgent', 'TesterAgent', 'ReviewerAgent',
    'AgentOrchestrator', 'TeamConfig', 'ContextManager',
    'AgentMessage', 'MessageType',
    
    # Ultimate
    'UltimateSolver', 'UltimateSolverConfig', 'SolverMode',
    
    # Anthropic Counter-Systems
    'EntropyGovernor', 'EntropyCalculator', 'MCTSSimulator',
    'GovernorDecision', 'ComputeLevel', 'ConfidenceState',
    'StateVectorHandoff', 'StateSerializer', 'StateVector',
    'StateComponent', 'StateType', 'HandoffConfig',
]

