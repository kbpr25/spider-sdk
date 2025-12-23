"""
S.P.I.D.E.R. Dynamic Tool Selection
====================================

Adaptive tool selection based on task phase and learned patterns.
Reduces LLM confusion and token waste by presenting only relevant tools.

Phases:
1. EXPLORATION: Read, search, list - understand the codebase
2. ANALYSIS: Focus on understanding, no modification tools
3. PLANNING: Create a fix strategy
4. IMPLEMENTATION: Write, apply patches - make changes
5. VERIFICATION: Run tests - validate changes

Key Insight:
Presenting ALL tools at ALL times confuses smaller LLMs.
By limiting tools to task-relevant ones, we improve focus and accuracy.

This is the +3% improvement component.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# TASK PHASES
# =============================================================================

class TaskPhase(Enum):
    """Phases of a task lifecycle."""
    EXPLORATION = auto()      # Understanding the codebase
    ANALYSIS = auto()         # Analyzing the problem
    PLANNING = auto()         # Creating a fix strategy
    IMPLEMENTATION = auto()   # Making changes
    VERIFICATION = auto()     # Testing changes
    COMPLETE = auto()         # Task finished


@dataclass
class PhaseConfig:
    """Configuration for a task phase."""
    name: str
    description: str
    allowed_tools: Set[str]
    blocked_tools: Set[str] = field(default_factory=set)
    max_duration_seconds: int = 300
    transition_hints: List[str] = field(default_factory=list)
    
    def is_tool_allowed(self, tool_name: str) -> bool:
        """Check if a tool is allowed in this phase."""
        if tool_name in self.blocked_tools:
            return False
        if not self.allowed_tools:  # Empty means all allowed
            return True
        return tool_name in self.allowed_tools


# Built-in phase configurations
PHASE_CONFIGS = {
    TaskPhase.EXPLORATION: PhaseConfig(
        name="Exploration",
        description="Understand the codebase and locate relevant files",
        allowed_tools={
            "read_file", "search_code", "list_files", 
            "find_definition", "find_references",
        },
        blocked_tools={"write_file", "apply_patch", "run_tests"},
        max_duration_seconds=120,
        transition_hints=[
            "Found the relevant files",
            "Understand the codebase structure",
            "Located the bug",
        ],
    ),
    TaskPhase.ANALYSIS: PhaseConfig(
        name="Analysis",
        description="Analyze the problem and understand root cause",
        allowed_tools={
            "read_file", "search_code", "find_definition",
            "analyze_code", "get_context",
        },
        blocked_tools={"write_file", "apply_patch"},
        max_duration_seconds=180,
        transition_hints=[
            "Understood the root cause",
            "Identified the fix location",
            "Know what needs to change",
        ],
    ),
    TaskPhase.PLANNING: PhaseConfig(
        name="Planning",
        description="Create a fix strategy before implementing",
        allowed_tools={
            "read_file", "get_context",
        },
        blocked_tools={"write_file", "apply_patch", "run_tests"},
        max_duration_seconds=60,
        transition_hints=[
            "Have a clear fix strategy",
            "Know the exact changes needed",
        ],
    ),
    TaskPhase.IMPLEMENTATION: PhaseConfig(
        name="Implementation",
        description="Implement the fix",
        allowed_tools={
            "read_file", "write_file", "apply_patch",
            "search_code", "get_context",
        },
        max_duration_seconds=180,
        transition_hints=[
            "Fix is implemented",
            "Code changes complete",
            "Ready to test",
        ],
    ),
    TaskPhase.VERIFICATION: PhaseConfig(
        name="Verification",
        description="Verify the fix works",
        allowed_tools={
            "run_tests", "read_file", "search_code",
        },
        blocked_tools={"write_file"},  # No modifications during verification
        max_duration_seconds=300,
        transition_hints=[
            "Tests pass",
            "Fix verified",
            "All checks green",
        ],
    ),
    TaskPhase.COMPLETE: PhaseConfig(
        name="Complete",
        description="Task is finished",
        allowed_tools=set(),
        max_duration_seconds=0,
    ),
}


# =============================================================================
# TOOL REGISTRY
# =============================================================================

@dataclass
class ToolMeta:
    """Metadata about a tool."""
    name: str
    description: str
    category: str  # exploration, analysis, modification, verification
    risk_level: int = 0  # 0-10, higher = more risky
    usage_count: int = 0
    success_count: int = 0
    avg_duration_ms: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.usage_count == 0:
            return 0.0
        return self.success_count / self.usage_count
    
    def update_stats(self, success: bool, duration_ms: float) -> None:
        """Update tool statistics."""
        self.usage_count += 1
        if success:
            self.success_count += 1
        
        # Running average
        old_avg = self.avg_duration_ms
        self.avg_duration_ms = (old_avg * (self.usage_count - 1) + duration_ms) / self.usage_count


class ToolRegistry:
    """
    Registry of available tools with metadata.
    
    Tracks:
    - Tool categories
    - Success rates
    - Usage patterns
    """
    
    def __init__(self):
        self.tools: Dict[str, ToolMeta] = {}
        self._register_defaults()
    
    def _register_defaults(self) -> None:
        """Register default tools."""
        defaults = [
            # Exploration tools
            ToolMeta("read_file", "Read file contents", "exploration", risk_level=0),
            ToolMeta("search_code", "Search for patterns in code", "exploration", risk_level=0),
            ToolMeta("list_files", "List directory contents", "exploration", risk_level=0),
            ToolMeta("find_definition", "Find where something is defined", "exploration", risk_level=0),
            ToolMeta("find_references", "Find all references", "exploration", risk_level=0),
            
            # Analysis tools
            ToolMeta("analyze_code", "Analyze code semantics", "analysis", risk_level=0),
            ToolMeta("get_context", "Get context for a function", "analysis", risk_level=0),
            ToolMeta("explain_code", "Explain what code does", "analysis", risk_level=0),
            
            # Modification tools
            ToolMeta("write_file", "Write to a file", "modification", risk_level=5),
            ToolMeta("apply_patch", "Apply a diff patch", "modification", risk_level=7),
            ToolMeta("edit_file", "Edit specific lines", "modification", risk_level=5),
            
            # Verification tools
            ToolMeta("run_tests", "Run test suite", "verification", risk_level=2),
            ToolMeta("lint_code", "Run linter", "verification", risk_level=1),
            ToolMeta("type_check", "Run type checker", "verification", risk_level=1),
        ]
        
        for tool in defaults:
            self.tools[tool.name] = tool
    
    def register(self, tool: ToolMeta) -> None:
        """Register a new tool."""
        self.tools[tool.name] = tool
    
    def get(self, name: str) -> Optional[ToolMeta]:
        """Get tool metadata."""
        return self.tools.get(name)
    
    def get_by_category(self, category: str) -> List[ToolMeta]:
        """Get all tools in a category."""
        return [t for t in self.tools.values() if t.category == category]
    
    def record_usage(
        self, 
        tool_name: str, 
        success: bool, 
        duration_ms: float,
    ) -> None:
        """Record tool usage for learning."""
        tool = self.get(tool_name)
        if tool:
            tool.update_stats(success, duration_ms)


# =============================================================================
# PHASE DETECTOR
# =============================================================================

class PhaseDetector:
    """
    Detects the current phase based on conversation/action history.
    
    Uses heuristics and patterns to determine:
    - When to transition phases
    - What phase we're currently in
    - What phase should come next
    """
    
    def __init__(self):
        self.current_phase = TaskPhase.EXPLORATION
        self.phase_history: List[Tuple[TaskPhase, float]] = []
        self.action_history: List[str] = []
        self.phase_start_time = time.time()
        
        # Transition triggers
        self.phase_keywords = {
            TaskPhase.EXPLORATION: ["found", "located", "identified", "see"],
            TaskPhase.ANALYSIS: ["understand", "cause", "because", "root"],
            TaskPhase.PLANNING: ["plan", "strategy", "will", "should"],
            TaskPhase.IMPLEMENTATION: ["fix", "change", "modify", "implement"],
            TaskPhase.VERIFICATION: ["test", "verify", "check", "run"],
        }
    
    def record_action(self, action: str) -> None:
        """Record an action for phase detection."""
        self.action_history.append(action.lower())
        
        # Keep only recent history
        if len(self.action_history) > 50:
            self.action_history = self.action_history[-50:]
    
    def detect_phase(self, recent_text: str = "") -> TaskPhase:
        """
        Detect current phase based on context.
        
        Args:
            recent_text: Recent conversation or action text
            
        Returns:
            Detected phase
        """
        text_lower = recent_text.lower()
        
        # Check for phase keywords
        scores = defaultdict(int)
        for phase, keywords in self.phase_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    scores[phase] += 1
        
        # Check action patterns
        recent_tools = self.action_history[-10:] if self.action_history else []
        
        if any("write" in t or "apply" in t for t in recent_tools):
            scores[TaskPhase.IMPLEMENTATION] += 2
        elif any("test" in t or "run" in t for t in recent_tools):
            scores[TaskPhase.VERIFICATION] += 2
        elif any("read" in t or "search" in t for t in recent_tools):
            scores[TaskPhase.EXPLORATION] += 1
        
        # Pick highest scoring phase
        if scores:
            detected = max(scores, key=lambda k: scores[k])
            return detected
        
        return self.current_phase
    
    def should_transition(self) -> bool:
        """Check if we should transition to next phase."""
        config = PHASE_CONFIGS[self.current_phase]
        
        # Check timeout
        elapsed = time.time() - self.phase_start_time
        if elapsed > config.max_duration_seconds:
            return True
        
        # Check action patterns
        if self.current_phase == TaskPhase.EXPLORATION:
            # Transition if we've done enough reading
            read_count = sum(1 for a in self.action_history[-10:] if "read" in a)
            if read_count >= 3:
                return True
        
        elif self.current_phase == TaskPhase.IMPLEMENTATION:
            # Transition if we've made changes
            write_count = sum(1 for a in self.action_history[-5:] if "write" in a or "apply" in a)
            if write_count >= 1:
                return True
        
        return False
    
    def transition_to(self, phase: TaskPhase) -> None:
        """Transition to a new phase."""
        self.phase_history.append((self.current_phase, time.time() - self.phase_start_time))
        self.current_phase = phase
        self.phase_start_time = time.time()
        logger.info(f"Transitioned to phase: {phase.name}")
    
    def get_next_phase(self) -> TaskPhase:
        """Get the recommended next phase."""
        order = [
            TaskPhase.EXPLORATION,
            TaskPhase.ANALYSIS,
            TaskPhase.PLANNING,
            TaskPhase.IMPLEMENTATION,
            TaskPhase.VERIFICATION,
            TaskPhase.COMPLETE,
        ]
        
        try:
            idx = order.index(self.current_phase)
            if idx < len(order) - 1:
                return order[idx + 1]
        except ValueError:
            pass
        
        return TaskPhase.COMPLETE


# =============================================================================
# ADAPTIVE TOOL SELECTOR
# =============================================================================

class AdaptiveToolSelector:
    """
    Dynamically selects which tools to expose based on task context.
    
    This is the main class for adaptive tool selection.
    
    Features:
    1. Phase-based filtering: Only show relevant tools for current phase
    2. Success-based ranking: Prefer tools that work well
    3. Risk awareness: Warn about high-risk tools
    4. Learning: Improve selection based on history
    """
    
    def __init__(self, enable_learning: bool = True):
        """
        Initialize the adaptive tool selector.
        
        Args:
            enable_learning: Whether to learn from usage patterns
        """
        self.registry = ToolRegistry()
        self.detector = PhaseDetector()
        self.enable_learning = enable_learning
        
        # Learning: track what tools work for what situations
        self.successful_patterns: List[Dict[str, Any]] = []
        
        self.stats = {
            "selections": 0,
            "tools_filtered": 0,
            "phase_transitions": 0,
        }
    
    def get_available_tools(
        self,
        context: str = "",
        force_phase: Optional[TaskPhase] = None,
    ) -> List[ToolMeta]:
        """
        Get list of available tools for current context.
        
        Args:
            context: Current context text (conversation, etc.)
            force_phase: Override detected phase
            
        Returns:
            List of available tools, filtered and ranked
        """
        self.stats["selections"] += 1
        
        # Detect or use forced phase
        if force_phase:
            phase = force_phase
        else:
            phase = self.detector.detect_phase(context)
            
            # Check for transitions
            if self.detector.should_transition():
                next_phase = self.detector.get_next_phase()
                self.detector.transition_to(next_phase)
                phase = next_phase
                self.stats["phase_transitions"] += 1
        
        # Get phase config
        config = PHASE_CONFIGS[phase]
        
        # Filter tools
        available = []
        for tool in self.registry.tools.values():
            if config.is_tool_allowed(tool.name):
                available.append(tool)
            else:
                self.stats["tools_filtered"] += 1
        
        # Rank by success rate
        available.sort(key=lambda t: t.success_rate, reverse=True)
        
        return available
    
    def get_tool_prompt(
        self,
        context: str = "",
        force_phase: Optional[TaskPhase] = None,
    ) -> str:
        """
        Get a prompt-friendly list of available tools.
        
        Returns formatted tool list for LLM consumption.
        """
        tools = self.get_available_tools(context, force_phase)
        
        phase = force_phase or self.detector.current_phase
        config = PHASE_CONFIGS[phase]
        
        lines = [
            f"## Available Tools ({config.name} Phase)",
            f"Current phase: {config.description}",
            "",
            "You can use these tools:",
        ]
        
        for tool in tools:
            risk_indicator = "⚠️ " if tool.risk_level >= 5 else ""
            lines.append(f"  - {risk_indicator}{tool.name}: {tool.description}")
        
        if config.transition_hints:
            lines.append("")
            lines.append("Transition to next phase when:")
            for hint in config.transition_hints:
                lines.append(f"  - {hint}")
        
        return "\n".join(lines)
    
    def record_tool_use(
        self,
        tool_name: str,
        success: bool,
        duration_ms: float,
        context: str = "",
    ) -> None:
        """
        Record a tool usage for learning.
        
        Args:
            tool_name: Name of tool used
            success: Whether it was successful
            duration_ms: How long it took
            context: Context when tool was used
        """
        self.detector.record_action(tool_name)
        self.registry.record_usage(tool_name, success, duration_ms)
        
        if self.enable_learning and success:
            self.successful_patterns.append({
                "tool": tool_name,
                "phase": self.detector.current_phase.name,
                "context_snippet": context[:100] if context else "",
                "timestamp": time.time(),
            })
            
            # Keep only recent patterns
            if len(self.successful_patterns) > 100:
                self.successful_patterns = self.successful_patterns[-100:]
    
    def set_phase(self, phase: TaskPhase) -> None:
        """Manually set the current phase."""
        self.detector.transition_to(phase)
    
    def get_phase(self) -> TaskPhase:
        """Get current phase."""
        return self.detector.current_phase
    
    def get_recommended_tools(
        self,
        task_type: str = "",
        top_k: int = 5,
    ) -> List[str]:
        """
        Get recommended tools based on learned patterns.
        
        Args:
            task_type: Type of task (if known)
            top_k: Number of tools to recommend
            
        Returns:
            List of recommended tool names
        """
        # Count successful tool uses
        tool_counts = defaultdict(int)
        for pattern in self.successful_patterns:
            tool_counts[pattern["tool"]] += 1
        
        # Sort by frequency
        ranked = sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [tool for tool, count in ranked[:top_k]]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get selector statistics."""
        tool_stats = {
            name: {
                "usage_count": tool.usage_count,
                "success_rate": f"{tool.success_rate:.1%}",
                "avg_duration_ms": f"{tool.avg_duration_ms:.0f}",
            }
            for name, tool in self.registry.tools.items()
            if tool.usage_count > 0
        }
        
        return {
            **self.stats,
            "current_phase": self.detector.current_phase.name,
            "patterns_learned": len(self.successful_patterns),
            "tool_stats": tool_stats,
        }
    
    def print_stats(self) -> None:
        """Print selector statistics."""
        stats = self.get_stats()
        print("\n" + "=" * 60)
        print("ADAPTIVE TOOL SELECTOR STATISTICS")
        print("=" * 60)
        print(f"Total Selections:    {stats['selections']}")
        print(f"Tools Filtered:      {stats['tools_filtered']}")
        print(f"Phase Transitions:   {stats['phase_transitions']}")
        print(f"Current Phase:       {stats['current_phase']}")
        print(f"Patterns Learned:    {stats['patterns_learned']}")
        
        if stats['tool_stats']:
            print("\nTool Usage Stats:")
            for name, data in stats['tool_stats'].items():
                print(f"  {name}: {data['usage_count']} uses, "
                      f"{data['success_rate']} success, "
                      f"{data['avg_duration_ms']}ms avg")
        print("=" * 60)


# =============================================================================
# TOOL ORCHESTRATOR
# =============================================================================

class ToolOrchestrator:
    """
    High-level orchestrator for tool selection and execution.
    
    Combines:
    - Adaptive tool selection
    - Phase management
    - Execution tracking
    
    Usage:
        orchestrator = ToolOrchestrator()
        
        # Get tools for current phase
        tools = orchestrator.get_tools("I need to find the bug...")
        
        # Execute a tool
        result = orchestrator.execute("read_file", path="main.py")
    """
    
    def __init__(
        self,
        tool_implementations: Optional[Dict[str, Callable]] = None,
    ):
        """
        Initialize the orchestrator.
        
        Args:
            tool_implementations: Dict mapping tool names to functions
        """
        self.selector = AdaptiveToolSelector()
        self.implementations = tool_implementations or {}
    
    def register_tool(
        self,
        name: str,
        implementation: Callable,
        description: str = "",
        category: str = "other",
        risk_level: int = 0,
    ) -> None:
        """Register a tool with its implementation."""
        self.implementations[name] = implementation
        self.selector.registry.register(ToolMeta(
            name=name,
            description=description,
            category=category,
            risk_level=risk_level,
        ))
    
    def get_tools(self, context: str = "") -> List[str]:
        """Get available tool names for current context."""
        tools = self.selector.get_available_tools(context)
        return [t.name for t in tools]
    
    def get_tool_prompt(self, context: str = "") -> str:
        """Get tool prompt for LLM."""
        return self.selector.get_tool_prompt(context)
    
    def execute(
        self,
        tool_name: str,
        **kwargs,
    ) -> Tuple[bool, Any]:
        """
        Execute a tool.
        
        Args:
            tool_name: Name of tool to execute
            **kwargs: Tool arguments
            
        Returns:
            Tuple of (success, result)
        """
        if tool_name not in self.implementations:
            return False, f"Tool {tool_name} not implemented"
        
        start_time = time.time()
        
        try:
            result = self.implementations[tool_name](**kwargs)
            success = True
        except Exception as e:
            result = str(e)
            success = False
        
        duration_ms = (time.time() - start_time) * 1000
        
        # Record for learning
        self.selector.record_tool_use(
            tool_name=tool_name,
            success=success,
            duration_ms=duration_ms,
        )
        
        return success, result
    
    def set_phase(self, phase: TaskPhase) -> None:
        """Set the current phase."""
        self.selector.set_phase(phase)
    
    def get_phase(self) -> TaskPhase:
        """Get current phase."""
        return self.selector.get_phase()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_phase_tools(phase: TaskPhase) -> List[str]:
    """Get list of tools allowed in a phase."""
    config = PHASE_CONFIGS[phase]
    return list(config.allowed_tools)


def create_tool_prompt_for_phase(phase: TaskPhase) -> str:
    """Create a tool prompt for a specific phase."""
    selector = AdaptiveToolSelector()
    return selector.get_tool_prompt(force_phase=phase)


# =============================================================================
# MAIN (DEMO)
# =============================================================================

if __name__ == "__main__":
    print("S.P.I.D.E.R. Dynamic Tool Selection Demo")
    print("=" * 50)
    
    selector = AdaptiveToolSelector()
    
    # Show tools for each phase
    for phase in [TaskPhase.EXPLORATION, TaskPhase.IMPLEMENTATION, TaskPhase.VERIFICATION]:
        print(f"\n{phase.name} Phase:")
        print("-" * 40)
        print(selector.get_tool_prompt(force_phase=phase))
    
    # Demo phase transitions
    print("\n" + "=" * 50)
    print("Phase Transition Demo")
    print("=" * 50)
    
    # Simulate actions
    selector.detector.record_action("read_file")
    selector.detector.record_action("read_file")
    selector.detector.record_action("read_file")
    
    print(f"After 3 read_file actions: should_transition = {selector.detector.should_transition()}")
    
    if selector.detector.should_transition():
        next_phase = selector.detector.get_next_phase()
        selector.detector.transition_to(next_phase)
        print(f"Transitioned to: {next_phase.name}")
    
    selector.print_stats()
