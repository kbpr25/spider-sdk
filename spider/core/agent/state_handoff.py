"""
S.P.I.D.E.R. State Vector Handoff - Fork-Based Context Continuity
==================================================================

Born from: Counter to Anthropic-1.5 (new_context_tool)

The Anthropic Weakness:
"They introduced new_context_tool - 'Sometimes context is so polluted,
we need to declare bankruptcy and start over.' The agent wipes memory
and starts fresh. This breaks the Chain of Causality."

The S.P.I.D.E.R. Evolution:
Never "Start Over." Instead, we "Fork."

Mechanism:
1. State Serialization: Serialize PhantomOS state + Knowledge Graph
2. Agent Fork: Spawn fresh LLM context
3. State Injection: Inject State Vector into new agent's System Prompt
4. Continuity: New agent has empty tokens but FULL structural understanding

Result: Claude resets and re-reads files.
S.P.I.D.E.R. inherits wisdom without token baggage.
"""

import hashlib
import json
import logging
import time
import zlib
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# STATE TYPES
# =============================================================================

class StateType(Enum):
    """Types of state data."""
    FILE_SYSTEM = auto()      # Current file state
    ENVIRONMENT = auto()      # Environment variables
    KNOWLEDGE = auto()        # Learned constraints/facts
    HISTORY = auto()          # Key decisions made
    FAILURES = auto()         # Failed approaches
    FOCUS = auto()            # Current focus/goal


@dataclass
class StateComponent:
    """A component of the state vector."""
    component_type: StateType
    data: Dict[str, Any]
    priority: int = 1         # Higher = more important
    token_cost: int = 0       # Estimated tokens to represent


@dataclass
class StateVector:
    """
    Complete state vector for handoff.
    
    Contains everything a new agent needs to continue work
    without re-reading context.
    """
    vector_id: str
    created_at: float
    
    # Core state
    components: List[StateComponent]
    
    # Summary for injection
    summary: str
    system_prompt_injection: str
    
    # Metadata
    source_agent_id: str = ""
    total_tokens_original: int = 0
    total_tokens_compressed: int = 0
    compression_ratio: float = 0.0


@dataclass
class HandoffConfig:
    """Configuration for state handoff."""
    max_injection_tokens: int = 2000
    include_file_contents: bool = False
    include_failure_history: bool = True
    priority_threshold: int = 2


# =============================================================================
# STATE SERIALIZER
# =============================================================================

class StateSerializer:
    """
    Serializes agent state for handoff.
    
    Captures:
    - File system state (what files exist, key contents)
    - Environment state (variables, configs)
    - Knowledge state (learned facts, constraints)
    - Decision history (what was tried, what failed)
    """
    
    def __init__(self):
        self.state_components: List[StateComponent] = []
    
    def capture_filesystem(
        self,
        files: Dict[str, str],
        modified_files: Set[str] = None,
    ) -> StateComponent:
        """Capture file system state."""
        modified = modified_files or set()
        
        file_summary = {
            "files": list(files.keys())[:50],  # Cap at 50 files
            "modified": list(modified)[:20],
            "key_files": {},
        }
        
        # Include snippets of key files
        for path, content in list(files.items())[:10]:
            if len(content) < 500:
                file_summary["key_files"][path] = content
            else:
                file_summary["key_files"][path] = content[:200] + "... [truncated]"
        
        component = StateComponent(
            component_type=StateType.FILE_SYSTEM,
            data=file_summary,
            priority=3,
            token_cost=len(str(file_summary)) // 4,
        )
        
        self.state_components.append(component)
        return component
    
    def capture_environment(
        self,
        env_vars: Dict[str, str],
        config: Dict[str, Any] = None,
    ) -> StateComponent:
        """Capture environment state."""
        env_summary = {
            "variables": {k: v for k, v in list(env_vars.items())[:20]},
            "config": config or {},
        }
        
        component = StateComponent(
            component_type=StateType.ENVIRONMENT,
            data=env_summary,
            priority=2,
            token_cost=len(str(env_summary)) // 4,
        )
        
        self.state_components.append(component)
        return component
    
    def capture_knowledge(
        self,
        facts: List[str],
        constraints: List[str] = None,
        dependencies: Dict[str, List[str]] = None,
    ) -> StateComponent:
        """Capture learned knowledge."""
        knowledge = {
            "facts": facts[:30],
            "constraints": (constraints or [])[:20],
            "dependencies": dependencies or {},
        }
        
        component = StateComponent(
            component_type=StateType.KNOWLEDGE,
            data=knowledge,
            priority=4,  # Highest priority
            token_cost=len(str(knowledge)) // 4,
        )
        
        self.state_components.append(component)
        return component
    
    def capture_history(
        self,
        decisions: List[str],
        current_goal: str = "",
        progress: float = 0.0,
    ) -> StateComponent:
        """Capture decision history."""
        history = {
            "decisions": decisions[-20:],  # Last 20 decisions
            "current_goal": current_goal,
            "progress": progress,
        }
        
        component = StateComponent(
            component_type=StateType.HISTORY,
            data=history,
            priority=3,
            token_cost=len(str(history)) // 4,
        )
        
        self.state_components.append(component)
        return component
    
    def capture_failures(
        self,
        failed_approaches: List[Dict[str, str]],
    ) -> StateComponent:
        """Capture what didn't work."""
        failures = {
            "approaches": failed_approaches[-10:],  # Last 10 failures
        }
        
        component = StateComponent(
            component_type=StateType.FAILURES,
            data=failures,
            priority=4,  # Very important - don't repeat!
            token_cost=len(str(failures)) // 4,
        )
        
        self.state_components.append(component)
        return component
    
    def get_components(self) -> List[StateComponent]:
        return self.state_components
    
    def clear(self) -> None:
        self.state_components.clear()


# =============================================================================
# PROMPT GENERATOR
# =============================================================================

class StatePromptGenerator:
    """
    Generates system prompt injection from state vector.
    
    Creates a compact, structured injection that gives the new agent
    full context without the token history.
    """
    
    INJECTION_TEMPLATE = '''
## INHERITED STATE (from previous agent context)

You are continuing work from a previous agent. Here is the critical state:

### Current Goal
{current_goal}

### Progress
{progress}

### Key Files Modified
{files_modified}

### Learned Facts (DO NOT re-discover)
{facts}

### Constraints (MUST respect)
{constraints}

### Failed Approaches (DO NOT repeat)
{failures}

### Key Decisions Made
{decisions}

---
Continue from where the previous agent left off. You have inherited all knowledge.
'''
    
    def generate(
        self,
        state: StateVector,
        config: HandoffConfig = None,
    ) -> str:
        """Generate system prompt injection."""
        config = config or HandoffConfig()
        
        # Extract data from components
        current_goal = ""
        progress = "0%"
        files_modified = []
        facts = []
        constraints = []
        failures = []
        decisions = []
        
        for component in sorted(state.components, key=lambda c: -c.priority):
            if component.component_type == StateType.HISTORY:
                current_goal = component.data.get("current_goal", "")
                prog = component.data.get("progress", 0)
                progress = f"{prog:.0%}"
                decisions = component.data.get("decisions", [])
            
            elif component.component_type == StateType.FILE_SYSTEM:
                files_modified = component.data.get("modified", [])
            
            elif component.component_type == StateType.KNOWLEDGE:
                facts = component.data.get("facts", [])
                constraints = component.data.get("constraints", [])
            
            elif component.component_type == StateType.FAILURES:
                for approach in component.data.get("approaches", []):
                    failures.append(f"- {approach.get('what', '')}: {approach.get('why', '')}")
        
        injection = self.INJECTION_TEMPLATE.format(
            current_goal=current_goal or "Continue ongoing work",
            progress=progress,
            files_modified="\n".join(f"- {f}" for f in files_modified[:10]) or "None",
            facts="\n".join(f"- {f}" for f in facts[:15]) or "None",
            constraints="\n".join(f"- {c}" for c in constraints[:10]) or "None",
            failures="\n".join(failures[:10]) or "None",
            decisions="\n".join(f"- {d}" for d in decisions[-10:]) or "None",
        )
        
        # Truncate if too long
        if len(injection) // 4 > config.max_injection_tokens:
            injection = injection[:config.max_injection_tokens * 4] + "\n[truncated]"
        
        return injection


# =============================================================================
# STATE VECTOR HANDOFF
# =============================================================================

class StateVectorHandoff:
    """
    The State-Vector Handoff System - Fork-Based Continuity.
    
    Never "wipes" context. Instead, performs a fork() operation:
    1. Serialize current agent's knowledge
    2. Spawn fresh context
    3. Inject state vector
    4. Continue with empty tokens but full wisdom
    
    Usage:
        handoff = StateVectorHandoff()
        
        # Capture current state
        handoff.capture_file_state(files, modified)
        handoff.capture_knowledge(facts, constraints)
        handoff.capture_failures(failed_attempts)
        
        # If context is polluted
        if context_too_long():
            state_vector = handoff.create_vector()
            
            # Fork to new agent
            new_agent = spawn_agent()
            new_agent.inject(state_vector.system_prompt_injection)
            
            # New agent continues with full knowledge
    """
    
    def __init__(
        self,
        agent_id: str = None,
        config: HandoffConfig = None,
    ):
        """
        Initialize State Vector Handoff.
        
        Args:
            agent_id: Current agent identifier
            config: Handoff configuration
        """
        self.agent_id = agent_id or hashlib.md5(
            str(time.time()).encode()
        ).hexdigest()[:8]
        
        self.config = config or HandoffConfig()
        
        self.serializer = StateSerializer()
        self.prompt_gen = StatePromptGenerator()
        
        # Current state
        self.current_vector: Optional[StateVector] = None
        self.handoff_history: List[str] = []  # Vector IDs
        
        self._stats = {
            "handoffs_created": 0,
            "total_tokens_saved": 0,
            "average_compression": 0.0,
        }
    
    def capture_file_state(
        self,
        files: Dict[str, str],
        modified: Set[str] = None,
    ) -> None:
        """Capture current file system state."""
        self.serializer.capture_filesystem(files, modified)
    
    def capture_environment(
        self,
        env_vars: Dict[str, str],
        config: Dict[str, Any] = None,
    ) -> None:
        """Capture environment state."""
        self.serializer.capture_environment(env_vars, config)
    
    def capture_knowledge(
        self,
        facts: List[str],
        constraints: List[str] = None,
        dependencies: Dict[str, List[str]] = None,
    ) -> None:
        """Capture learned knowledge."""
        self.serializer.capture_knowledge(facts, constraints, dependencies)
    
    def capture_history(
        self,
        decisions: List[str],
        current_goal: str = "",
        progress: float = 0.0,
    ) -> None:
        """Capture decision history."""
        self.serializer.capture_history(decisions, current_goal, progress)
    
    def capture_failures(
        self,
        failed_approaches: List[Dict[str, str]],
    ) -> None:
        """Capture what didn't work."""
        self.serializer.capture_failures(failed_approaches)
    
    def create_vector(
        self,
        original_tokens: int = 0,
    ) -> StateVector:
        """
        Create a state vector from captured state.
        
        Args:
            original_tokens: Token count of original context
            
        Returns:
            StateVector ready for injection
        """
        vector_id = hashlib.md5(
            f"{self.agent_id}{time.time()}".encode()
        ).hexdigest()[:12]
        
        components = list(self.serializer.get_components())  # Copy to avoid clear()
        
        # Generate injection
        temp_vector = StateVector(
            vector_id=vector_id,
            created_at=time.time(),
            components=components,
            summary="",
            system_prompt_injection="",
            source_agent_id=self.agent_id,
        )
        
        injection = self.prompt_gen.generate(temp_vector, self.config)
        
        # Create summary
        summary = f"State from agent {self.agent_id}: {len(components)} components"
        
        # Calculate compression
        compressed_tokens = len(injection) // 4
        compression_ratio = compressed_tokens / max(original_tokens, 1)
        
        vector = StateVector(
            vector_id=vector_id,
            created_at=time.time(),
            components=components,
            summary=summary,
            system_prompt_injection=injection,
            source_agent_id=self.agent_id,
            total_tokens_original=original_tokens,
            total_tokens_compressed=compressed_tokens,
            compression_ratio=compression_ratio,
        )
        
        self.current_vector = vector
        self.handoff_history.append(vector_id)
        
        # Update stats
        self._stats["handoffs_created"] += 1
        self._stats["total_tokens_saved"] += max(0, original_tokens - compressed_tokens)
        if self._stats["handoffs_created"] > 0:
            self._stats["average_compression"] = (
                self._stats["total_tokens_saved"] / self._stats["handoffs_created"]
            )
        
        # Clear serializer for next capture
        self.serializer.clear()
        
        return vector
    
    def get_injection(self) -> str:
        """Get the system prompt injection for handoff."""
        if self.current_vector:
            return self.current_vector.system_prompt_injection
        return ""
    
    def serialize_vector(self, vector: StateVector) -> bytes:
        """Serialize vector for storage/transmission."""
        data = {
            "vector_id": vector.vector_id,
            "created_at": vector.created_at,
            "summary": vector.summary,
            "injection": vector.system_prompt_injection,
            "source_agent": vector.source_agent_id,
            "components": [
                {
                    "type": c.component_type.name,
                    "data": c.data,
                    "priority": c.priority,
                }
                for c in vector.components
            ],
        }
        json_str = json.dumps(data)
        return zlib.compress(json_str.encode())
    
    def deserialize_vector(self, data: bytes) -> StateVector:
        """Deserialize a stored vector."""
        json_str = zlib.decompress(data).decode()
        data = json.loads(json_str)
        
        components = [
            StateComponent(
                component_type=StateType[c["type"]],
                data=c["data"],
                priority=c["priority"],
            )
            for c in data["components"]
        ]
        
        return StateVector(
            vector_id=data["vector_id"],
            created_at=data["created_at"],
            components=components,
            summary=data["summary"],
            system_prompt_injection=data["injection"],
            source_agent_id=data["source_agent"],
        )
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            **self._stats,
            "handoff_chain_length": len(self.handoff_history),
        }
    
    def print_status(self) -> None:
        """Print handoff status."""
        print("\n" + "=" * 60)
        print("[*] STATE VECTOR HANDOFF STATUS")
        print("=" * 60)
        
        if self.current_vector:
            v = self.current_vector
            print(f"\n[V] Current Vector: {v.vector_id}")
            print(f"   Components: {len(v.components)}")
            print(f"   Original tokens: {v.total_tokens_original}")
            print(f"   Compressed to: {v.total_tokens_compressed}")
            print(f"   Compression: {v.compression_ratio:.1%}")
        
        print(f"\n[H] Handoff Chain: {len(self.handoff_history)} generations")
        
        print(f"\n[%] Stats:")
        for key, val in self.get_stats().items():
            print(f"   {key}: {val}")
        
        print("=" * 60)


# =============================================================================
# PACKAGE INIT
# =============================================================================

__all__ = [
    "StateVectorHandoff",
    "StateSerializer",
    "StatePromptGenerator",
    "StateVector",
    "StateComponent",
    "StateType",
    "HandoffConfig",
]


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("[*] S.P.I.D.E.R. State Vector Handoff - Demo")
    print("=" * 70)
    
    handoff = StateVectorHandoff(agent_id="agent_001")
    
    # Capture state
    print("\n[1] Capturing agent state...")
    
    handoff.capture_file_state(
        files={
            "src/main.py": "def main(): print('hello')",
            "src/utils.py": "def helper(): pass",
            "tests/test_main.py": "def test_main(): assert True",
        },
        modified={"src/main.py", "src/utils.py"},
    )
    
    handoff.capture_knowledge(
        facts=[
            "The bug is in main.py line 42",
            "Uses Python 3.11",
            "Dependencies: requests, flask",
        ],
        constraints=[
            "Must maintain backward compatibility",
            "Cannot modify public API",
        ],
    )
    
    handoff.capture_history(
        decisions=[
            "Started with test-first approach",
            "Fixed import issue",
            "Refactored helper function",
        ],
        current_goal="Fix the null pointer bug in auth module",
        progress=0.6,
    )
    
    handoff.capture_failures([
        {"what": "Added null check at line 45", "why": "Didn't fix root cause"},
        {"what": "Tried refactoring validate()", "why": "Broke existing tests"},
    ])
    
    # Create vector
    print("\n[2] Creating state vector...")
    vector = handoff.create_vector(original_tokens=50000)
    
    print(f"   Vector ID: {vector.vector_id}")
    print(f"   Components: {len(vector.components)}")
    print(f"   Compression: 50000 -> {vector.total_tokens_compressed} tokens")
    
    # Show injection
    print("\n[3] System prompt injection (first 500 chars):")
    print("-" * 50)
    print(vector.system_prompt_injection[:500])
    print("-" * 50)
    print(f"[Total: {len(vector.system_prompt_injection)} chars]")
    
    handoff.print_status()
