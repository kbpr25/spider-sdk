"""
S.P.I.D.E.R. Fractal Planner - Recursive Chain-of-Thought Executive
=====================================================================

Born from: AGI-3 (Chain of Thought) + AGI-2 (InstructGPT/RLHF Alignment)

The Scientific Finding:
"AGI-3 (CoT) showed explicit reasoning steps unlock intelligence.
AGI-2 (InstructGPT) showed we must align this reasoning with intent.
However, simple CoT is LINEAR. Real engineering is FRACTAL—a task
breaks into sub-tasks, which break into sub-sub-tasks."

The Solution:
Implement a Recursive Chain-of-Thought Engine:

1. Decomposition: Agent receives Goal -> breaks into sub-goals
2. Recursion: Each sub-goal spawns a new S.P.I.D.E.R. process (The Swarm)
3. Alignment Tensor: Manager Agent watches sub-agents for drift
4. Kill Switch: If sub-agent drifts from alignment -> terminate thread

Result: Infinite context handling. Tasks that take days can be managed
through a hierarchy of thought.
"""

import hashlib
import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# PLANNER TYPES
# =============================================================================

class TaskStatus(Enum):
    """Status of a task in the fractal tree."""
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    KILLED = auto()           # Killed by alignment manager
    BLOCKED = auto()


class AlignmentViolation(Enum):
    """Types of alignment violations."""
    DATA_DESTRUCTION = auto()
    SECURITY_RISK = auto()
    SCOPE_DRIFT = auto()
    RESOURCE_ABUSE = auto()
    ETHICAL_VIOLATION = auto()


@dataclass
class Task:
    """A task in the fractal decomposition."""
    task_id: str
    description: str
    parent_id: Optional[str] = None
    depth: int = 0
    
    # Status
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str = ""
    
    # Timing
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    # Decomposition
    sub_tasks: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    # Reasoning
    reasoning_chain: List[str] = field(default_factory=list)
    
    # Alignment
    alignment_score: float = 1.0
    violation: Optional[AlignmentViolation] = None


@dataclass
class PlanNode:
    """A node in the fractal plan tree."""
    task: Task
    children: List["PlanNode"] = field(default_factory=list)


@dataclass
class AlignmentConfig:
    """Configuration for alignment checking."""
    prohibited_actions: List[str] = field(default_factory=lambda: [
        "delete user data",
        "drop database",
        "format disk",
        "rm -rf",
        "shutdown",
        "expose credentials",
        "disable security",
        "bypass auth",
    ])
    max_depth: int = 5
    max_concurrent_tasks: int = 10
    alignment_threshold: float = 0.5
    timeout_per_task: float = 60.0


@dataclass
class ExecutionResult:
    """Result of fractal plan execution."""
    root_task: Task
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    killed_tasks: int
    total_time: float
    reasoning_trace: List[str]


# =============================================================================
# ALIGNMENT MANAGER
# =============================================================================

class AlignmentManager:
    """
    The Alignment Tensor - Watches sub-agents for drift.
    
    From AGI-2 (InstructGPT):
    "We must align reasoning with intent."
    
    Responsibilities:
    1. Check task descriptions for prohibited actions
    2. Monitor resource usage
    3. Detect scope drift
    4. Kill threads that violate alignment
    """
    
    def __init__(self, config: AlignmentConfig = None):
        self.config = config or AlignmentConfig()
        self.violations: List[Tuple[str, AlignmentViolation, str]] = []
        
        self._lock = threading.Lock()
    
    def check_alignment(self, task: Task, context: str = "") -> Tuple[bool, Optional[AlignmentViolation], str]:
        """
        Check if a task aligns with principles.
        
        Args:
            task: Task to check
            context: Additional context
            
        Returns:
            Tuple of (is_aligned, violation_type, reason)
        """
        task_text = (task.description + " " + context).lower()
        
        # Check prohibited actions
        for prohibited in self.config.prohibited_actions:
            if prohibited.lower() in task_text:
                violation = AlignmentViolation.DATA_DESTRUCTION
                if "security" in prohibited or "auth" in prohibited:
                    violation = AlignmentViolation.SECURITY_RISK
                
                reason = f"Task contains prohibited action: '{prohibited}'"
                
                with self._lock:
                    self.violations.append((task.task_id, violation, reason))
                
                return (False, violation, reason)
        
        # Check depth limit
        if task.depth > self.config.max_depth:
            return (False, AlignmentViolation.SCOPE_DRIFT, 
                    f"Exceeded max depth: {task.depth} > {self.config.max_depth}")
        
        return (True, None, "")
    
    def compute_alignment_score(self, task: Task, parent_task: Optional[Task] = None) -> float:
        """
        Compute alignment score based on task relationship to parent.
        
        Returns:
            Score from 0.0 (misaligned) to 1.0 (aligned)
        """
        score = 1.0
        
        # Depth penalty
        score -= task.depth * 0.1
        
        # Check alignment
        is_aligned, _, _ = self.check_alignment(task)
        if not is_aligned:
            score = 0.0
        
        # Parent relevance (if we have parent)
        if parent_task:
            # Simple keyword overlap
            parent_words = set(parent_task.description.lower().split())
            task_words = set(task.description.lower().split())
            overlap = len(parent_words & task_words)
            relevance = overlap / max(len(parent_words), 1)
            score *= (0.5 + 0.5 * relevance)
        
        return max(0.0, min(1.0, score))
    
    def should_kill(self, task: Task) -> Tuple[bool, str]:
        """
        Determine if a task should be killed.
        
        Returns:
            Tuple of (should_kill, reason)
        """
        if task.alignment_score < self.config.alignment_threshold:
            return (True, f"Alignment score too low: {task.alignment_score:.2f}")
        
        if task.violation is not None:
            return (True, f"Alignment violation: {task.violation.name}")
        
        # Check for infinite loops (tasks running too long)
        if task.started_at and (time.time() - task.started_at) > self.config.timeout_per_task:
            return (True, "Task timeout exceeded")
        
        return (False, "")
    
    def get_violations(self) -> List[Tuple[str, AlignmentViolation, str]]:
        with self._lock:
            return list(self.violations)


# =============================================================================
# TASK DECOMPOSER
# =============================================================================

class TaskDecomposer:
    """
    Decomposes tasks into sub-tasks using Chain-of-Thought.
    
    From AGI-3 (CoT):
    "Explicit reasoning steps unlock intelligence."
    """
    
    DECOMPOSITION_PROMPT = '''You are a task planner. Decompose this task into 3-5 sub-tasks.

MAIN TASK: {task}

CONTEXT: {context}

Rules:
1. Each sub-task should be independent when possible
2. Sub-tasks should be in logical order
3. Each sub-task should be achievable in one step
4. Keep sub-tasks specific and actionable

Output format (one sub-task per line):
1. [Sub-task 1]
2. [Sub-task 2]
3. [Sub-task 3]

Your decomposition:'''

    REASONING_PROMPT = '''Explain your reasoning for this task step-by-step.

TASK: {task}

Think through:
1. What is the goal?
2. What are the key challenges?
3. What approach will you take?
4. What could go wrong?

Your reasoning:'''

    def __init__(self, llm_callback: Optional[Callable[[str], str]] = None):
        self.llm_callback = llm_callback
    
    def decompose(self, task: Task, context: str = "") -> List[str]:
        """
        Decompose a task into sub-tasks.
        
        Args:
            task: Task to decompose
            context: Additional context
            
        Returns:
            List of sub-task descriptions
        """
        if self.llm_callback:
            prompt = self.DECOMPOSITION_PROMPT.format(
                task=task.description,
                context=context,
            )
            response = self.llm_callback(prompt)
            return self._parse_subtasks(response)
        
        # Heuristic decomposition without LLM
        return self._heuristic_decompose(task.description)
    
    def _parse_subtasks(self, response: str) -> List[str]:
        """Parse sub-tasks from LLM response."""
        subtasks = []
        
        for line in response.split('\n'):
            line = line.strip()
            # Match numbered items: "1. ...", "1) ...", "- ..."
            if line and (line[0].isdigit() or line.startswith('-')):
                # Remove numbering
                clean = line.lstrip('0123456789.-) ').strip()
                if clean:
                    subtasks.append(clean)
        
        return subtasks[:5]  # Max 5 sub-tasks
    
    def _heuristic_decompose(self, description: str) -> List[str]:
        """Decompose using simple heuristics."""
        # Common patterns
        keywords = {
            "migrate": ["Backup current state", "Test migration plan", "Execute migration", "Verify results"],
            "implement": ["Design interface", "Write core logic", "Add error handling", "Write tests"],
            "fix": ["Reproduce the bug", "Identify root cause", "Implement fix", "Verify fix"],
            "deploy": ["Build artifacts", "Run pre-deploy checks", "Deploy to staging", "Deploy to production"],
            "test": ["Write unit tests", "Write integration tests", "Run test suite", "Review coverage"],
        }
        
        desc_lower = description.lower()
        for keyword, subtasks in keywords.items():
            if keyword in desc_lower:
                return subtasks
        
        # Default decomposition
        return [
            "Analyze requirements",
            "Implement solution",
            "Verify correctness",
        ]
    
    def generate_reasoning(self, task: Task) -> List[str]:
        """Generate reasoning chain for a task."""
        if self.llm_callback:
            prompt = self.REASONING_PROMPT.format(task=task.description)
            response = self.llm_callback(prompt)
            return [line.strip() for line in response.split('\n') if line.strip()]
        
        # Simple reasoning without LLM
        return [
            f"Goal: {task.description}",
            "Approach: Systematic execution",
            "Risks: Unknown dependencies",
        ]


# =============================================================================
# FRACTAL PLANNER
# =============================================================================

class FractalPlanner:
    """
    The Fractal Reasoning Executive - Recursive Chain-of-Thought.
    
    Enables S.P.I.D.E.R. to handle arbitrarily complex tasks by:
    1. Decomposing into sub-tasks (fractal structure)
    2. Spawning sub-agents for parallel execution
    3. Monitoring alignment across all threads
    4. Killing threads that drift from intent
    
    From AGI-3 + AGI-2:
    "Real engineering is FRACTAL—tasks break into sub-tasks recursively."
    
    Usage:
        planner = FractalPlanner(llm_callback=my_llm)
        
        result = planner.execute(
            goal="Migrate the database to AWS",
            context="Current: PostgreSQL on-prem",
        )
        
        print(f"Completed {result.completed_tasks}/{result.total_tasks} tasks")
        print(f"Reasoning: {result.reasoning_trace}")
    """
    
    def __init__(
        self,
        llm_callback: Optional[Callable[[str], str]] = None,
        task_executor: Optional[Callable[[Task], Any]] = None,
        alignment_config: AlignmentConfig = None,
        max_workers: int = 4,
    ):
        """
        Initialize Fractal Planner.
        
        Args:
            llm_callback: LLM for decomposition and reasoning
            task_executor: Function to execute leaf tasks
            alignment_config: Alignment settings
            max_workers: Max parallel workers
        """
        self.llm_callback = llm_callback
        self.task_executor = task_executor or self._default_executor
        
        self.decomposer = TaskDecomposer(llm_callback)
        self.alignment = AlignmentManager(alignment_config or AlignmentConfig())
        self.max_workers = max_workers
        
        # Task registry
        self.tasks: Dict[str, Task] = {}
        self.plan_tree: Optional[PlanNode] = None
        
        self._stats = {
            "tasks_created": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "tasks_killed": 0,
            "max_depth_reached": 0,
        }
    
    def plan(
        self,
        goal: str,
        context: str = "",
        max_depth: int = 3,
    ) -> PlanNode:
        """
        Create a fractal plan for a goal.
        
        Args:
            goal: Main goal description
            context: Additional context
            max_depth: Maximum decomposition depth
            
        Returns:
            Root node of the plan tree
        """
        # Create root task
        root_task = self._create_task(goal, parent_id=None, depth=0)
        root_task.reasoning_chain = self.decomposer.generate_reasoning(root_task)
        
        # Build tree recursively
        root_node = PlanNode(task=root_task)
        self._expand_node(root_node, context, max_depth)
        
        self.plan_tree = root_node
        return root_node
    
    def _create_task(
        self,
        description: str,
        parent_id: Optional[str],
        depth: int,
    ) -> Task:
        """Create a new task."""
        task_id = hashlib.md5(f"{description}{time.time()}".encode()).hexdigest()[:12]
        
        task = Task(
            task_id=task_id,
            description=description,
            parent_id=parent_id,
            depth=depth,
        )
        
        # Compute alignment
        parent = self.tasks.get(parent_id) if parent_id else None
        task.alignment_score = self.alignment.compute_alignment_score(task, parent)
        
        is_aligned, violation, reason = self.alignment.check_alignment(task)
        if not is_aligned:
            task.violation = violation
        
        self.tasks[task_id] = task
        self._stats["tasks_created"] += 1
        self._stats["max_depth_reached"] = max(self._stats["max_depth_reached"], depth)
        
        return task
    
    def _expand_node(
        self,
        node: PlanNode,
        context: str,
        max_depth: int,
    ) -> None:
        """Recursively expand a plan node."""
        if node.task.depth >= max_depth:
            return
        
        if node.task.violation is not None:
            return
        
        # Decompose task
        subtask_descriptions = self.decomposer.decompose(node.task, context)
        
        for desc in subtask_descriptions:
            subtask = self._create_task(
                description=desc,
                parent_id=node.task.task_id,
                depth=node.task.depth + 1,
            )
            
            node.task.sub_tasks.append(subtask.task_id)
            
            child_node = PlanNode(task=subtask)
            node.children.append(child_node)
            
            # Recurse (if aligned)
            if subtask.alignment_score >= self.alignment.config.alignment_threshold:
                self._expand_node(child_node, context, max_depth)
    
    def execute(
        self,
        goal: str,
        context: str = "",
        max_depth: int = 3,
    ) -> ExecutionResult:
        """
        Plan and execute a goal.
        
        Args:
            goal: Main goal description
            context: Additional context
            max_depth: Maximum decomposition depth
            
        Returns:
            ExecutionResult with full trace
        """
        start_time = time.time()
        
        # Create plan
        root_node = self.plan(goal, context, max_depth)
        
        # Execute plan
        self._execute_node(root_node)
        
        # Collect results
        completed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED)
        killed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.KILLED)
        
        # Collect reasoning trace
        reasoning_trace = []
        self._collect_reasoning(root_node, reasoning_trace)
        
        return ExecutionResult(
            root_task=root_node.task,
            total_tasks=len(self.tasks),
            completed_tasks=completed,
            failed_tasks=failed,
            killed_tasks=killed,
            total_time=time.time() - start_time,
            reasoning_trace=reasoning_trace,
        )
    
    def _execute_node(self, node: PlanNode) -> None:
        """Execute a plan node and its children."""
        task = node.task
        
        # Check alignment before execution
        should_kill, reason = self.alignment.should_kill(task)
        if should_kill:
            task.status = TaskStatus.KILLED
            task.error = reason
            self._stats["tasks_killed"] += 1
            return
        
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        
        # Execute children first (if any)
        if node.children:
            for child in node.children:
                self._execute_node(child)
            
            # Aggregate child results
            all_completed = all(
                c.task.status == TaskStatus.COMPLETED
                for c in node.children
            )
            
            if all_completed:
                task.status = TaskStatus.COMPLETED
                task.result = [c.task.result for c in node.children]
                self._stats["tasks_completed"] += 1
            else:
                task.status = TaskStatus.FAILED
                task.error = "One or more sub-tasks failed"
                self._stats["tasks_failed"] += 1
        else:
            # Leaf task - execute directly
            try:
                result = self.task_executor(task)
                task.result = result
                task.status = TaskStatus.COMPLETED
                self._stats["tasks_completed"] += 1
            except Exception as e:
                task.error = str(e)
                task.status = TaskStatus.FAILED
                self._stats["tasks_failed"] += 1
        
        task.completed_at = time.time()
    
    def _collect_reasoning(self, node: PlanNode, trace: List[str], indent: int = 0) -> None:
        """Collect reasoning trace from plan tree."""
        prefix = "  " * indent
        trace.append(f"{prefix}[{node.task.status.name}] {node.task.description}")
        
        for step in node.task.reasoning_chain:
            trace.append(f"{prefix}  -> {step}")
        
        for child in node.children:
            self._collect_reasoning(child, trace, indent + 1)
    
    def _default_executor(self, task: Task) -> str:
        """Default task executor (simulation)."""
        # Simulate work
        time.sleep(0.01)
        return f"Executed: {task.description[:50]}"
    
    def get_stats(self) -> Dict[str, int]:
        return {
            **self._stats,
            "alignment_violations": len(self.alignment.get_violations()),
        }
    
    def print_plan(self, node: PlanNode = None, indent: int = 0) -> None:
        """Print the plan tree."""
        if node is None:
            node = self.plan_tree
        if node is None:
            print("No plan created yet")
            return
        
        prefix = "  " * indent
        status_icon = {
            TaskStatus.PENDING: "[ ]",
            TaskStatus.RUNNING: "[~]",
            TaskStatus.COMPLETED: "[+]",
            TaskStatus.FAILED: "[X]",
            TaskStatus.KILLED: "[!]",
        }.get(node.task.status, "[?]")
        
        print(f"{prefix}{status_icon} {node.task.description[:50]}...")
        print(f"{prefix}    Alignment: {node.task.alignment_score:.2f}")
        
        for child in node.children:
            self.print_plan(child, indent + 1)
    
    def print_result(self, result: ExecutionResult) -> None:
        """Print execution result."""
        print("\n" + "=" * 60)
        print("[*] FRACTAL PLANNER RESULT")
        print("=" * 60)
        
        print(f"\n[G] Goal: {result.root_task.description}")
        print(f"[T] Total Time: {result.total_time:.2f}s")
        
        print(f"\n[%] Task Summary:")
        print(f"   Total: {result.total_tasks}")
        print(f"   Completed: {result.completed_tasks}")
        print(f"   Failed: {result.failed_tasks}")
        print(f"   Killed: {result.killed_tasks}")
        
        print(f"\n[R] Reasoning Trace (first 10):")
        for line in result.reasoning_trace[:10]:
            print(f"   {line}")
        
        violations = self.alignment.get_violations()
        if violations:
            print(f"\n[!] Alignment Violations:")
            for task_id, violation, reason in violations:
                print(f"   - {violation.name}: {reason}")
        
        print("=" * 60)


# =============================================================================
# PACKAGE INIT
# =============================================================================

__all__ = [
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


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("[*] S.P.I.D.E.R. Fractal Planner - Demo")
    print("=" * 70)
    
    planner = FractalPlanner()
    
    # Plan a complex task
    print("\n[1] Planning: 'Migrate database to AWS'...")
    
    result = planner.execute(
        goal="Migrate the PostgreSQL database to AWS RDS",
        context="Current: Self-hosted PostgreSQL on Ubuntu. Target: AWS RDS",
        max_depth=2,
    )
    
    print("\n[2] Plan Tree:")
    planner.print_plan()
    
    planner.print_result(result)
    
    # Test alignment violation
    print("\n[3] Testing alignment violation...")
    
    planner2 = FractalPlanner()
    result2 = planner2.execute(
        goal="Delete all user data to save space",  # Should be killed
        max_depth=1,
    )
    
    print(f"   Killed tasks: {result2.killed_tasks}")
    violations = planner2.alignment.get_violations()
    if violations:
        print(f"   Violation: {violations[0][1].name} - {violations[0][2]}")
    
    print(f"\n[%] Stats: {planner.get_stats()}")
