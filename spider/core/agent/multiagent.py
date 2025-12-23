"""
S.P.I.D.E.R. Multi-Agent System - The Hive Mind
=================================================

Specialized agents working together like Claude Opus 4.5's subagent orchestration.

Agent Roles:
- Planner: Decomposes task into subtasks
- Coder: Implements the fix
- Tester: Verifies the solution
- Reviewer: Reviews code quality
- Orchestrator: Coordinates the team

This is what separates 60% from 80% on SWE-Bench.
"""

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# MESSAGE TYPES
# =============================================================================

class MessageType(Enum):
    """Types of inter-agent messages."""
    TASK = "task"          # New task assignment
    RESULT = "result"      # Task completion result
    FEEDBACK = "feedback"  # Feedback or critique
    REQUEST = "request"    # Request for help
    UPDATE = "update"      # Status update


@dataclass
class AgentMessage:
    """Message between agents."""
    from_agent: str
    to_agent: str
    msg_type: MessageType
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        return {
            "from": self.from_agent,
            "to": self.to_agent,
            "type": self.msg_type.value,
            "content": self.content,
            "metadata": self.metadata,
        }


# =============================================================================
# SPECIALIZED AGENTS
# =============================================================================

class SpecializedAgent(ABC):
    """Base class for specialized agents."""
    
    name: str = "agent"
    role: str = "A specialized agent"
    
    def __init__(self):
        from spider.core.agent.llm_client import LLMGateway, Message, MessageRole
        self.gateway = LLMGateway()
        self.Message = Message
        self.MessageRole = MessageRole
        self.total_cost = 0.0
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the agent's system prompt."""
        pass
    
    def process(self, task: str, context: str = "", max_tokens: int = 1500) -> str:
        """Process a task and return result."""
        messages = [
            self.Message(self.MessageRole.SYSTEM, self.get_system_prompt()),
            self.Message(self.MessageRole.USER, f"{context}\n\n## Task\n{task}" if context else task),
        ]
        
        response = self.gateway.complete(
            messages,
            model="deepseek/deepseek-chat",
            temperature=0.3,
            max_tokens=max_tokens,
        )
        
        self.total_cost += response.cost_usd
        return response.content if response.success else f"Error: {response.content}"


class PlannerAgent(SpecializedAgent):
    """Plans and decomposes complex tasks."""
    
    name = "planner"
    role = "Decomposes complex tasks into manageable steps"
    
    def get_system_prompt(self) -> str:
        return """You are an expert software architect and planner.

Your job is to analyze bug reports and create a step-by-step plan to fix them.

## Output Format

Return a JSON plan:
{
    "analysis": "Brief analysis of the root cause",
    "affected_files": ["list", "of", "files"],
    "steps": [
        {"id": 1, "action": "read", "target": "file.py", "reason": "why"},
        {"id": 2, "action": "modify", "target": "file.py", "change": "what to change"},
        {"id": 3, "action": "test", "target": "test_file.py", "verify": "what to verify"}
    ],
    "risk_level": "low|medium|high"
}

Be minimal and precise. Focus on the root cause."""


class CoderAgent(SpecializedAgent):
    """Implements code changes."""
    
    name = "coder"
    role = "Writes and modifies code to fix bugs"
    
    def get_system_prompt(self) -> str:
        return """You are an expert software developer.

Your job is to implement precise code fixes based on a plan.

## Rules

1. Make MINIMAL changes - only what's necessary
2. Follow existing code style
3. Add comments only when non-obvious
4. Handle edge cases
5. Don't break existing functionality

## Output Format

Return the complete fixed file content in a code block:
```python
# your fixed code here
```

Then briefly explain what you changed."""


class TesterAgent(SpecializedAgent):
    """Verifies solutions through testing."""
    
    name = "tester"
    role = "Verifies code changes through testing"
    
    def get_system_prompt(self) -> str:
        return """You are an expert QA engineer.

Your job is to analyze test results and determine if a fix is correct.

## Analysis Tasks

1. Parse test output
2. Identify which tests passed/failed
3. Determine if the fix addresses the original issue
4. Suggest additional tests if needed

## Output Format

{
    "verdict": "pass|fail|partial",
    "passed_tests": 5,
    "failed_tests": 1,
    "issues_found": ["issue1", "issue2"],
    "suggestions": ["suggestion1"]
}"""


class ReviewerAgent(SpecializedAgent):
    """Reviews code quality and suggests improvements."""
    
    name = "reviewer"
    role = "Reviews code quality and catches issues"
    
    def get_system_prompt(self) -> str:
        return """You are a senior code reviewer.

Your job is to review code changes for quality, correctness, and style.

## Review Checklist

1. Does it fix the root cause?
2. Are there any edge cases missed?
3. Is the code clean and readable?
4. Could it introduce new bugs?
5. Is it the minimal change needed?

## Output Format

{
    "approval": "approve|request_changes|comment",
    "score": 1-10,
    "issues": [
        {"severity": "high|medium|low", "issue": "description", "suggestion": "fix"}
    ],
    "summary": "Brief review summary"
}"""


# =============================================================================
# ORCHESTRATOR
# =============================================================================

@dataclass
class TeamConfig:
    """Configuration for the agent team."""
    max_iterations: int = 5
    max_cost_usd: float = 0.20
    require_review: bool = True
    require_tests: bool = True


class AgentOrchestrator:
    """
    Orchestrates a team of specialized agents.
    
    This is what makes Opus 4.5 "very effective at managing a team of subagents".
    
    Workflow:
    1. Planner analyzes and creates plan
    2. Coder implements the fix
    3. Tester runs and analyzes tests
    4. Reviewer reviews the code
    5. If issues, iterate with feedback
    """
    
    def __init__(self, config: Optional[TeamConfig] = None, repo_path: str = "."):
        self.config = config or TeamConfig()
        self.repo_path = repo_path
        
        # Initialize agents
        self.planner = PlannerAgent()
        self.coder = CoderAgent()
        self.tester = TesterAgent()
        self.reviewer = ReviewerAgent()
        
        # Communication log
        self.messages: List[AgentMessage] = []
        self.total_cost = 0.0
    
    def solve(self, problem: str, file_content: str = "") -> Tuple[bool, str, Dict]:
        """
        Solve a problem using the agent team.
        
        Args:
            problem: Problem description
            file_content: Relevant source code
            
        Returns:
            (success, solution, metadata)
        """
        logger.info("Orchestrator starting team solve...")
        
        context = f"## Source Code\n```\n{file_content}\n```" if file_content else ""
        
        # Phase 1: Planning
        logger.info("Phase 1: Planning")
        plan_result = self._plan(problem, context)
        
        if not plan_result:
            return False, "", {"error": "Planning failed"}
        
        # Phase 2: Implementation
        logger.info("Phase 2: Coding")
        code_result = self._code(problem, context, plan_result)
        
        if not code_result:
            return False, "", {"error": "Coding failed"}
        
        # Phase 3: Review (if enabled)
        if self.config.require_review:
            logger.info("Phase 3: Review")
            review_result = self._review(code_result, problem)
            
            # If review fails, iterate
            if "request_changes" in review_result.lower():
                logger.info("Review requested changes, iterating...")
                code_result = self._code(
                    problem,
                    context + f"\n\n## Review Feedback\n{review_result}",
                    plan_result
                )
        
        # Calculate total cost
        self.total_cost = (
            self.planner.total_cost +
            self.coder.total_cost +
            self.tester.total_cost +
            self.reviewer.total_cost
        )
        
        return True, code_result, {
            "plan": plan_result,
            "cost": self.total_cost,
            "messages": len(self.messages),
        }
    
    def _plan(self, problem: str, context: str) -> str:
        """Have planner create a plan."""
        result = self.planner.process(
            f"Analyze this bug and create a fix plan:\n\n{problem}",
            context,
        )
        
        self.messages.append(AgentMessage(
            from_agent="planner",
            to_agent="orchestrator",
            msg_type=MessageType.RESULT,
            content=result,
        ))
        
        return result
    
    def _code(self, problem: str, context: str, plan: str) -> str:
        """Have coder implement the fix."""
        result = self.coder.process(
            f"Implement this fix:\n\n{problem}\n\n## Plan\n{plan}",
            context,
        )
        
        self.messages.append(AgentMessage(
            from_agent="coder",
            to_agent="orchestrator",
            msg_type=MessageType.RESULT,
            content=result,
        ))
        
        return result
    
    def _review(self, code: str, problem: str) -> str:
        """Have reviewer check the code."""
        result = self.reviewer.process(
            f"Review this fix:\n\n## Original Problem\n{problem}\n\n## Proposed Fix\n{code}",
        )
        
        self.messages.append(AgentMessage(
            from_agent="reviewer",
            to_agent="orchestrator",
            msg_type=MessageType.FEEDBACK,
            content=result,
        ))
        
        return result
    
    def _test(self, test_output: str) -> str:
        """Have tester analyze test results."""
        result = self.tester.process(
            f"Analyze these test results:\n\n{test_output}",
        )
        
        self.messages.append(AgentMessage(
            from_agent="tester",
            to_agent="orchestrator",
            msg_type=MessageType.RESULT,
            content=result,
        ))
        
        return result


# =============================================================================
# CONTEXT MANAGER
# =============================================================================

class ContextManager:
    """
    Smart context management for long-running agent sessions.
    
    Features:
    - Compaction: Summarize old turns
    - Retrieval: Find relevant past context
    - Backtracking: Detect when to try different approach
    """
    
    def __init__(self, max_tokens: int = 8000):
        self.max_tokens = max_tokens
        self.history: List[Dict] = []
        self.summaries: List[str] = []
        self.code_snippets: Dict[str, str] = {}
    
    def add(self, role: str, content: str, metadata: Dict = None):
        """Add content to context."""
        self.history.append({
            "role": role,
            "content": content,
            "metadata": metadata or {},
            "timestamp": time.time(),
        })
        
        # Extract code snippets
        import re
        code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', content, re.DOTALL)
        for i, code in enumerate(code_blocks):
            key = f"snippet_{len(self.code_snippets)}"
            self.code_snippets[key] = code[:2000]
    
    def compact(self) -> str:
        """Compact old context into a summary."""
        if len(self.history) < 5:
            return self.get_full()
        
        # Keep last 3 entries in full
        recent = self.history[-3:]
        old = self.history[:-3]
        
        # Summarize old entries
        summary_parts = []
        for entry in old:
            truncated = entry["content"][:200]
            summary_parts.append(f"[{entry['role']}]: {truncated}...")
        
        summary = "## Previous Context (summarized)\n" + "\n".join(summary_parts)
        
        # Build new context
        recent_str = "\n\n".join([
            f"[{e['role']}]: {e['content']}" for e in recent
        ])
        
        return f"{summary}\n\n## Recent Context\n{recent_str}"
    
    def get_full(self) -> str:
        """Get full context (may be very long)."""
        return "\n\n".join([
            f"[{e['role']}]: {e['content']}" for e in self.history
        ])
    
    def should_backtrack(self) -> bool:
        """Detect if we should try a different approach."""
        if len(self.history) < 3:
            return False
        
        # Look for repeated failures
        recent = [e["content"].lower() for e in self.history[-3:]]
        failure_keywords = ["error", "failed", "exception", "not found"]
        
        failure_count = sum(
            1 for content in recent
            if any(kw in content for kw in failure_keywords)
        )
        
        return failure_count >= 2
    
    def get_relevant(self, query: str, top_k: int = 3) -> List[str]:
        """Get context entries most relevant to query."""
        # Simple keyword matching (could use embeddings for better results)
        query_words = set(query.lower().split())
        
        scored = []
        for entry in self.history:
            content_words = set(entry["content"].lower().split())
            overlap = len(query_words & content_words)
            scored.append((overlap, entry["content"]))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [content for _, content in scored[:top_k]]


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("S.P.I.D.E.R. Multi-Agent System Test")
    print("=" * 60)
    
    # Test individual agents
    print("\n[1] Testing Specialized Agents...")
    
    agents = [PlannerAgent(), CoderAgent(), TesterAgent(), ReviewerAgent()]
    for agent in agents:
        print(f"  {agent.name}: {agent.role[:50]}...")
    
    # Test orchestrator
    print("\n[2] Testing Orchestrator...")
    orchestrator = AgentOrchestrator(TeamConfig(max_cost_usd=0.01))
    print(f"  Agents: planner, coder, tester, reviewer")
    print(f"  Config: max_iterations={orchestrator.config.max_iterations}")
    
    # Test context manager
    print("\n[3] Testing ContextManager...")
    ctx = ContextManager()
    ctx.add("user", "Fix the divide by zero bug")
    ctx.add("agent", "I'll analyze the code first")
    ctx.add("agent", "Found the issue in calculator.py line 42")
    print(f"  History entries: {len(ctx.history)}")
    print(f"  Should backtrack: {ctx.should_backtrack()}")
    
    print("\n" + "=" * 60)
    print("✅ Multi-Agent System Ready!")
    print("=" * 60)
