"""
S.P.I.D.E.R. Agentic Core - The Nervous System
================================================

ReAct-style agentic loop with tool orchestration:
- Think: LLM reasons about current state
- Act: Execute a tool (read_file, search, run_tests, apply_patch)
- Observe: Parse result and update context

This is what makes Claude Opus 4.5 get 80% on SWE-Bench.
"""

import json
import os
import re
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# TOOL ABSTRACTION
# =============================================================================

class ToolStatus(Enum):
    """Status of a tool execution."""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class ToolResult:
    """Result from executing a tool."""
    tool_name: str
    status: ToolStatus
    output: str
    error: str = ""
    duration_ms: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        return self.status == ToolStatus.SUCCESS
    
    def to_message(self) -> str:
        """Format for LLM consumption."""
        if self.success:
            return f"[{self.tool_name}] Success:\n{self.output[:2000]}"
        else:
            return f"[{self.tool_name}] Error: {self.error}\n{self.output[:1000]}"


class Tool(ABC):
    """Abstract base class for agent tools."""
    
    name: str = "tool"
    description: str = "A tool"
    parameters: Dict[str, str] = {}
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        pass
    
    def schema(self) -> Dict:
        """Return tool schema for LLM."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


# =============================================================================
# CONCRETE TOOLS
# =============================================================================

class ReadFileTool(Tool):
    """Read contents of a file."""
    
    name = "read_file"
    description = "Read the contents of a file from the repository"
    parameters = {"path": "Path to the file to read"}
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
    
    def execute(self, path: str, **kwargs) -> ToolResult:
        start = time.time()
        try:
            full_path = self.repo_path / path
            if not full_path.exists():
                return ToolResult(
                    tool_name=self.name,
                    status=ToolStatus.ERROR,
                    output="",
                    error=f"File not found: {path}",
                )
            
            content = full_path.read_text(encoding='utf-8', errors='replace')
            
            # Add line numbers for easier reference
            lines = content.split('\n')
            numbered = '\n'.join(f"{i+1:4d} | {line}" for i, line in enumerate(lines))
            
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.SUCCESS,
                output=numbered[:8000],  # Limit output
                duration_ms=(time.time() - start) * 1000,
                metadata={"path": path, "lines": len(lines)},
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.ERROR,
                output="",
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )


class SearchCodeTool(Tool):
    """Search for patterns in the codebase."""
    
    name = "search_code"
    description = "Search for a pattern in the codebase using grep"
    parameters = {
        "pattern": "Pattern to search for",
        "file_pattern": "Optional file glob pattern (e.g., '*.py')",
    }
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
    
    def execute(self, pattern: str, file_pattern: str = "*.py", **kwargs) -> ToolResult:
        start = time.time()
        try:
            # Use grep or findstr depending on OS
            if os.name == 'nt':  # Windows
                cmd = f'findstr /S /N /C:"{pattern}" {file_pattern}'
            else:  # Unix
                cmd = f'grep -rn "{pattern}" --include="{file_pattern}"'
            
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            output = result.stdout[:5000]
            if not output:
                output = f"No matches found for '{pattern}'"
            
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.SUCCESS,
                output=output,
                duration_ms=(time.time() - start) * 1000,
                metadata={"pattern": pattern, "matches": output.count('\n')},
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.TIMEOUT,
                output="",
                error="Search timed out",
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.ERROR,
                output="",
                error=str(e),
            )


class ListFilesTool(Tool):
    """List files in a directory."""
    
    name = "list_files"
    description = "List files and directories in a path"
    parameters = {"path": "Directory path to list (default: root)"}
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
    
    def execute(self, path: str = ".", **kwargs) -> ToolResult:
        start = time.time()
        try:
            target = self.repo_path / path
            if not target.exists():
                return ToolResult(
                    tool_name=self.name,
                    status=ToolStatus.ERROR,
                    output="",
                    error=f"Path not found: {path}",
                )
            
            items = []
            for item in sorted(target.iterdir()):
                if item.name.startswith('.'):
                    continue
                prefix = "[DIR]" if item.is_dir() else "[FILE]"
                items.append(f"{prefix} {item.name}")
            
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.SUCCESS,
                output='\n'.join(items[:100]),
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.ERROR,
                output="",
                error=str(e),
            )


class RunTestsTool(Tool):
    """Run tests and return results."""
    
    name = "run_tests"
    description = "Run tests to verify the fix works"
    parameters = {
        "test_path": "Specific test file or directory (optional)",
        "test_filter": "Filter to run specific tests (optional)",
    }
    
    def __init__(self, repo_path: str = ".", timeout: int = 120):
        self.repo_path = Path(repo_path)
        self.timeout = timeout
    
    def execute(self, test_path: str = "", test_filter: str = "", **kwargs) -> ToolResult:
        start = time.time()
        try:
            cmd = ["python", "-m", "pytest", "-v", "--tb=short"]
            
            if test_path:
                cmd.append(test_path)
            if test_filter:
                cmd.extend(["-k", test_filter])
            
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            
            output = result.stdout + "\n" + result.stderr
            
            # Parse results
            passed = len(re.findall(r'PASSED', output))
            failed = len(re.findall(r'FAILED', output))
            
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.SUCCESS if result.returncode == 0 else ToolStatus.ERROR,
                output=output[-4000:],  # Last 4000 chars
                error="" if result.returncode == 0 else f"Tests failed: {failed} failures",
                duration_ms=(time.time() - start) * 1000,
                metadata={"passed": passed, "failed": failed, "exit_code": result.returncode},
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.TIMEOUT,
                output="",
                error=f"Tests timed out after {self.timeout}s",
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.ERROR,
                output="",
                error=str(e),
            )


class WriteFileTool(Tool):
    """Write content to a file."""
    
    name = "write_file"
    description = "Write or overwrite a file with new content"
    parameters = {
        "path": "Path to the file",
        "content": "The complete file content to write",
    }
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
    
    def execute(self, path: str, content: str, **kwargs) -> ToolResult:
        start = time.time()
        try:
            full_path = self.repo_path / path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding='utf-8')
            
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.SUCCESS,
                output=f"Successfully wrote {len(content)} chars to {path}",
                duration_ms=(time.time() - start) * 1000,
                metadata={"path": path, "size": len(content)},
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.ERROR,
                output="",
                error=str(e),
            )


class ApplyPatchTool(Tool):
    """Apply a git patch."""
    
    name = "apply_patch"
    description = "Apply a unified diff patch to the repository"
    parameters = {"patch": "The patch content in unified diff format"}
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
    
    def execute(self, patch: str, **kwargs) -> ToolResult:
        start = time.time()
        try:
            # Write patch to temp file
            patch_file = self.repo_path / ".spider_patch.tmp"
            patch_file.write_text(patch, encoding='utf-8')
            
            # Try to apply
            result = subprocess.run(
                ["git", "apply", "--verbose", str(patch_file)],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
            )
            
            patch_file.unlink(missing_ok=True)
            
            if result.returncode == 0:
                return ToolResult(
                    tool_name=self.name,
                    status=ToolStatus.SUCCESS,
                    output=f"Patch applied successfully\n{result.stdout}",
                    duration_ms=(time.time() - start) * 1000,
                )
            else:
                return ToolResult(
                    tool_name=self.name,
                    status=ToolStatus.ERROR,
                    output=result.stdout,
                    error=f"Patch failed: {result.stderr}",
                )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.ERROR,
                output="",
                error=str(e),
            )


class ThinkTool(Tool):
    """Think/reason about the problem (no-op but useful for structured thinking)."""
    
    name = "think"
    description = "Reason about the problem and plan next steps. Use this to organize your thoughts."
    parameters = {"thought": "Your reasoning and analysis"}
    
    def execute(self, thought: str, **kwargs) -> ToolResult:
        return ToolResult(
            tool_name=self.name,
            status=ToolStatus.SUCCESS,
            output=f"Thought recorded: {thought[:500]}...",
        )


class SubmitTool(Tool):
    """Submit the final solution."""
    
    name = "submit"
    description = "Submit your final solution. Only use when you're confident the fix is correct."
    parameters = {"explanation": "Brief explanation of what was fixed and why"}
    
    def execute(self, explanation: str, **kwargs) -> ToolResult:
        return ToolResult(
            tool_name=self.name,
            status=ToolStatus.SUCCESS,
            output=f"Solution submitted: {explanation}",
            metadata={"submitted": True},
        )


# =============================================================================
# TOOL REGISTRY
# =============================================================================

class ToolRegistry:
    """Registry of available tools."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = repo_path
        self._tools: Dict[str, Tool] = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register the default toolset."""
        tools = [
            ThinkTool(),
            ReadFileTool(self.repo_path),
            ListFilesTool(self.repo_path),
            SearchCodeTool(self.repo_path),
            WriteFileTool(self.repo_path),
            RunTestsTool(self.repo_path),
            ApplyPatchTool(self.repo_path),
            SubmitTool(),
        ]
        for tool in tools:
            self._tools[tool.name] = tool
    
    def register(self, tool: Tool):
        """Register a custom tool."""
        self._tools[tool.name] = tool
    
    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def execute(self, name: str, **kwargs) -> ToolResult:
        """Execute a tool by name."""
        tool = self.get(name)
        if not tool:
            return ToolResult(
                tool_name=name,
                status=ToolStatus.ERROR,
                output="",
                error=f"Unknown tool: {name}",
            )
        return tool.execute(**kwargs)
    
    def get_schemas(self) -> List[Dict]:
        """Get schemas for all tools."""
        return [tool.schema() for tool in self._tools.values()]
    
    def get_tool_descriptions(self) -> str:
        """Get formatted tool descriptions for prompt."""
        lines = ["Available tools:"]
        for tool in self._tools.values():
            params = ", ".join(f"{k}: {v}" for k, v in tool.parameters.items())
            lines.append(f"- {tool.name}({params}): {tool.description}")
        return "\n".join(lines)


# =============================================================================
# REACT AGENT
# =============================================================================

@dataclass
class AgentStep:
    """A single step in the agent's execution."""
    step_num: int
    thought: str
    action: str
    action_input: Dict[str, Any]
    observation: str
    timestamp: float = field(default_factory=time.time)


@dataclass 
class AgentConfig:
    """Configuration for the ReAct agent."""
    max_steps: int = 15              # Max reasoning steps
    max_cost_usd: float = 0.10       # Cost budget
    temperature: float = 0.3         # Lower = more deterministic
    model: str = "deepseek/deepseek-chat"


class ReActAgent:
    """
    ReAct (Reason + Act) Agent for autonomous problem solving.
    
    This is the core of Opus 4.5's 80% SWE-Bench performance.
    
    Loop:
    1. THINK: Reason about current state
    2. ACT: Choose and execute a tool
    3. OBSERVE: Process the result
    4. Repeat until solved or budget exhausted
    """
    
    SYSTEM_PROMPT = """You are an expert software engineer debugging a codebase.

You have access to these tools:
{tools}

## Response Format

You MUST respond in this exact JSON format:
{{
    "thought": "Your reasoning about what to do next",
    "action": "tool_name",
    "action_input": {{"param1": "value1", "param2": "value2"}}
}}

## Strategy

1. First, understand the problem by reading relevant files
2. Search for related code patterns
3. Identify the root cause
4. Make minimal changes to fix the issue
5. Run tests to verify
6. Submit when confident

## Rules

- ALWAYS respond with valid JSON
- Use "think" action to reason through complex problems
- Use "submit" action ONLY when you're confident the fix works
- Be minimal - change only what's necessary
- If tests fail, analyze the error and try again"""

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        repo_path: str = ".",
    ):
        self.config = config or AgentConfig()
        self.tools = ToolRegistry(repo_path)
        self.history: List[AgentStep] = []
        self.total_cost = 0.0
        
        # Import LLM gateway
        from spider.core.agent.llm_client import LLMGateway, Message, MessageRole
        self.gateway = LLMGateway()
        self.Message = Message
        self.MessageRole = MessageRole
    
    def solve(self, problem: str) -> Tuple[bool, str]:
        """
        Solve a problem using the ReAct loop.
        
        Args:
            problem: Problem description
            
        Returns:
            (success, explanation)
        """
        self.history = []
        
        messages = [
            self.Message(
                self.MessageRole.SYSTEM,
                self.SYSTEM_PROMPT.format(tools=self.tools.get_tool_descriptions())
            ),
            self.Message(
                self.MessageRole.USER,
                f"## Problem\n\n{problem}\n\nBegin by analyzing the problem and exploring the codebase."
            ),
        ]
        
        for step_num in range(self.config.max_steps):
            # Check budget
            if self.total_cost >= self.config.max_cost_usd:
                logger.warning(f"Budget exhausted: ${self.total_cost:.4f}")
                return False, "Budget exhausted"
            
            # Get next action from LLM
            response = self.gateway.complete(
                messages,
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=1000,
            )
            self.total_cost += response.cost_usd
            
            if not response.success:
                logger.error(f"LLM error: {response.content}")
                continue
            
            # Parse action
            action, action_input, thought = self._parse_response(response.content)
            
            if not action:
                # Ask for clarification
                messages.append(self.Message(
                    self.MessageRole.ASSISTANT,
                    response.content
                ))
                messages.append(self.Message(
                    self.MessageRole.USER,
                    "Please respond with valid JSON in the format: {\"thought\": \"...\", \"action\": \"...\", \"action_input\": {...}}"
                ))
                continue
            
            # Execute tool
            result = self.tools.execute(action, **action_input)
            
            # Record step
            step = AgentStep(
                step_num=step_num,
                thought=thought,
                action=action,
                action_input=action_input,
                observation=result.to_message(),
            )
            self.history.append(step)
            
            logger.info(f"Step {step_num}: {action} -> {result.status.value}")
            
            # Check for submission
            if action == "submit" and result.success:
                explanation = action_input.get("explanation", "Fix applied")
                return True, explanation
            
            # Add to conversation
            messages.append(self.Message(
                self.MessageRole.ASSISTANT,
                response.content
            ))
            messages.append(self.Message(
                self.MessageRole.USER,
                f"Observation:\n{result.to_message()}\n\nContinue with your next action."
            ))
        
        return False, "Max steps reached"
    
    def _parse_response(self, response: str) -> Tuple[str, Dict, str]:
        """Parse LLM response to extract action."""
        try:
            # Try to find JSON in response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                action = data.get("action", "")
                action_input = data.get("action_input", {})
                thought = data.get("thought", "")
                return action, action_input, thought
        except json.JSONDecodeError:
            pass
        
        return "", {}, ""
    
    def get_trace(self) -> str:
        """Get execution trace for debugging."""
        lines = ["## Agent Trace\n"]
        for step in self.history:
            lines.append(f"### Step {step.step_num}")
            lines.append(f"**Thought**: {step.thought}")
            lines.append(f"**Action**: {step.action}({step.action_input})")
            lines.append(f"**Observation**: {step.observation[:500]}...")
            lines.append("")
        return "\n".join(lines)


# =============================================================================
# QUICK TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("S.P.I.D.E.R. Agentic Core Test")
    print("=" * 60)
    
    # Test tool registry
    print("\n[1] Testing Tool Registry...")
    tools = ToolRegistry(".")
    print(f"  Registered tools: {list(tools._tools.keys())}")
    
    # Test individual tools
    print("\n[2] Testing ReadFileTool...")
    result = tools.execute("read_file", path="test_production.py")
    print(f"  Status: {result.status.value}")
    print(f"  Lines read: {result.metadata.get('lines', 0)}")
    
    print("\n[3] Testing ListFilesTool...")
    result = tools.execute("list_files", path="spider/core")
    print(f"  Status: {result.status.value}")
    print(f"  Output preview: {result.output[:200]}...")
    
    print("\n[4] Testing SearchCodeTool...")
    result = tools.execute("search_code", pattern="def solve", file_pattern="*.py")
    print(f"  Status: {result.status.value}")
    print(f"  Matches: {result.metadata.get('matches', 0)}")
    
    # Test agent initialization
    print("\n[5] Testing ReActAgent...")
    try:
        agent = ReActAgent(AgentConfig(max_steps=2, max_cost_usd=0.01))
        print(f"  Agent initialized!")
        print(f"  Tools: {list(agent.tools._tools.keys())}")
        print(f"  Model: {agent.config.model}")
    except Exception as e:
        print(f"  Agent init: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Agentic Core Ready!")
    print("=" * 60)
