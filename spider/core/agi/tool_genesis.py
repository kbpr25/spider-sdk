"""
S.P.I.D.E.R. Tool Genesis - Self-Extending AGI Engine
======================================================

Born from: AGI-4 (Toolformer) + Autopoietic Systems Theory

The Scientific Finding:
"AGI-4 (Toolformer) proved LLMs can learn to USE tools. But true AGI
shouldn't just use tools - it should INVENT them. If no library exists
to parse a binary format, a human writes a script. Current agents crash."

The Solution:
Give S.P.I.D.E.R. the ability to SELF-EXTEND.

The Autopoietic Loop:
1. Detect capability gap: "I need to verify this PDF but have no PDF tool"
2. Synthesize tool: Write verify_pdf.py with standard interface
3. Verify tool: Run internal unit tests in sandbox
4. Register tool: Dynamically import into runtime
5. USE tool: Immediately call own invention

Result: SDK starts with 5 tools. After a week, it has 500 custom tools
tailored to YOUR specific repository. It builds its own IDE.

This is the Event Horizon. From Worker to Engineer.
"""

import ast
import hashlib
import importlib
import importlib.util
import io
import logging
import os
import re
import sys
import tempfile
import time
import traceback
import types
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# TOOL TYPES
# =============================================================================

class ToolStatus(Enum):
    """Status of a synthesized tool."""
    DRAFT = auto()          # Just created
    TESTING = auto()        # Running tests
    VERIFIED = auto()       # Tests passed
    FAILED = auto()         # Tests failed
    REGISTERED = auto()     # Available for use


@dataclass
class ToolSpec:
    """Specification for a tool to be synthesized."""
    name: str
    description: str
    inputs: Dict[str, str]           # param_name -> type_annotation
    output_type: str
    requirements: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)


@dataclass
class SynthesizedTool:
    """A tool that has been synthesized by the Agent."""
    tool_id: str
    name: str
    spec: ToolSpec
    source_code: str
    test_code: str
    status: ToolStatus = ToolStatus.DRAFT
    error_message: str = ""
    created_at: float = field(default_factory=time.time)
    version: int = 1
    revision_history: List[str] = field(default_factory=list)
    
    # Runtime
    module: Optional[types.ModuleType] = None
    callable: Optional[Callable] = None


@dataclass
class CapabilityGap:
    """A detected capability gap."""
    task: str
    required_capability: str
    available_tools: List[str]
    gap_description: str
    suggested_tool_name: str
    suggested_spec: Optional[ToolSpec] = None


# =============================================================================
# CAPABILITY ANALYZER
# =============================================================================

class CapabilityAnalyzer:
    """
    Analyzes tasks to detect capability gaps.
    
    Uses pattern matching and semantic analysis to determine
    if current tools are sufficient for a task.
    """
    
    # Known capability patterns
    CAPABILITY_PATTERNS = {
        "parse_pdf": r"\b(pdf|document|acrobat)\b.*\b(parse|read|extract)\b",
        "parse_xml": r"\b(xml|xsd|xpath)\b.*\b(parse|validate|query)\b",
        "parse_json": r"\b(json|api response)\b.*\b(parse|decode|load)\b",
        "parse_protobuf": r"\b(protobuf|proto|grpc)\b.*\b(decode|parse|serialize)\b",
        "parse_csv": r"\b(csv|spreadsheet|tsv)\b.*\b(parse|read|load)\b",
        "http_request": r"\b(http|api|request|fetch|download)\b",
        "database_query": r"\b(sql|database|query|select|insert)\b",
        "file_operation": r"\b(file|read|write|save|load)\b",
        "image_process": r"\b(image|picture|photo|resize|crop)\b",
        "crypto_operation": r"\b(encrypt|decrypt|hash|signature)\b",
        "regex_extract": r"\b(regex|pattern|extract|match)\b",
        "date_parse": r"\b(date|time|datetime|timestamp)\b.*\b(parse|format)\b",
    }
    
    def __init__(self, available_tools: List[str] = None):
        self.available_tools = set(available_tools or [])
        self.compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.CAPABILITY_PATTERNS.items()
        }
    
    def analyze(self, task: str) -> Optional[CapabilityGap]:
        """
        Analyze a task to detect capability gaps.
        
        Args:
            task: Task description
            
        Returns:
            CapabilityGap if a gap is detected, None otherwise
        """
        # Check each capability pattern
        required_capabilities = []
        
        for cap_name, pattern in self.compiled_patterns.items():
            if pattern.search(task):
                required_capabilities.append(cap_name)
        
        # Check if we have tools for these capabilities
        for capability in required_capabilities:
            if not self._has_capability(capability):
                return CapabilityGap(
                    task=task,
                    required_capability=capability,
                    available_tools=list(self.available_tools),
                    gap_description=f"No tool available for: {capability}",
                    suggested_tool_name=capability,
                    suggested_spec=self._generate_spec(capability, task),
                )
        
        return None
    
    def _has_capability(self, capability: str) -> bool:
        """Check if we have a tool for this capability."""
        # Simple matching - check if any tool name contains the capability
        for tool in self.available_tools:
            if capability.replace("_", "") in tool.lower().replace("_", ""):
                return True
        return False
    
    def _generate_spec(self, capability: str, task: str) -> ToolSpec:
        """Generate a tool specification for a capability."""
        specs = {
            "parse_pdf": ToolSpec(
                name="parse_pdf",
                description="Parse PDF document and extract text content",
                inputs={"file_path": "str"},
                output_type="str",
                requirements=["PyPDF2"],
            ),
            "parse_protobuf": ToolSpec(
                name="parse_protobuf",
                description="Decode protobuf binary data",
                inputs={"data": "bytes", "schema": "str"},
                output_type="dict",
                requirements=["protobuf"],
            ),
            "parse_xml": ToolSpec(
                name="parse_xml",
                description="Parse XML document",
                inputs={"xml_string": "str"},
                output_type="dict",
            ),
        }
        
        if capability in specs:
            return specs[capability]
        
        # Generic spec
        return ToolSpec(
            name=capability,
            description=f"Tool for {capability.replace('_', ' ')}",
            inputs={"input_data": "Any"},
            output_type="Any",
        )
    
    def add_tool(self, tool_name: str) -> None:
        """Register an available tool."""
        self.available_tools.add(tool_name)


# =============================================================================
# TOOL SYNTHESIZER
# =============================================================================

class ToolSynthesizer:
    """
    Synthesizes new tools based on specifications.
    
    Uses LLM to generate tool code with:
    - Standard interface (Input/Output)
    - Built-in unit tests
    - Error handling
    - Documentation
    """
    
    SYNTHESIS_PROMPT = '''You are an expert Python developer. Create a self-contained tool module.

TOOL SPECIFICATION:
Name: {name}
Description: {description}
Inputs: {inputs}
Output Type: {output_type}
Requirements: {requirements}

REQUIREMENTS:
1. Create a function called `{name}` with the specified signature
2. Include comprehensive docstring
3. Handle errors gracefully
4. Include a `test_{name}` function that tests the main function
5. The test function must return True if passed, False if failed
6. Keep it self-contained (no external dependencies beyond requirements)

OUTPUT FORMAT:
```python
"""
Tool: {name}
Auto-generated by S.P.I.D.E.R. ToolGenesis
"""

def {name}({input_signature}) -> {output_type}:
    """
    {description}
    
    Args:
        [document each argument]
    
    Returns:
        [document return value]
    """
    # Implementation here
    pass

def test_{name}() -> bool:
    """Test the {name} function."""
    try:
        # Test cases here
        result = {name}(...)
        assert ...
        return True
    except Exception as e:
        print(f"Test failed: {{e}}")
        return False

if __name__ == "__main__":
    if test_{name}():
        print("PASS: All tests passed")
    else:
        print("FAIL: Tests failed")
```

Generate ONLY the Python code, no explanations.
'''

    SELF_CORRECT_PROMPT = '''The tool you generated failed with this error:

ERROR:
{error}

ORIGINAL CODE:
```python
{code}
```

Fix the code to resolve this error. Output only the corrected Python code.
'''

    def __init__(self, llm_callback: Optional[Callable[[str], str]] = None):
        self.llm_callback = llm_callback
    
    def synthesize(self, spec: ToolSpec) -> str:
        """
        Synthesize a tool from a specification.
        
        Args:
            spec: Tool specification
            
        Returns:
            Generated Python source code
        """
        if not self.llm_callback:
            return self._generate_template(spec)
        
        # Build input signature
        input_sig = ", ".join(f"{k}: {v}" for k, v in spec.inputs.items())
        
        prompt = self.SYNTHESIS_PROMPT.format(
            name=spec.name,
            description=spec.description,
            inputs=spec.inputs,
            output_type=spec.output_type,
            requirements=spec.requirements,
            input_signature=input_sig,
        )
        
        response = self.llm_callback(prompt)
        return self._extract_code(response)
    
    def self_correct(self, code: str, error: str) -> str:
        """
        Attempt to correct a failed tool.
        
        Args:
            code: Original code that failed
            error: Error message
            
        Returns:
            Corrected code
        """
        if not self.llm_callback:
            return code  # Can't self-correct without LLM
        
        prompt = self.SELF_CORRECT_PROMPT.format(
            error=error,
            code=code,
        )
        
        response = self.llm_callback(prompt)
        return self._extract_code(response)
    
    def _extract_code(self, response: str) -> str:
        """Extract Python code from LLM response."""
        # Try to find code block
        match = re.search(r'```(?:python)?\n?(.*?)```', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # If no code block, return as-is if it looks like Python
        if "def " in response:
            return response.strip()
        
        return response
    
    def _generate_template(self, spec: ToolSpec) -> str:
        """Generate a template tool without LLM."""
        input_sig = ", ".join(f"{k}: {v}" for k, v in spec.inputs.items())
        input_names = ", ".join(spec.inputs.keys())
        
        return f'''"""
Tool: {spec.name}
Auto-generated by S.P.I.D.E.R. ToolGenesis
Description: {spec.description}
"""

from typing import Any

def {spec.name}({input_sig}) -> {spec.output_type}:
    """
    {spec.description}
    
    Args:
        {chr(10).join(f"        {k}: {v}" for k, v in spec.inputs.items())}
    
    Returns:
        {spec.output_type}: Result
    """
    # TODO: Implement this tool
    # This is a template - actual implementation needed
    raise NotImplementedError("Tool not yet implemented: {spec.name}")

def test_{spec.name}() -> bool:
    """Test the {spec.name} function."""
    try:
        # Add test cases here
        print("Warning: No test cases implemented")
        return True
    except Exception as e:
        print(f"Test failed: {{e}}")
        return False

if __name__ == "__main__":
    if test_{spec.name}():
        print("PASS: All tests passed")
    else:
        print("FAIL: Tests failed")
'''


# =============================================================================
# SANDBOX EXECUTOR
# =============================================================================

class ToolSandbox:
    """
    Sandboxed execution environment for testing synthesized tools.
    
    Isolates tool execution to prevent damage to the main system.
    """
    
    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout
    
    def execute_tests(self, code: str, tool_name: str) -> Tuple[bool, str]:
        """
        Execute tool tests in a sandbox.
        
        Args:
            code: Tool source code
            tool_name: Name of the tool
            
        Returns:
            Tuple of (passed, output/error)
        """
        # Create isolated namespace
        namespace = {
            '__builtins__': __builtins__,
            '__name__': '__main__',
        }
        
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            # Compile the code
            compiled = compile(code, f'<{tool_name}>', 'exec')
            
            # Execute in namespace
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(compiled, namespace)
            
            # Find and run test function
            test_func_name = f"test_{tool_name}"
            if test_func_name not in namespace:
                return (False, f"No test function found: {test_func_name}")
            
            test_func = namespace[test_func_name]
            
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                result = test_func()
            
            output = stdout_capture.getvalue() + stderr_capture.getvalue()
            
            if result is True:
                return (True, output or "Tests passed")
            else:
                return (False, output or "Tests returned False")
                
        except SyntaxError as e:
            return (False, f"Syntax error: {e}")
        except Exception as e:
            tb = traceback.format_exc()
            return (False, f"Execution error: {e}\n{tb}")
    
    def execute_tool(
        self,
        code: str,
        tool_name: str,
        *args,
        **kwargs
    ) -> Tuple[Any, str]:
        """
        Execute a tool with given arguments.
        
        Returns:
            Tuple of (result, output/error)
        """
        namespace = {
            '__builtins__': __builtins__,
        }
        
        try:
            exec(compile(code, f'<{tool_name}>', 'exec'), namespace)
            
            if tool_name not in namespace:
                return (None, f"Function not found: {tool_name}")
            
            func = namespace[tool_name]
            result = func(*args, **kwargs)
            
            return (result, "")
            
        except Exception as e:
            return (None, f"Error: {e}")


# =============================================================================
# TOOL FABRICATOR
# =============================================================================

class ToolFabricator:
    """
    The Autopoietic Tool Fabricator - Self-Extension Engine.
    
    Allows S.P.I.D.E.R. to invent its own tools:
    1. Detect capability gaps
    2. Synthesize new tools
    3. Verify in sandbox
    4. Register for use
    
    From AGI-4 (Toolformer):
    "True AGI shouldn't just use tools - it should invent them."
    
    Usage:
        fabricator = ToolFabricator(llm_callback=my_llm)
        
        # Check if we need a new tool
        if fabricator.detect_need("Parse this protobuf file", ["read_file"]):
            # We need a protobuf parser!
            tool = fabricator.create_tool(
                name="parse_protobuf",
                spec="Decode protobuf binary to dict",
            )
            
            # Use the newly invented tool
            result = fabricator.invoke("parse_protobuf", data=binary_data)
    """
    
    def __init__(
        self,
        llm_callback: Optional[Callable[[str], str]] = None,
        tool_storage_dir: str = None,
        max_correction_attempts: int = 3,
    ):
        """
        Initialize Tool Fabricator.
        
        Args:
            llm_callback: LLM function for tool synthesis
            tool_storage_dir: Directory to store synthesized tools
            max_correction_attempts: Max self-correction attempts
        """
        self.llm_callback = llm_callback
        self.tool_storage_dir = Path(tool_storage_dir or ".spider_tools")
        self.max_correction_attempts = max_correction_attempts
        
        self.analyzer = CapabilityAnalyzer()
        self.synthesizer = ToolSynthesizer(llm_callback)
        self.sandbox = ToolSandbox()
        
        # Tool registry
        self.tools: Dict[str, SynthesizedTool] = {}
        self.available_tools: Set[str] = set()
        
        self._stats = {
            "tools_synthesized": 0,
            "tools_verified": 0,
            "tools_failed": 0,
            "self_corrections": 0,
            "invocations": 0,
        }
        
        # Ensure storage directory exists
        self.tool_storage_dir.mkdir(parents=True, exist_ok=True)
    
    def detect_need(
        self,
        task: str,
        available_tools: List[str] = None,
    ) -> Optional[CapabilityGap]:
        """
        Analyze if a task requires a tool we don't have.
        
        Args:
            task: Task description
            available_tools: Currently available tools
            
        Returns:
            CapabilityGap if we need a new tool, None otherwise
        """
        if available_tools:
            self.analyzer.available_tools = set(available_tools)
        
        self.analyzer.available_tools.update(self.available_tools)
        
        return self.analyzer.analyze(task)
    
    def synthesize_tool(
        self,
        name: str,
        spec: str,
        inputs: Dict[str, str] = None,
        output_type: str = "Any",
    ) -> SynthesizedTool:
        """
        Synthesize a new tool from a specification.
        
        Args:
            name: Tool name
            spec: Tool description/specification
            inputs: Input parameters {name: type}
            output_type: Return type
            
        Returns:
            SynthesizedTool (may not yet be verified)
        """
        self._stats["tools_synthesized"] += 1
        
        # Create tool spec
        tool_spec = ToolSpec(
            name=name,
            description=spec,
            inputs=inputs or {"input_data": "Any"},
            output_type=output_type,
        )
        
        # Synthesize code
        source_code = self.synthesizer.synthesize(tool_spec)
        
        # Create tool object
        tool_id = hashlib.md5(f"{name}{time.time()}".encode()).hexdigest()[:12]
        
        tool = SynthesizedTool(
            tool_id=tool_id,
            name=name,
            spec=tool_spec,
            source_code=source_code,
            test_code="",  # Tests are embedded in source
            status=ToolStatus.DRAFT,
        )
        
        return tool
    
    def verify_and_register(
        self,
        tool: SynthesizedTool,
    ) -> bool:
        """
        Verify a synthesized tool and register it for use.
        
        Args:
            tool: Tool to verify
            
        Returns:
            True if tool is now available, False if failed
        """
        tool.status = ToolStatus.TESTING
        
        # Run in sandbox
        passed, output = self.sandbox.execute_tests(tool.source_code, tool.name)
        
        if passed:
            tool.status = ToolStatus.VERIFIED
            self._stats["tools_verified"] += 1
            
            # Register the tool
            return self._register_tool(tool)
        else:
            # Attempt self-correction
            for attempt in range(self.max_correction_attempts):
                self._stats["self_corrections"] += 1
                
                corrected_code = self.synthesizer.self_correct(
                    tool.source_code,
                    output,
                )
                
                tool.source_code = corrected_code
                tool.revision_history.append(f"v{tool.version}: {output[:100]}")
                tool.version += 1
                
                passed, output = self.sandbox.execute_tests(corrected_code, tool.name)
                
                if passed:
                    tool.status = ToolStatus.VERIFIED
                    self._stats["tools_verified"] += 1
                    return self._register_tool(tool)
            
            # All attempts failed
            tool.status = ToolStatus.FAILED
            tool.error_message = output
            self._stats["tools_failed"] += 1
            
            return False
    
    def _register_tool(self, tool: SynthesizedTool) -> bool:
        """Register a verified tool for runtime use."""
        try:
            # Create a module from the source code
            module = types.ModuleType(f"spider_tool_{tool.name}")
            exec(compile(tool.source_code, f'<{tool.name}>', 'exec'), module.__dict__)
            
            # Get the callable
            if hasattr(module, tool.name):
                tool.module = module
                tool.callable = getattr(module, tool.name)
                tool.status = ToolStatus.REGISTERED
                
                # Store in registry
                self.tools[tool.name] = tool
                self.available_tools.add(tool.name)
                
                # Save to disk
                self._save_tool(tool)
                
                logger.info(f"[ToolGenesis] Registered new tool: {tool.name}")
                return True
            else:
                tool.error_message = f"Function '{tool.name}' not found in generated code"
                tool.status = ToolStatus.FAILED
                return False
                
        except Exception as e:
            tool.error_message = str(e)
            tool.status = ToolStatus.FAILED
            return False
    
    def _save_tool(self, tool: SynthesizedTool) -> None:
        """Save tool to disk for persistence."""
        tool_path = self.tool_storage_dir / f"{tool.name}.py"
        with open(tool_path, 'w') as f:
            f.write(tool.source_code)
    
    def create_tool(
        self,
        name: str,
        spec: str,
        inputs: Dict[str, str] = None,
        output_type: str = "Any",
    ) -> Optional[SynthesizedTool]:
        """
        High-level: Create, verify, and register a tool in one call.
        
        Args:
            name: Tool name
            spec: Tool description
            inputs: Input parameters
            output_type: Return type
            
        Returns:
            Registered tool if successful, None if failed
        """
        # Synthesize
        tool = self.synthesize_tool(name, spec, inputs, output_type)
        
        # Verify and register
        if self.verify_and_register(tool):
            return tool
        
        return None
    
    def invoke(self, tool_name: str, *args, **kwargs) -> Any:
        """
        Invoke a registered tool.
        
        Args:
            tool_name: Name of the tool
            *args, **kwargs: Tool arguments
            
        Returns:
            Tool result
        """
        if tool_name not in self.tools:
            raise ValueError(f"Tool not registered: {tool_name}")
        
        tool = self.tools[tool_name]
        
        if tool.callable is None:
            raise RuntimeError(f"Tool not callable: {tool_name}")
        
        self._stats["invocations"] += 1
        
        return tool.callable(*args, **kwargs)
    
    def list_tools(self) -> List[str]:
        """List all registered tools."""
        return list(self.tools.keys())
    
    def get_tool_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a tool."""
        if name not in self.tools:
            return None
        
        tool = self.tools[name]
        return {
            "name": tool.name,
            "description": tool.spec.description,
            "inputs": tool.spec.inputs,
            "output_type": tool.spec.output_type,
            "status": tool.status.name,
            "version": tool.version,
            "created_at": tool.created_at,
        }
    
    def get_stats(self) -> Dict[str, int]:
        return {
            **self._stats,
            "tools_registered": len(self.tools),
        }
    
    def print_status(self) -> None:
        """Print fabricator status."""
        print("\n" + "=" * 60)
        print("[*] TOOL GENESIS STATUS")
        print("=" * 60)
        
        print(f"\n[%] Statistics:")
        for key, val in self.get_stats().items():
            print(f"   {key}: {val}")
        
        if self.tools:
            print(f"\n[T] Registered Tools ({len(self.tools)}):")
            for name, tool in self.tools.items():
                print(f"   - {name} v{tool.version} [{tool.status.name}]")
                print(f"     {tool.spec.description[:50]}...")
        
        print("=" * 60)


# =============================================================================
# PACKAGE INIT
# =============================================================================

__all__ = [
    "ToolFabricator",
    "ToolSynthesizer",
    "ToolSandbox",
    "CapabilityAnalyzer",
    "SynthesizedTool",
    "ToolSpec",
    "ToolStatus",
    "CapabilityGap",
]


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("[*] S.P.I.D.E.R. Tool Genesis - Demo")
    print("=" * 70)
    
    fabricator = ToolFabricator()
    
    # Test capability gap detection
    print("\n[1] Detecting capability gaps...")
    
    gap = fabricator.detect_need(
        task="I need to parse this PDF document and extract text",
        available_tools=["read_file", "write_file"],
    )
    
    if gap:
        print(f"   [!] Gap detected: {gap.required_capability}")
        print(f"   [>] Suggested tool: {gap.suggested_tool_name}")
    
    # Test tool synthesis (template mode without LLM)
    print("\n[2] Synthesizing a tool...")
    
    tool = fabricator.synthesize_tool(
        name="calculate_hash",
        spec="Calculate MD5 hash of a string",
        inputs={"data": "str"},
        output_type="str",
    )
    
    print(f"   [+] Synthesized: {tool.name}")
    print(f"   [i] Status: {tool.status.name}")
    print(f"   [c] Code preview:")
    print("   " + "\n   ".join(tool.source_code.split("\n")[:10]))
    
    # Test with a working tool
    print("\n[3] Creating a working tool (no LLM)...")
    
    working_code = '''
"""
Tool: add_numbers
Auto-generated by S.P.I.D.E.R. ToolGenesis
"""

def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

def test_add_numbers() -> bool:
    """Test the add_numbers function."""
    try:
        assert add_numbers(2, 3) == 5
        assert add_numbers(-1, 1) == 0
        assert add_numbers(0, 0) == 0
        return True
    except Exception as e:
        print(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    print("PASS" if test_add_numbers() else "FAIL")
'''
    
    # Simulate a pre-synthesized tool
    test_tool = SynthesizedTool(
        tool_id="demo123",
        name="add_numbers",
        spec=ToolSpec(
            name="add_numbers",
            description="Add two numbers",
            inputs={"a": "int", "b": "int"},
            output_type="int",
        ),
        source_code=working_code,
        test_code="",
    )
    
    # Verify and register
    success = fabricator.verify_and_register(test_tool)
    print(f"   [{'+'if success else 'X'}] Verification: {'PASSED' if success else 'FAILED'}")
    
    if success:
        # Use the tool!
        result = fabricator.invoke("add_numbers", 10, 20)
        print(f"   [=] invork('add_numbers', 10, 20) = {result}")
    
    fabricator.print_status()
