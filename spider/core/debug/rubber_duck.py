"""
S.P.I.D.E.R. Rubber Duck Engine - Trace-Based Debugging
========================================================

Born from: Debug-2 (Self-Debugging), Debug-4 (Execution-Guided Synthesis)

The Scientific Finding:
"Simply showing error messages isn't enough. The model needs to perform
'Rubber Duck Debugging'—explaining the code line-by-line to itself.
LLMs are BETTER at debugging than coding (40% → 90% accuracy)."

The Solution:
We stop sending "Error Logs" to the LLM. We send "Execution Traces."

1. Instrument code to capture variable state at each line
2. Execute and capture the state history
3. Present trace to LLM: "Variable X changed from 5 to None at line 10"
4. LLM EXPLAINS first, then FIXES

Result: State-Based Repair instead of Error-Based Repair.
"""

import ast
import hashlib
import io
import logging
import re
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# TRACE TYPES
# =============================================================================

@dataclass
class TraceEntry:
    """A single trace entry capturing execution state."""
    line_no: int
    event: str                     # "line", "call", "return", "exception"
    func_name: str = ""
    variables: Dict[str, Any] = field(default_factory=dict)
    code_snippet: str = ""
    timestamp: float = 0.0


@dataclass
class StateChange:
    """Detected change in variable state."""
    var_name: str
    old_value: Any
    new_value: Any
    line_no: int
    code_snippet: str = ""


@dataclass
class ExecutionTrace:
    """Complete execution trace."""
    entries: List[TraceEntry] = field(default_factory=list)
    state_changes: List[StateChange] = field(default_factory=list)
    exception: Optional[str] = None
    exception_line: Optional[int] = None
    exception_traceback: str = ""
    final_variables: Dict[str, Any] = field(default_factory=dict)
    
    def get_trace_at_line(self, line_no: int) -> Optional[TraceEntry]:
        """Get trace entry at specific line."""
        for entry in reversed(self.entries):
            if entry.line_no == line_no:
                return entry
        return None
    
    def get_variable_history(self, var_name: str) -> List[Tuple[int, Any]]:
        """Get history of a variable's values."""
        history = []
        for entry in self.entries:
            if var_name in entry.variables:
                history.append((entry.line_no, entry.variables[var_name]))
        return history


@dataclass
class Diagnosis:
    """Diagnosis of a bug."""
    root_cause: str
    explanation: str
    suspicious_lines: List[int]
    suspicious_variables: List[str]
    suggested_fix: str = ""
    confidence: float = 0.0


# =============================================================================
# CODE INSTRUMENTER
# =============================================================================

class TraceInstrumenter(ast.NodeTransformer):
    """
    AST transformer that injects trace statements into code.
    
    Inserts print statements to capture variable state at key points:
    - Before/after assignments
    - At loop iterations
    - At function calls
    - At conditional branches
    """
    
    TRACE_PREFIX = "[TRACE]"
    
    def __init__(self, capture_all: bool = True):
        self.capture_all = capture_all
        self.line_offset = 0
    
    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        """Inject trace after assignment."""
        self.generic_visit(node)
        
        # Build trace statement
        trace_stmt = self._create_trace_stmt(
            node.lineno,
            "assign",
            node.targets[0] if node.targets else None
        )
        
        return [node, trace_stmt] if trace_stmt else node
    
    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:
        """Inject trace after augmented assignment."""
        self.generic_visit(node)
        
        trace_stmt = self._create_trace_stmt(node.lineno, "aug_assign", node.target)
        return [node, trace_stmt] if trace_stmt else node
    
    def visit_For(self, node: ast.For) -> ast.AST:
        """Inject trace at loop iteration."""
        self.generic_visit(node)
        
        # Add trace at start of loop body
        trace_stmt = self._create_trace_stmt(node.lineno, "for_iter", node.target)
        if trace_stmt:
            node.body.insert(0, trace_stmt)
        
        return node
    
    def visit_While(self, node: ast.While) -> ast.AST:
        """Inject trace at while loop."""
        self.generic_visit(node)
        
        trace_stmt = self._create_trace_stmt(node.lineno, "while")
        if trace_stmt:
            node.body.insert(0, trace_stmt)
        
        return node
    
    def visit_If(self, node: ast.If) -> ast.AST:
        """Inject trace at conditional."""
        self.generic_visit(node)
        
        # Trace the condition evaluation
        trace_stmt = self._create_trace_stmt(node.lineno, "if_branch")
        if trace_stmt:
            node.body.insert(0, trace_stmt)
            if node.orelse:
                else_trace = self._create_trace_stmt(node.lineno, "else_branch")
                if else_trace:
                    if isinstance(node.orelse[0], ast.If):
                        pass  # elif - will be handled recursively
                    else:
                        node.orelse.insert(0, else_trace)
        
        return node
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        """Inject trace at function entry."""
        self.generic_visit(node)
        
        # Trace function entry with arguments
        args_capture = self._create_args_trace(node)
        if args_capture:
            node.body.insert(0, args_capture)
        
        return node
    
    def visit_Return(self, node: ast.Return) -> ast.AST:
        """Inject trace before return."""
        self.generic_visit(node)
        
        trace_stmt = self._create_trace_stmt(node.lineno, "return")
        return [trace_stmt, node] if trace_stmt else node
    
    def _create_trace_stmt(
        self,
        lineno: int,
        event: str,
        target: ast.AST = None
    ) -> Optional[ast.Expr]:
        """Create a trace print statement."""
        # Build the print expression
        if target and isinstance(target, ast.Name):
            var_name = target.id
            # print(f"[TRACE] Line {lineno} {event}: {var_name}={repr(var_name)}")
            format_str = f'{self.TRACE_PREFIX} Line {lineno} {event}: {var_name}='
            print_call = ast.Expr(
                value=ast.Call(
                    func=ast.Name(id='print', ctx=ast.Load()),
                    args=[
                        ast.JoinedStr(values=[
                            ast.Constant(value=format_str),
                            ast.FormattedValue(
                                value=ast.Call(
                                    func=ast.Name(id='repr', ctx=ast.Load()),
                                    args=[ast.Name(id=var_name, ctx=ast.Load())],
                                    keywords=[]
                                ),
                                conversion=-1,
                            )
                        ])
                    ],
                    keywords=[]
                )
            )
        else:
            # Simple trace without variable
            print_call = ast.Expr(
                value=ast.Call(
                    func=ast.Name(id='print', ctx=ast.Load()),
                    args=[ast.Constant(value=f'{self.TRACE_PREFIX} Line {lineno} {event}')],
                    keywords=[]
                )
            )
        
        ast.fix_missing_locations(print_call)
        return print_call
    
    def _create_args_trace(self, func_node: ast.FunctionDef) -> Optional[ast.Expr]:
        """Create trace for function arguments."""
        args = [arg.arg for arg in func_node.args.args]
        if not args:
            return None
        
        # Build locals capture
        # print(f"[TRACE] Line {lineno} call {func_name}: {locals()}")
        format_str = f'{self.TRACE_PREFIX} Line {func_node.lineno} call {func_node.name}: '
        
        print_call = ast.Expr(
            value=ast.Call(
                func=ast.Name(id='print', ctx=ast.Load()),
                args=[
                    ast.BinOp(
                        left=ast.Constant(value=format_str),
                        op=ast.Add(),
                        right=ast.Call(
                            func=ast.Name(id='str', ctx=ast.Load()),
                            args=[
                                ast.Dict(
                                    keys=[ast.Constant(value=a) for a in args],
                                    values=[ast.Name(id=a, ctx=ast.Load()) for a in args]
                                )
                            ],
                            keywords=[]
                        )
                    )
                ],
                keywords=[]
            )
        )
        
        ast.fix_missing_locations(print_call)
        return print_call


# =============================================================================
# TRACE CAPTURE
# =============================================================================

class TraceCapture:
    """
    Captures execution traces by running instrumented code.
    
    Usage:
        capture = TraceCapture()
        
        # Instrument and run
        trace = capture.run('''
def foo(x):
    y = x * 2
    return y
    
result = foo(5)
''')
        
        # Analyze trace
        print(trace.entries)
        print(trace.state_changes)
    """
    
    def __init__(
        self,
        timeout: float = 5.0,
        capture_locals: bool = True,
        max_trace_entries: int = 1000,
    ):
        self.timeout = timeout
        self.capture_locals = capture_locals
        self.max_trace_entries = max_trace_entries
        self.instrumenter = TraceInstrumenter()
    
    def instrument_code(self, code: str) -> str:
        """
        Instrument code with trace statements.
        
        Args:
            code: Source code to instrument
            
        Returns:
            Instrumented code string
        """
        try:
            tree = ast.parse(code)
            instrumented_tree = self.instrumenter.visit(tree)
            ast.fix_missing_locations(instrumented_tree)
            return ast.unparse(instrumented_tree)
        except SyntaxError as e:
            logger.warning(f"Failed to instrument code: {e}")
            return code
    
    def run(
        self,
        code: str,
        globals_dict: Dict[str, Any] = None,
        instrument: bool = True,
    ) -> ExecutionTrace:
        """
        Run code and capture execution trace.
        
        Args:
            code: Source code to execute
            globals_dict: Optional global variables
            instrument: Whether to instrument code
            
        Returns:
            ExecutionTrace with full state history
        """
        trace = ExecutionTrace()
        
        # Instrument code
        if instrument:
            exec_code = self.instrument_code(code)
        else:
            exec_code = code
        
        # Prepare execution environment
        exec_globals = globals_dict or {}
        exec_globals['__builtins__'] = __builtins__
        exec_locals = {}
        
        # Capture stdout/stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        # State tracking for change detection
        previous_state: Dict[str, Any] = {}
        
        def trace_callback(frame, event, arg):
            """Trace function for sys.settrace."""
            if len(trace.entries) >= self.max_trace_entries:
                return None
            
            # Only trace user code
            if frame.f_code.co_filename != '<string>':
                return trace_callback
            
            # Capture entry
            entry = TraceEntry(
                line_no=frame.f_lineno,
                event=event,
                func_name=frame.f_code.co_name,
            )
            
            # Capture local variables
            if self.capture_locals:
                for name, value in frame.f_locals.items():
                    if not name.startswith('_'):
                        try:
                            entry.variables[name] = repr(value)[:100]
                        except Exception:
                            entry.variables[name] = "<error>"
                        
                        # Detect state changes
                        key = f"{frame.f_code.co_name}.{name}"
                        old = previous_state.get(key)
                        new = entry.variables[name]
                        if old is not None and old != new:
                            trace.state_changes.append(StateChange(
                                var_name=name,
                                old_value=old,
                                new_value=new,
                                line_no=frame.f_lineno,
                            ))
                        previous_state[key] = new
            
            trace.entries.append(entry)
            return trace_callback
        
        # Execute with tracing
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                sys.settrace(trace_callback)
                try:
                    exec(compile(exec_code, '<string>', 'exec'), exec_globals, exec_locals)
                finally:
                    sys.settrace(None)
            
            # Capture final state
            trace.final_variables = {
                k: repr(v)[:100] for k, v in exec_locals.items()
                if not k.startswith('_')
            }
            
        except Exception as e:
            trace.exception = str(e)
            trace.exception_traceback = traceback.format_exc()
            
            # Try to extract exception line
            tb = sys.exc_info()[2]
            if tb:
                while tb.tb_next:
                    tb = tb.tb_next
                trace.exception_line = tb.tb_lineno
        
        # Parse trace output for additional info
        stdout_content = stdout_capture.getvalue()
        self._parse_trace_output(stdout_content, trace)
        
        return trace
    
    def _parse_trace_output(self, output: str, trace: ExecutionTrace) -> None:
        """Parse [TRACE] lines from output."""
        for line in output.split('\n'):
            if line.startswith('[TRACE]'):
                # Parse trace line
                match = re.match(
                    r'\[TRACE\] Line (\d+) (\w+)(?:: (\w+)=(.+))?',
                    line
                )
                if match:
                    lineno = int(match.group(1))
                    event = match.group(2)
                    var_name = match.group(3)
                    var_value = match.group(4)
                    
                    # Update or add entry
                    found = False
                    for entry in trace.entries:
                        if entry.line_no == lineno:
                            if var_name:
                                entry.variables[var_name] = var_value
                            found = True
                            break
                    
                    if not found:
                        entry = TraceEntry(
                            line_no=lineno,
                            event=event,
                            variables={var_name: var_value} if var_name else {}
                        )
                        trace.entries.append(entry)
    
    def run_with_tests(
        self,
        code: str,
        test_code: str,
    ) -> Tuple[ExecutionTrace, bool]:
        """
        Run code with test suite and capture trace on failure.
        
        Returns:
            Tuple of (trace, test_passed)
        """
        combined_code = f"{code}\n\n# Tests\n{test_code}"
        trace = self.run(combined_code)
        passed = trace.exception is None
        return (trace, passed)


# =============================================================================
# RUBBER DUCK AGENT
# =============================================================================

class RubberDuckAgent:
    """
    The Rubber Duck Debugging Agent.
    
    Uses execution traces to guide LLM debugging:
    1. Presents state history to LLM
    2. Asks LLM to EXPLAIN the bug first
    3. Then asks for the FIX
    
    From Debug-2: "LLMs are better at debugging than coding"
    
    Usage:
        duck = RubberDuckAgent()
        
        diagnosis = duck.diagnose(
            code=buggy_code,
            error="IndexError: list index out of range",
            trace=execution_trace,
            llm_callback=my_llm,
        )
        
        print(diagnosis.root_cause)
        print(diagnosis.suggested_fix)
    """
    
    DIAGNOSIS_PROMPT = """You are a Senior Debugging Engineer performing "Rubber Duck Debugging."

BUGGY CODE:
```python
{code}
```

ERROR:
{error}

EXECUTION TRACE (Variable states over time):
{trace}

STATE CHANGES DETECTED:
{state_changes}

EXCEPTION OCCURRED AT: Line {exception_line}

Your task:
1. EXPLAIN step-by-step what the code was doing before it crashed
2. IDENTIFY the exact line and variable that caused the bug
3. EXPLAIN WHY that variable has an unexpected value
4. Suggest the ROOT CAUSE (not just the symptom)

Respond in this format:
EXPLANATION: [Step-by-step execution analysis]
ROOT_CAUSE: [One sentence describing the fundamental bug]
SUSPICIOUS_LINES: [Comma-separated line numbers]
SUSPICIOUS_VARIABLES: [Comma-separated variable names]
CONFIDENCE: [0-100]
"""

    FIX_PROMPT = """Based on your diagnosis:

ROOT CAUSE: {root_cause}
EXPLANATION: {explanation}
SUSPICIOUS LINES: {suspicious_lines}

Now write the FIXED code. Only output the corrected Python code, nothing else.
Preserve the original structure but fix the bug.

ORIGINAL CODE:
```python
{code}
```

FIXED CODE:
"""
    
    def __init__(self):
        self.capture = TraceCapture()
        self._stats = {
            "diagnoses": 0,
            "fixes_generated": 0,
            "avg_confidence": 0.0,
        }
    
    def diagnose(
        self,
        code: str,
        error: str,
        trace: ExecutionTrace = None,
        llm_callback: Optional[Callable[[str], str]] = None,
    ) -> Diagnosis:
        """
        Diagnose a bug using execution trace.
        
        Args:
            code: The buggy source code
            error: The error message/exception
            trace: Execution trace (if None, will generate)
            llm_callback: LLM function for diagnosis
            
        Returns:
            Diagnosis with root cause and suggested fix
        """
        self._stats["diagnoses"] += 1
        
        # Generate trace if not provided
        if trace is None:
            trace = self.capture.run(code)
        
        # Format trace for LLM
        trace_text = self._format_trace(trace)
        state_changes_text = self._format_state_changes(trace)
        
        # Use LLM for diagnosis
        if llm_callback:
            prompt = self.DIAGNOSIS_PROMPT.format(
                code=code,
                error=error,
                trace=trace_text,
                state_changes=state_changes_text,
                exception_line=trace.exception_line or "Unknown",
            )
            
            response = llm_callback(prompt)
            diagnosis = self._parse_diagnosis(response)
        else:
            # Heuristic diagnosis without LLM
            diagnosis = self._heuristic_diagnosis(code, error, trace)
        
        return diagnosis
    
    def diagnose_and_fix(
        self,
        code: str,
        error: str,
        llm_callback: Callable[[str], str],
        max_attempts: int = 3,
    ) -> Tuple[str, Diagnosis, bool]:
        """
        Diagnose bug and generate fix.
        
        Returns:
            Tuple of (fixed_code, diagnosis, verified)
        """
        trace = self.capture.run(code)
        diagnosis = self.diagnose(code, error, trace, llm_callback)
        
        # Generate fix
        fix_prompt = self.FIX_PROMPT.format(
            root_cause=diagnosis.root_cause,
            explanation=diagnosis.explanation,
            suspicious_lines=", ".join(str(l) for l in diagnosis.suspicious_lines),
            code=code,
        )
        
        fix_response = llm_callback(fix_prompt)
        fixed_code = self._extract_code(fix_response)
        self._stats["fixes_generated"] += 1
        
        # Verify fix
        new_trace = self.capture.run(fixed_code)
        verified = new_trace.exception is None
        
        return (fixed_code, diagnosis, verified)
    
    def _format_trace(self, trace: ExecutionTrace) -> str:
        """Format trace for LLM consumption."""
        lines = []
        seen_lines = set()
        
        for entry in trace.entries[-50:]:  # Last 50 entries
            if entry.line_no in seen_lines:
                continue
            seen_lines.add(entry.line_no)
            
            vars_str = ", ".join(
                f"{k}={v}" for k, v in list(entry.variables.items())[:5]
            )
            lines.append(f"Line {entry.line_no} [{entry.event}]: {vars_str}")
        
        return "\n".join(lines) or "No trace captured"
    
    def _format_state_changes(self, trace: ExecutionTrace) -> str:
        """Format state changes for LLM."""
        changes = []
        for change in trace.state_changes[-20:]:  # Last 20 changes
            changes.append(
                f"Line {change.line_no}: {change.var_name} changed from "
                f"{change.old_value} → {change.new_value}"
            )
        return "\n".join(changes) or "No state changes detected"
    
    def _parse_diagnosis(self, response: str) -> Diagnosis:
        """Parse LLM diagnosis response."""
        # Extract fields
        explanation = self._extract_field(response, "EXPLANATION")
        root_cause = self._extract_field(response, "ROOT_CAUSE")
        suspicious_lines_str = self._extract_field(response, "SUSPICIOUS_LINES")
        suspicious_vars_str = self._extract_field(response, "SUSPICIOUS_VARIABLES")
        confidence_str = self._extract_field(response, "CONFIDENCE")
        
        # Parse lines
        suspicious_lines = []
        if suspicious_lines_str:
            for part in suspicious_lines_str.split(","):
                try:
                    suspicious_lines.append(int(part.strip()))
                except ValueError:
                    pass
        
        # Parse variables
        suspicious_vars = []
        if suspicious_vars_str:
            suspicious_vars = [v.strip() for v in suspicious_vars_str.split(",")]
        
        # Parse confidence
        try:
            confidence = float(confidence_str) / 100.0 if confidence_str else 0.5
        except ValueError:
            confidence = 0.5
        
        return Diagnosis(
            root_cause=root_cause or "Unable to determine",
            explanation=explanation or response,
            suspicious_lines=suspicious_lines,
            suspicious_variables=suspicious_vars,
            confidence=confidence,
        )
    
    def _extract_field(self, text: str, field: str) -> str:
        """Extract field value from structured response."""
        pattern = rf'{field}:\s*(.+?)(?=\n[A-Z_]+:|$)'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""
    
    def _extract_code(self, response: str) -> str:
        """Extract code block from LLM response."""
        # Try to find code block
        match = re.search(r'```(?:python)?\n?(.*?)```', response, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Return as-is if no code block
        return response.strip()
    
    def _heuristic_diagnosis(
        self,
        code: str,
        error: str,
        trace: ExecutionTrace,
    ) -> Diagnosis:
        """Generate diagnosis using heuristics (no LLM)."""
        suspicious_lines = []
        suspicious_vars = []
        root_cause = "Unable to determine without LLM"
        
        # Look for common patterns in error
        if "IndexError" in error:
            root_cause = "Array index out of bounds"
            for change in trace.state_changes:
                if "len" in str(change.old_value) or "[]" in str(change.new_value):
                    suspicious_lines.append(change.line_no)
                    suspicious_vars.append(change.var_name)
        
        elif "KeyError" in error:
            root_cause = "Dictionary key not found"
            match = re.search(r"KeyError: '?(\w+)'?", error)
            if match:
                suspicious_vars.append(match.group(1))
        
        elif "TypeError" in error:
            root_cause = "Type mismatch or None value"
            for change in trace.state_changes:
                if "None" in str(change.new_value):
                    suspicious_lines.append(change.line_no)
                    suspicious_vars.append(change.var_name)
        
        elif "AttributeError" in error:
            root_cause = "Accessing attribute on None or wrong type"
            match = re.search(r"'(\w+)' object has no attribute", error)
            if match:
                root_cause = f"Object of type '{match.group(1)}' doesn't have expected attribute"
        
        # Add exception line
        if trace.exception_line:
            suspicious_lines.append(trace.exception_line)
        
        return Diagnosis(
            root_cause=root_cause,
            explanation=f"Error occurred: {error}. Trace shows {len(trace.state_changes)} state changes.",
            suspicious_lines=list(set(suspicious_lines)),
            suspicious_variables=list(set(suspicious_vars)),
            confidence=0.3,
        )
    
    def get_stats(self) -> Dict[str, Any]:
        return self._stats.copy()
    
    def print_diagnosis(self, diagnosis: Diagnosis) -> None:
        """Print formatted diagnosis."""
        print("\n" + "=" * 60)
        print("[DUCK] RUBBER DUCK DIAGNOSIS")
        print("=" * 60)
        
        print(f"\n[*] ROOT CAUSE: {diagnosis.root_cause}")
        print(f"\n[i] EXPLANATION:")
        print(f"   {diagnosis.explanation[:300]}...")
        
        print(f"\n[?] SUSPICIOUS LINES: {diagnosis.suspicious_lines}")
        print(f"[?] SUSPICIOUS VARIABLES: {diagnosis.suspicious_variables}")
        print(f"\n[%] CONFIDENCE: {diagnosis.confidence:.0%}")
        
        print("=" * 60)


# =============================================================================
# PACKAGE INIT
# =============================================================================

__all__ = [
    "RubberDuckAgent",
    "TraceCapture",
    "ExecutionTrace",
    "Diagnosis",
    "TraceEntry",
    "StateChange",
    "TraceInstrumenter",
]


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("[DUCK] S.P.I.D.E.R. Rubber Duck Engine - Demo")
    print("=" * 70)
    
    duck = RubberDuckAgent()
    
    # Buggy code with off-by-one error
    buggy_code = '''
def get_last_item(items):
    """Get the last item from a list."""
    length = len(items)
    last_index = length  # BUG: should be length - 1
    return items[last_index]

# Test
result = get_last_item([1, 2, 3, 4, 5])
print(f"Last item: {result}")
'''
    
    print("\n[i] BUGGY CODE:")
    print(buggy_code)
    
    # Capture trace
    capture = TraceCapture()
    trace = capture.run(buggy_code)
    
    print("\n[*] EXECUTION TRACE:")
    for entry in trace.entries[:10]:
        print(f"   Line {entry.line_no} [{entry.event}]: {dict(list(entry.variables.items())[:3])}")
    
    print(f"\n[X] EXCEPTION: {trace.exception}")
    print(f"   At line: {trace.exception_line}")
    
    print("\n[~] STATE CHANGES:")
    for change in trace.state_changes[:5]:
        print(f"   Line {change.line_no}: {change.var_name}: {change.old_value} -> {change.new_value}")
    
    # Diagnose (without LLM - heuristic mode)
    diagnosis = duck.diagnose(
        code=buggy_code,
        error=trace.exception or "Unknown error",
        trace=trace,
    )
    
    duck.print_diagnosis(diagnosis)
    
    print(f"\n[%] Stats: {duck.get_stats()}")
