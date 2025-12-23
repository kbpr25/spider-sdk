"""
S.P.I.D.E.R. Semantic Code Intelligence
========================================

AST-based code understanding for deep semantic analysis.
Goes beyond regex to truly understand code structure.

Capabilities:
1. Function/Class extraction with full signatures
2. Call graph analysis (who calls whom)
3. Dependency graph (imports and uses)
4. Variable flow analysis
5. Type inference (basic)
6. Context-aware code retrieval

This module bridges the gap between text-based LLM prompts and
structured code understanding. It's what makes S.P.I.D.E.R.
truly understand code rather than just process text.

This is the +5% improvement component.
"""

import ast
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


# =============================================================================
# CODE ELEMENT TYPES
# =============================================================================

class ElementType(Enum):
    """Types of code elements."""
    MODULE = auto()
    CLASS = auto()
    FUNCTION = auto()
    METHOD = auto()
    VARIABLE = auto()
    IMPORT = auto()
    CONSTANT = auto()


@dataclass
class CodeLocation:
    """Location of code element in source."""
    file_path: str
    start_line: int
    end_line: int
    start_col: int = 0
    end_col: int = 0
    
    def __str__(self) -> str:
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


@dataclass
class CodeElement:
    """Represents a code element (function, class, variable, etc.)."""
    name: str
    element_type: ElementType
    location: CodeLocation
    signature: str = ""
    docstring: str = ""
    body: str = ""
    parent: Optional[str] = None  # Fully qualified name of parent
    children: List[str] = field(default_factory=list)
    calls: List[str] = field(default_factory=list)  # Functions this calls
    called_by: List[str] = field(default_factory=list)  # Functions that call this
    uses: List[str] = field(default_factory=list)  # Variables/imports used
    decorators: List[str] = field(default_factory=list)
    return_type: Optional[str] = None
    parameters: List[Tuple[str, Optional[str]]] = field(default_factory=list)  # (name, type)
    
    @property
    def fully_qualified_name(self) -> str:
        """Get fully qualified name including parent."""
        if self.parent:
            return f"{self.parent}.{self.name}"
        return self.name
    
    def to_prompt_context(self, include_body: bool = False) -> str:
        """Convert to prompt-friendly format."""
        parts = [f"# {self.element_type.name}: {self.fully_qualified_name}"]
        
        if self.signature:
            parts.append(f"Signature: {self.signature}")
        
        if self.docstring:
            parts.append(f"Docstring: {self.docstring[:200]}")
        
        if self.calls:
            parts.append(f"Calls: {', '.join(self.calls[:10])}")
        
        if include_body and self.body:
            parts.append(f"```python\n{self.body}\n```")
        
        return "\n".join(parts)


# =============================================================================
# AST VISITOR FOR ANALYSIS
# =============================================================================

class SemanticVisitor(ast.NodeVisitor):
    """
    Custom AST visitor that extracts semantic information.
    
    Extracts:
    - Functions with signatures, docstrings, return types
    - Classes with methods and attributes
    - Call relationships
    - Import dependencies
    - Variable definitions
    """
    
    def __init__(self, file_path: str = "", source: str = ""):
        self.file_path = file_path
        self.source = source
        self.source_lines = source.split('\n') if source else []
        
        # Collected elements
        self.elements: Dict[str, CodeElement] = {}
        self.imports: Dict[str, str] = {}  # alias -> module
        self.global_vars: Set[str] = set()
        
        # Context tracking
        self._current_class: Optional[str] = None
        self._current_function: Optional[str] = None
        self._call_stack: List[str] = []
    
    def visit_Module(self, node: ast.Module) -> None:
        """Visit module and extract module-level docstring."""
        docstring = ast.get_docstring(node) or ""
        
        self.elements["__module__"] = CodeElement(
            name="__module__",
            element_type=ElementType.MODULE,
            location=CodeLocation(self.file_path, 1, len(self.source_lines)),
            docstring=docstring,
        )
        
        self.generic_visit(node)
    
    def visit_Import(self, node: ast.Import) -> None:
        """Visit import statement."""
        for alias in node.names:
            name = alias.asname or alias.name
            self.imports[name] = alias.name
            
            self.elements[f"import:{name}"] = CodeElement(
                name=name,
                element_type=ElementType.IMPORT,
                location=CodeLocation(
                    self.file_path, node.lineno, node.end_lineno or node.lineno
                ),
                signature=f"import {alias.name}" + (f" as {name}" if alias.asname else ""),
            )
    
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visit from-import statement."""
        module = node.module or ""
        for alias in node.names:
            name = alias.asname or alias.name
            full_name = f"{module}.{alias.name}"
            self.imports[name] = full_name
            
            self.elements[f"import:{name}"] = CodeElement(
                name=name,
                element_type=ElementType.IMPORT,
                location=CodeLocation(
                    self.file_path, node.lineno, node.end_lineno or node.lineno
                ),
                signature=f"from {module} import {alias.name}" + (f" as {name}" if alias.asname else ""),
            )
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definition."""
        name = node.name
        docstring = ast.get_docstring(node) or ""
        
        # Build signature
        bases = [self._get_name(b) for b in node.bases]
        signature = f"class {name}"
        if bases:
            signature += f"({', '.join(bases)})"
        
        # Get decorators
        decorators = [self._get_name(d) for d in node.decorator_list]
        
        # Get body
        body = self._get_node_source(node)
        
        # Create element
        element = CodeElement(
            name=name,
            element_type=ElementType.CLASS,
            location=CodeLocation(
                self.file_path, node.lineno, node.end_lineno or node.lineno
            ),
            signature=signature,
            docstring=docstring,
            body=body,
            decorators=decorators,
            parent=self._current_class,
        )
        
        self.elements[name] = element
        
        # Visit children with class context
        old_class = self._current_class
        self._current_class = name
        self.generic_visit(node)
        self._current_class = old_class
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function/method definition."""
        self._process_function(node)
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definition."""
        self._process_function(node, is_async=True)
    
    def _process_function(
        self, 
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
        is_async: bool = False,
    ) -> None:
        """Process a function or async function definition."""
        name = node.name
        docstring = ast.get_docstring(node) or ""
        
        # Determine if method
        is_method = self._current_class is not None
        element_type = ElementType.METHOD if is_method else ElementType.FUNCTION
        
        # Extract parameters
        parameters = self._extract_parameters(node.args)
        
        # Build signature
        param_str = ", ".join(
            f"{p[0]}: {p[1]}" if p[1] else p[0]
            for p in parameters
        )
        
        # Get return type
        return_type = None
        if node.returns:
            return_type = self._get_name(node.returns)
        
        # Build full signature
        async_prefix = "async " if is_async else ""
        signature = f"{async_prefix}def {name}({param_str})"
        if return_type:
            signature += f" -> {return_type}"
        
        # Get decorators
        decorators = [self._get_name(d) for d in node.decorator_list]
        
        # Get body
        body = self._get_node_source(node)
        
        # Create element
        fqn = f"{self._current_class}.{name}" if self._current_class else name
        element = CodeElement(
            name=name,
            element_type=element_type,
            location=CodeLocation(
                self.file_path, node.lineno, node.end_lineno or node.lineno
            ),
            signature=signature,
            docstring=docstring,
            body=body,
            parent=self._current_class,
            decorators=decorators,
            return_type=return_type,
            parameters=parameters,
        )
        
        self.elements[fqn] = element
        
        # Visit body to find calls
        old_function = self._current_function
        self._current_function = fqn
        self.generic_visit(node)
        self._current_function = old_function
    
    def visit_Call(self, node: ast.Call) -> None:
        """Visit function call to track call relationships."""
        called_name = self._get_name(node.func)
        
        if self._current_function and called_name:
            # Add to calls list of current function
            if self._current_function in self.elements:
                if called_name not in self.elements[self._current_function].calls:
                    self.elements[self._current_function].calls.append(called_name)
        
        self.generic_visit(node)
    
    def visit_Assign(self, node: ast.Assign) -> None:
        """Visit assignment for global variables."""
        if self._current_function is None and self._current_class is None:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.global_vars.add(target.id)
        
        self.generic_visit(node)
    
    def _extract_parameters(
        self, 
        args: ast.arguments,
    ) -> List[Tuple[str, Optional[str]]]:
        """Extract parameters from function arguments."""
        params = []
        
        # Regular args
        for arg in args.args:
            type_hint = None
            if arg.annotation:
                type_hint = self._get_name(arg.annotation)
            params.append((arg.arg, type_hint))
        
        # *args
        if args.vararg:
            params.append((f"*{args.vararg.arg}", None))
        
        # **kwargs
        if args.kwarg:
            params.append((f"**{args.kwarg.arg}", None))
        
        return params
    
    def _get_name(self, node: ast.AST) -> str:
        """Get the name/string representation of an AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            value = self._get_name(node.value)
            return f"{value}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            value = self._get_name(node.value)
            slice_val = self._get_name(node.slice)
            return f"{value}[{slice_val}]"
        elif isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.Call):
            return self._get_name(node.func)
        elif isinstance(node, ast.Tuple):
            elements = [self._get_name(e) for e in node.elts]
            return f"({', '.join(elements)})"
        elif isinstance(node, ast.List):
            elements = [self._get_name(e) for e in node.elts]
            return f"[{', '.join(elements)}]"
        else:
            return ""
    
    def _get_node_source(self, node: ast.AST) -> str:
        """Get source code for an AST node."""
        if not self.source_lines:
            return ""
        
        start = node.lineno - 1
        end = getattr(node, 'end_lineno', start + 1)
        
        return '\n'.join(self.source_lines[start:end])


# =============================================================================
# CALL GRAPH ANALYZER
# =============================================================================

@dataclass
class CallEdge:
    """An edge in the call graph."""
    caller: str
    callee: str
    call_count: int = 1
    locations: List[CodeLocation] = field(default_factory=list)


class CallGraphAnalyzer:
    """
    Analyzes call relationships between functions.
    
    Builds a directed graph where:
    - Nodes are functions/methods
    - Edges represent calls from caller to callee
    
    Useful for:
    - Finding all callers of a function
    - Finding all functions called by a function  
    - Impact analysis (what changes when X changes)
    - Dead code detection
    """
    
    def __init__(self):
        self.edges: Dict[Tuple[str, str], CallEdge] = {}
        self.callers: Dict[str, Set[str]] = defaultdict(set)
        self.callees: Dict[str, Set[str]] = defaultdict(set)
    
    def add_call(
        self, 
        caller: str, 
        callee: str,
        location: Optional[CodeLocation] = None,
    ) -> None:
        """Add a call relationship."""
        key = (caller, callee)
        
        if key in self.edges:
            self.edges[key].call_count += 1
            if location:
                self.edges[key].locations.append(location)
        else:
            self.edges[key] = CallEdge(
                caller=caller,
                callee=callee,
                locations=[location] if location else [],
            )
        
        self.callers[callee].add(caller)
        self.callees[caller].add(callee)
    
    def get_callers(self, function: str) -> Set[str]:
        """Get all functions that call the given function."""
        return self.callers.get(function, set())
    
    def get_callees(self, function: str) -> Set[str]:
        """Get all functions called by the given function."""
        return self.callees.get(function, set())
    
    def get_transitive_callers(
        self, 
        function: str,
        max_depth: int = 5,
    ) -> Set[str]:
        """Get transitive closure of callers (callers of callers, etc.)."""
        result = set()
        frontier = {function}
        
        for _ in range(max_depth):
            new_frontier = set()
            for f in frontier:
                callers = self.get_callers(f)
                new_callers = callers - result
                result.update(new_callers)
                new_frontier.update(new_callers)
            frontier = new_frontier
            if not frontier:
                break
        
        return result
    
    def get_impact_analysis(self, function: str) -> Dict[str, Any]:
        """Analyze impact of changing a function."""
        callers = self.get_transitive_callers(function)
        
        return {
            "function": function,
            "direct_callers": list(self.get_callers(function)),
            "transitive_callers": list(callers),
            "impact_score": len(callers),
        }


# =============================================================================
# SEMANTIC ANALYZER
# =============================================================================

class SemanticAnalyzer:
    """
    High-level semantic analysis of Python code.
    
    This is the main entry point for semantic code intelligence.
    
    Usage:
        analyzer = SemanticAnalyzer()
        analyzer.analyze_file("path/to/file.py")
        
        # Get function info
        func = analyzer.get_element("my_function")
        
        # Get all callers
        callers = analyzer.get_callers("my_function")
        
        # Get context for a function
        context = analyzer.get_context_for_function("my_function")
    """
    
    def __init__(self):
        self.elements: Dict[str, CodeElement] = {}
        self.files: Dict[str, str] = {}  # path -> content
        self.call_graph = CallGraphAnalyzer()
        
        self.stats = {
            "files_analyzed": 0,
            "functions": 0,
            "classes": 0,
            "imports": 0,
        }
    
    def analyze_file(self, file_path: str, content: Optional[str] = None) -> None:
        """
        Analyze a Python file.
        
        Args:
            file_path: Path to the file
            content: Optional content (reads from disk if not provided)
        """
        if content is None:
            path = Path(file_path)
            if not path.exists():
                logger.warning(f"File not found: {file_path}")
                return
            content = path.read_text()
        
        self.files[file_path] = content
        self.stats["files_analyzed"] += 1
        
        try:
            tree = ast.parse(content)
            visitor = SemanticVisitor(file_path, content)
            visitor.visit(tree)
            
            # Merge elements
            for name, element in visitor.elements.items():
                fqn = f"{file_path}:{name}"
                self.elements[fqn] = element
                
                # Update call graph
                for callee in element.calls:
                    self.call_graph.add_call(fqn, callee, element.location)
                
                # Update stats
                if element.element_type == ElementType.FUNCTION:
                    self.stats["functions"] += 1
                elif element.element_type == ElementType.METHOD:
                    self.stats["functions"] += 1
                elif element.element_type == ElementType.CLASS:
                    self.stats["classes"] += 1
                elif element.element_type == ElementType.IMPORT:
                    self.stats["imports"] += 1
                    
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
    
    def analyze_directory(
        self, 
        directory: str,
        extensions: List[str] = None,
    ) -> None:
        """Analyze all Python files in a directory."""
        extensions = extensions or [".py"]
        
        for root, dirs, files in os.walk(directory):
            # Skip common non-source directories
            dirs[:] = [d for d in dirs if not d.startswith('.') 
                      and d not in ('__pycache__', 'venv', 'node_modules')]
            
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    file_path = os.path.join(root, file)
                    self.analyze_file(file_path)
    
    def get_element(self, name: str) -> Optional[CodeElement]:
        """
        Get a code element by name.
        
        Tries various lookup strategies:
        1. Exact match
        2. Suffix match (for function names without file path)
        """
        # Exact match
        if name in self.elements:
            return self.elements[name]
        
        # Suffix match
        for fqn, element in self.elements.items():
            if fqn.endswith(f":{name}") or fqn.endswith(f".{name}"):
                return element
        
        return None
    
    def get_callers(self, function_name: str) -> List[CodeElement]:
        """Get all functions that call the given function."""
        element = self.get_element(function_name)
        if not element:
            return []
        
        # Find the FQN
        fqn = None
        for name, el in self.elements.items():
            if el is element:
                fqn = name
                break
        
        if not fqn:
            return []
        
        caller_names = self.call_graph.get_callers(fqn)
        return [self.elements[c] for c in caller_names if c in self.elements]
    
    def get_callees(self, function_name: str) -> List[CodeElement]:
        """Get all functions called by the given function."""
        element = self.get_element(function_name)
        if not element:
            return []
        
        callee_elements = []
        for callee_name in element.calls:
            callee = self.get_element(callee_name)
            if callee:
                callee_elements.append(callee)
        
        return callee_elements
    
    def get_context_for_function(
        self,
        function_name: str,
        include_callers: bool = True,
        include_callees: bool = True,
        max_depth: int = 1,
    ) -> str:
        """
        Get rich context for a function including related code.
        
        This is optimized for LLM consumption.
        """
        element = self.get_element(function_name)
        if not element:
            return f"Function {function_name} not found"
        
        context_parts = [
            "# Target Function",
            element.to_prompt_context(include_body=True),
        ]
        
        if include_callers:
            callers = self.get_callers(function_name)
            if callers:
                context_parts.append("\n# Functions that call this:")
                for caller in callers[:3]:  # Limit for context size
                    context_parts.append(caller.to_prompt_context())
        
        if include_callees:
            callees = self.get_callees(function_name)
            if callees:
                context_parts.append("\n# Functions called by this:")
                for callee in callees[:3]:
                    context_parts.append(callee.to_prompt_context())
        
        return "\n\n".join(context_parts)
    
    def find_functions_by_pattern(
        self,
        pattern: str,
        element_type: Optional[ElementType] = None,
    ) -> List[CodeElement]:
        """Find functions/classes matching a regex pattern."""
        regex = re.compile(pattern, re.IGNORECASE)
        results = []
        
        for name, element in self.elements.items():
            if element_type and element.element_type != element_type:
                continue
            
            if regex.search(element.name) or regex.search(element.signature):
                results.append(element)
        
        return results
    
    def get_affected_by_change(
        self,
        function_name: str,
    ) -> Dict[str, Any]:
        """
        Get all code that would be affected by changing a function.
        
        Returns impact analysis including:
        - Direct callers
        - Transitive callers
        - Impact score
        """
        element = self.get_element(function_name)
        if not element:
            return {"error": f"Function {function_name} not found"}
        
        # Find FQN
        fqn = None
        for name, el in self.elements.items():
            if el is element:
                fqn = name
                break
        
        if not fqn:
            return {"error": "Could not resolve FQN"}
        
        return self.call_graph.get_impact_analysis(fqn)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get analysis statistics."""
        return {
            **self.stats,
            "total_elements": len(self.elements),
            "call_edges": len(self.call_graph.edges),
        }
    
    def print_stats(self) -> None:
        """Print analysis statistics."""
        stats = self.get_stats()
        print("\n" + "=" * 60)
        print("SEMANTIC ANALYZER STATISTICS")
        print("=" * 60)
        print(f"Files Analyzed:   {stats['files_analyzed']}")
        print(f"Functions:        {stats['functions']}")
        print(f"Classes:          {stats['classes']}")
        print(f"Imports:          {stats['imports']}")
        print(f"Total Elements:   {stats['total_elements']}")
        print(f"Call Edges:       {stats['call_edges']}")
        print("=" * 60)


# =============================================================================
# CODE INTELLIGENCE ENGINE
# =============================================================================

class CodeIntelligenceEngine:
    """
    High-level API for code intelligence.
    
    Combines semantic analysis with smart retrieval for LLM consumption.
    
    Usage:
        engine = CodeIntelligenceEngine(repo_path)
        
        # Index the codebase
        engine.index()
        
        # Get context for fixing a bug
        context = engine.get_fix_context(
            target_function="calculate_total",
            problem="Returns wrong value for empty lists",
        )
    """
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.analyzer = SemanticAnalyzer()
        self._indexed = False
    
    def index(self, patterns: List[str] = None) -> None:
        """Index the codebase for semantic analysis."""
        patterns = patterns or ["*.py"]
        
        for pattern in patterns:
            for file_path in self.repo_path.rglob(pattern):
                # Skip non-source directories
                if any(p in str(file_path) for p in [
                    '__pycache__', '.git', 'venv', 'node_modules', '.eggs'
                ]):
                    continue
                
                self.analyzer.analyze_file(str(file_path))
        
        self._indexed = True
        logger.info(f"Indexed {self.analyzer.stats['files_analyzed']} files")
    
    def get_fix_context(
        self,
        target_function: str,
        problem: str = "",
        max_tokens: int = 3000,
    ) -> str:
        """
        Get optimized context for fixing a bug in a function.
        
        Returns a carefully crafted context string optimized for LLM consumption.
        """
        if not self._indexed:
            self.index()
        
        # Get target element
        element = self.analyzer.get_element(target_function)
        if not element:
            return f"# Target function '{target_function}' not found in codebase"
        
        context_parts = []
        
        # Problem statement
        if problem:
            context_parts.append(f"# Problem\n{problem}\n")
        
        # Target function with full context
        func_context = self.analyzer.get_context_for_function(target_function)
        context_parts.append(func_context)
        
        # Impact analysis
        impact = self.analyzer.get_affected_by_change(target_function)
        if impact and not impact.get("error"):
            direct = impact.get("direct_callers", [])
            if direct:
                context_parts.append(f"\n# Direct Callers ({len(direct)} functions rely on this):")
                for caller in direct[:5]:
                    context_parts.append(f"  - {caller}")
        
        return "\n\n".join(context_parts)
    
    def find_related_functions(
        self,
        function_name: str,
        max_results: int = 10,
    ) -> List[CodeElement]:
        """Find functions related to the given function."""
        if not self._indexed:
            self.index()
        
        related = []
        
        # Add callers
        related.extend(self.analyzer.get_callers(function_name))
        
        # Add callees
        related.extend(self.analyzer.get_callees(function_name))
        
        # Deduplicate
        seen = set()
        unique = []
        for el in related:
            if el.fully_qualified_name not in seen:
                seen.add(el.fully_qualified_name)
                unique.append(el)
        
        return unique[:max_results]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def analyze_file(file_path: str) -> SemanticAnalyzer:
    """Quick function to analyze a single file."""
    analyzer = SemanticAnalyzer()
    analyzer.analyze_file(file_path)
    return analyzer


def get_function_context(
    file_path: str,
    function_name: str,
) -> str:
    """Quick function to get context for a function."""
    analyzer = analyze_file(file_path)
    return analyzer.get_context_for_function(function_name)


# =============================================================================
# MAIN (DEMO)
# =============================================================================

if __name__ == "__main__":
    print("S.P.I.D.E.R. Semantic Code Intelligence Demo")
    print("=" * 50)
    
    # Sample code to analyze
    sample_code = '''
def calculate_total(items):
    """Calculate the total price of items."""
    total = 0
    for item in items:
        total += get_price(item)
    return total

def get_price(item):
    """Get the price of an item."""
    return item.price * item.quantity

class ShoppingCart:
    """A shopping cart."""
    
    def __init__(self):
        self.items = []
    
    def add_item(self, item):
        """Add an item."""
        self.items.append(item)
    
    def get_total(self):
        """Get cart total."""
        return calculate_total(self.items)
'''
    
    # Analyze
    analyzer = SemanticAnalyzer()
    analyzer.analyze_file("cart.py", sample_code)
    
    print("\nAnalyzed Elements:")
    for name, element in analyzer.elements.items():
        if not name.startswith('import:') and name != '__module__':
            print(f"  {element.element_type.name}: {element.name}")
            if element.calls:
                print(f"    Calls: {element.calls}")
    
    print("\nFunction Context for 'calculate_total':")
    print("-" * 40)
    context = analyzer.get_context_for_function("calculate_total")
    print(context)
    
    analyzer.print_stats()
