"""
S.P.I.D.E.R. Knowledge Graph - Semantic Code Navigation
========================================================

Born from: Linux-1.pdf (Linux Kernel Bug Localization / LINUXFL+)

The Insight:
"The Linux Kernel is too big for a Bloom Filter alone. Successful agents
don't just look at code; they look at Documentation (.rst, .txt) and
Maintainer Hierarchies. Standard RAG fails because it retrieves 
'similar text' not 'causal logic.'"

The Solution:
Upgrade Bloom Filter to a Semantic Knowledge Graph.

Nodes: Functions, Files, Documentation sections
Edges: "Calls", "Defines", "Documented By", "Maintained By"

When S.P.I.D.E.R. touches driver.c, the graph pulls the specific
driver.rst specs into the prompt.

Result: The LLM knows the INTENT of the code, not just the syntax.
"""

import hashlib
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# GRAPH TYPES
# =============================================================================

class NodeType(Enum):
    """Types of nodes in the Knowledge Graph."""
    FILE = auto()           # Source code file
    FUNCTION = auto()       # Function/method definition
    CLASS = auto()          # Class definition
    VARIABLE = auto()       # Global variable/constant
    DOCUMENTATION = auto()  # Documentation file (.rst, .md, .txt)
    TEST = auto()           # Test file/function
    CONFIG = auto()         # Configuration file
    MAINTAINER = auto()     # Code maintainer (from MAINTAINERS file)


class EdgeType(Enum):
    """Types of edges connecting nodes."""
    CALLS = auto()          # Function A calls Function B
    DEFINES = auto()        # File A defines Function B
    IMPORTS = auto()        # File A imports from File B
    DOCUMENTED_BY = auto()  # Code A is documented by Doc B
    TESTS = auto()          # Test A tests Function B
    MAINTAINED_BY = auto()  # File A is maintained by Person B
    DEPENDS_ON = auto()     # Component A depends on Component B
    RELATED_TO = auto()     # Loose semantic relationship
    PARENT_OF = auto()      # Directory contains File


# =============================================================================
# GRAPH NODES AND EDGES
# =============================================================================

@dataclass
class GraphNode:
    """A node in the Knowledge Graph."""
    id: str                              # Unique identifier
    name: str                            # Human-readable name
    node_type: NodeType                  # Type of node
    path: str = ""                       # File path if applicable
    content_hash: str = ""               # Hash of content for change detection
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Position for functions/classes
    start_line: int = 0
    end_line: int = 0
    
    # Extracted content (docstring, signature, etc.)
    signature: str = ""
    docstring: str = ""
    keywords: List[str] = field(default_factory=list)
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return isinstance(other, GraphNode) and self.id == other.id


@dataclass
class GraphEdge:
    """An edge connecting two nodes."""
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0                  # Strength of relationship
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash((self.source_id, self.target_id, self.edge_type))


# =============================================================================
# KNOWLEDGE GRAPH
# =============================================================================

class KnowledgeGraph:
    """
    Semantic Knowledge Graph for codebase navigation.
    
    Unlike a simple Bloom Filter (which only tells you "symbol exists"),
    the Knowledge Graph knows:
    - WHO calls WHOM
    - WHERE is the documentation
    - WHAT tests cover this code
    - WHO maintains this component
    
    Usage:
        kg = KnowledgeGraph()
        
        # Add nodes
        kg.add_node("driver.c", NodeType.FILE)
        kg.add_node("init_driver", NodeType.FUNCTION)
        kg.add_node("driver.rst", NodeType.DOCUMENTATION)
        
        # Add edges
        kg.add_edge("driver.c", "init_driver", EdgeType.DEFINES)
        kg.add_edge("init_driver", "driver.rst", EdgeType.DOCUMENTED_BY)
        
        # Query
        docs = kg.get_documentation_for("init_driver")
        callers = kg.get_callers("init_driver")
    """
    
    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        
        # Indices for fast lookup
        self._edges_by_source: Dict[str, List[GraphEdge]] = defaultdict(list)
        self._edges_by_target: Dict[str, List[GraphEdge]] = defaultdict(list)
        self._nodes_by_type: Dict[NodeType, Set[str]] = defaultdict(set)
        self._nodes_by_path: Dict[str, str] = {}
        
        # Keyword index (replaces Bloom Filter functionality)
        self._keyword_index: Dict[str, Set[str]] = defaultdict(set)
        
        self._stats = {
            "nodes": 0,
            "edges": 0,
            "queries": 0,
        }
    
    # -------------------------------------------------------------------------
    # NODE OPERATIONS
    # -------------------------------------------------------------------------
    
    def add_node(
        self,
        node_id: str,
        node_type: NodeType,
        name: str = None,
        path: str = "",
        **kwargs,
    ) -> GraphNode:
        """Add a node to the graph."""
        if node_id in self.nodes:
            # Update existing
            node = self.nodes[node_id]
            for k, v in kwargs.items():
                if hasattr(node, k):
                    setattr(node, k, v)
            return node
        
        node = GraphNode(
            id=node_id,
            name=name or node_id,
            node_type=node_type,
            path=path,
            **kwargs,
        )
        
        self.nodes[node_id] = node
        self._nodes_by_type[node_type].add(node_id)
        
        if path:
            self._nodes_by_path[path] = node_id
        
        # Index keywords
        for keyword in node.keywords:
            self._keyword_index[keyword.lower()].add(node_id)
        
        # Index name parts
        for part in self._split_identifier(node.name):
            self._keyword_index[part.lower()].add(node_id)
        
        self._stats["nodes"] = len(self.nodes)
        return node
    
    def get_node(self, node_id: str) -> Optional[GraphNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def get_nodes_by_type(self, node_type: NodeType) -> List[GraphNode]:
        """Get all nodes of a specific type."""
        return [self.nodes[nid] for nid in self._nodes_by_type.get(node_type, set())]
    
    def get_node_by_path(self, path: str) -> Optional[GraphNode]:
        """Get node by file path."""
        node_id = self._nodes_by_path.get(path)
        return self.nodes.get(node_id) if node_id else None
    
    # -------------------------------------------------------------------------
    # EDGE OPERATIONS
    # -------------------------------------------------------------------------
    
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        weight: float = 1.0,
        **metadata,
    ) -> GraphEdge:
        """Add an edge between two nodes."""
        edge = GraphEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=weight,
            metadata=metadata,
        )
        
        self.edges.append(edge)
        self._edges_by_source[source_id].append(edge)
        self._edges_by_target[target_id].append(edge)
        
        self._stats["edges"] = len(self.edges)
        return edge
    
    def get_edges_from(self, node_id: str, edge_type: EdgeType = None) -> List[GraphEdge]:
        """Get edges originating from a node."""
        edges = self._edges_by_source.get(node_id, [])
        if edge_type:
            edges = [e for e in edges if e.edge_type == edge_type]
        return edges
    
    def get_edges_to(self, node_id: str, edge_type: EdgeType = None) -> List[GraphEdge]:
        """Get edges pointing to a node."""
        edges = self._edges_by_target.get(node_id, [])
        if edge_type:
            edges = [e for e in edges if e.edge_type == edge_type]
        return edges
    
    # -------------------------------------------------------------------------
    # SEMANTIC QUERIES
    # -------------------------------------------------------------------------
    
    def search(self, query: str, limit: int = 10) -> List[Tuple[GraphNode, float]]:
        """
        Search for nodes matching a query.
        
        Uses keyword index for fast lookup.
        
        Args:
            query: Search query (space-separated keywords)
            limit: Maximum results
            
        Returns:
            List of (node, score) tuples sorted by relevance
        """
        self._stats["queries"] += 1
        
        keywords = self._split_identifier(query)
        candidates: Dict[str, float] = defaultdict(float)
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            # Exact match
            for node_id in self._keyword_index.get(keyword_lower, set()):
                candidates[node_id] += 2.0
            
            # Partial match
            for indexed_keyword, node_ids in self._keyword_index.items():
                if keyword_lower in indexed_keyword or indexed_keyword in keyword_lower:
                    for node_id in node_ids:
                        candidates[node_id] += 0.5
        
        # Sort by score
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for node_id, score in sorted_candidates[:limit]:
            node = self.nodes.get(node_id)
            if node:
                results.append((node, score))
        
        return results
    
    def get_documentation_for(self, node_id: str) -> List[GraphNode]:
        """Get documentation nodes for a code node."""
        self._stats["queries"] += 1
        
        doc_edges = self.get_edges_from(node_id, EdgeType.DOCUMENTED_BY)
        docs = []
        
        for edge in doc_edges:
            doc_node = self.get_node(edge.target_id)
            if doc_node:
                docs.append(doc_node)
        
        # Also check if the node is documented by edges pointing TO it
        doc_edges_to = self.get_edges_to(node_id, EdgeType.DOCUMENTED_BY)
        for edge in doc_edges_to:
            doc_node = self.get_node(edge.source_id)
            if doc_node and doc_node.node_type == NodeType.DOCUMENTATION:
                docs.append(doc_node)
        
        return docs
    
    def get_callers(self, function_id: str) -> List[GraphNode]:
        """Get functions that call this function."""
        self._stats["queries"] += 1
        
        call_edges = self.get_edges_to(function_id, EdgeType.CALLS)
        callers = []
        
        for edge in call_edges:
            caller = self.get_node(edge.source_id)
            if caller:
                callers.append(caller)
        
        return callers
    
    def get_callees(self, function_id: str) -> List[GraphNode]:
        """Get functions called by this function."""
        self._stats["queries"] += 1
        
        call_edges = self.get_edges_from(function_id, EdgeType.CALLS)
        callees = []
        
        for edge in call_edges:
            callee = self.get_node(edge.target_id)
            if callee:
                callees.append(callee)
        
        return callees
    
    def get_tests_for(self, node_id: str) -> List[GraphNode]:
        """Get test nodes that test this code."""
        self._stats["queries"] += 1
        
        test_edges = self.get_edges_to(node_id, EdgeType.TESTS)
        tests = []
        
        for edge in test_edges:
            test = self.get_node(edge.source_id)
            if test:
                tests.append(test)
        
        return tests
    
    def get_related(self, node_id: str, max_depth: int = 2) -> List[Tuple[GraphNode, int]]:
        """
        Get related nodes within a certain graph distance.
        
        Args:
            node_id: Starting node
            max_depth: Maximum edges to traverse
            
        Returns:
            List of (node, distance) tuples
        """
        self._stats["queries"] += 1
        
        visited = {node_id: 0}
        queue = [(node_id, 0)]
        results = []
        
        while queue:
            current_id, depth = queue.pop(0)
            
            if depth > max_depth:
                continue
            
            if depth > 0:
                node = self.get_node(current_id)
                if node:
                    results.append((node, depth))
            
            # Traverse edges
            for edge in self.get_edges_from(current_id):
                if edge.target_id not in visited:
                    visited[edge.target_id] = depth + 1
                    queue.append((edge.target_id, depth + 1))
            
            for edge in self.get_edges_to(current_id):
                if edge.source_id not in visited:
                    visited[edge.source_id] = depth + 1
                    queue.append((edge.source_id, depth + 1))
        
        return results
    
    def get_context_for_file(self, file_path: str) -> Dict[str, Any]:
        """
        Get rich context for a file to include in LLM prompt.
        
        This is the key method - replaces simple Bloom Filter lookup
        with semantic context gathering.
        
        Returns:
            {
                "file": GraphNode,
                "functions": [GraphNode],
                "documentation": [GraphNode],
                "tests": [GraphNode],
                "dependencies": [GraphNode],
                "callers": [GraphNode],
            }
        """
        self._stats["queries"] += 1
        
        context = {
            "file": None,
            "functions": [],
            "documentation": [],
            "tests": [],
            "dependencies": [],
            "callers": [],
        }
        
        # Get file node
        file_node = self.get_node_by_path(file_path)
        if not file_node:
            # Try by name
            file_name = Path(file_path).name
            results = self.search(file_name, limit=1)
            if results:
                file_node = results[0][0]
        
        if not file_node:
            return context
        
        context["file"] = file_node
        
        # Get defined functions
        for edge in self.get_edges_from(file_node.id, EdgeType.DEFINES):
            func = self.get_node(edge.target_id)
            if func and func.node_type == NodeType.FUNCTION:
                context["functions"].append(func)
        
        # Get documentation
        context["documentation"] = self.get_documentation_for(file_node.id)
        
        # Get tests
        context["tests"] = self.get_tests_for(file_node.id)
        
        # Get dependencies (imports)
        for edge in self.get_edges_from(file_node.id, EdgeType.IMPORTS):
            dep = self.get_node(edge.target_id)
            if dep:
                context["dependencies"].append(dep)
        
        # Get callers (files that import this)
        for edge in self.get_edges_to(file_node.id, EdgeType.IMPORTS):
            caller = self.get_node(edge.source_id)
            if caller:
                context["callers"].append(caller)
        
        return context
    
    # -------------------------------------------------------------------------
    # UTILITY METHODS
    # -------------------------------------------------------------------------
    
    def _split_identifier(self, identifier: str) -> List[str]:
        """Split identifier into component words."""
        # Handle camelCase and PascalCase
        words = re.sub(r'([a-z])([A-Z])', r'\1_\2', identifier)
        # Handle snake_case
        words = words.replace('-', '_').lower()
        parts = [w for w in words.split('_') if w and len(w) > 1]
        return parts
    
    def contains(self, keyword: str) -> bool:
        """Check if keyword exists (Bloom Filter replacement)."""
        return keyword.lower() in self._keyword_index
    
    def check(self, symbol: str) -> bool:
        """Alias for contains (Bloom Filter compatibility)."""
        return self.contains(symbol)
    
    # -------------------------------------------------------------------------
    # SERIALIZATION
    # -------------------------------------------------------------------------
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize graph to dictionary."""
        return {
            "nodes": [
                {
                    "id": n.id,
                    "name": n.name,
                    "type": n.node_type.name,
                    "path": n.path,
                    "signature": n.signature,
                    "docstring": n.docstring[:500] if n.docstring else "",
                    "keywords": n.keywords[:20],
                }
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "source": e.source_id,
                    "target": e.target_id,
                    "type": e.edge_type.name,
                    "weight": e.weight,
                }
                for e in self.edges
            ],
        }
    
    def save(self, path: str) -> None:
        """Save graph to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "KnowledgeGraph":
        """Load graph from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        kg = cls()
        
        for node_data in data.get("nodes", []):
            kg.add_node(
                node_id=node_data["id"],
                node_type=NodeType[node_data["type"]],
                name=node_data.get("name"),
                path=node_data.get("path", ""),
                signature=node_data.get("signature", ""),
                docstring=node_data.get("docstring", ""),
                keywords=node_data.get("keywords", []),
            )
        
        for edge_data in data.get("edges", []):
            kg.add_edge(
                source_id=edge_data["source"],
                target_id=edge_data["target"],
                edge_type=EdgeType[edge_data["type"]],
                weight=edge_data.get("weight", 1.0),
            )
        
        return kg
    
    def get_stats(self) -> Dict[str, int]:
        return {
            **self._stats,
            "keyword_index_size": len(self._keyword_index),
        }


# =============================================================================
# GRAPH BUILDER
# =============================================================================

class KnowledgeGraphBuilder:
    """
    Builds a Knowledge Graph from a codebase.
    
    Parses Python files to extract:
    - Function definitions
    - Class definitions
    - Import relationships
    - Documentation links
    """
    
    def __init__(self, kg: Optional[KnowledgeGraph] = None):
        self.kg = kg or KnowledgeGraph()
    
    def build_from_directory(self, root_path: str, extensions: List[str] = None) -> KnowledgeGraph:
        """
        Build graph from a directory.
        
        Args:
            root_path: Root directory to scan
            extensions: File extensions to include (default: py, rst, md)
        """
        extensions = extensions or [".py", ".rst", ".md", ".txt"]
        root = Path(root_path)
        
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in extensions:
                self._process_file(str(path))
        
        return self.kg
    
    def _process_file(self, file_path: str) -> None:
        """Process a single file."""
        path = Path(file_path)
        
        # Determine node type
        if path.suffix == ".py":
            node_type = NodeType.FILE
            if "test" in path.name.lower():
                node_type = NodeType.TEST
        elif path.suffix in (".rst", ".md", ".txt"):
            node_type = NodeType.DOCUMENTATION
        else:
            node_type = NodeType.FILE
        
        # Create file node
        file_node = self.kg.add_node(
            node_id=str(path),
            node_type=node_type,
            name=path.name,
            path=str(path),
            keywords=[path.stem],
        )
        
        # Parse Python files for functions/classes
        if path.suffix == ".py":
            self._parse_python_file(file_path, file_node)
        
        # Link documentation to code
        if node_type == NodeType.DOCUMENTATION:
            self._link_documentation(file_node)
    
    def _parse_python_file(self, file_path: str, file_node: GraphNode) -> None:
        """Parse Python file for structure."""
        import ast
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_id = f"{file_path}::{node.name}"
                    func_node = self.kg.add_node(
                        node_id=func_id,
                        node_type=NodeType.FUNCTION,
                        name=node.name,
                        path=file_path,
                        start_line=node.lineno,
                        end_line=node.end_lineno or node.lineno,
                        docstring=ast.get_docstring(node) or "",
                        signature=f"def {node.name}(...)",
                    )
                    
                    # File defines function
                    self.kg.add_edge(file_node.id, func_id, EdgeType.DEFINES)
                    
                    # Analyze calls
                    for child in ast.walk(node):
                        if isinstance(child, ast.Call):
                            if isinstance(child.func, ast.Name):
                                callee = child.func.id
                                self.kg.add_edge(func_id, callee, EdgeType.CALLS)
                
                elif isinstance(node, ast.ClassDef):
                    class_id = f"{file_path}::{node.name}"
                    self.kg.add_node(
                        node_id=class_id,
                        node_type=NodeType.CLASS,
                        name=node.name,
                        path=file_path,
                        start_line=node.lineno,
                        end_line=node.end_lineno or node.lineno,
                        docstring=ast.get_docstring(node) or "",
                    )
                    
                    self.kg.add_edge(file_node.id, class_id, EdgeType.DEFINES)
                
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        self.kg.add_edge(file_node.id, alias.name, EdgeType.IMPORTS)
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        self.kg.add_edge(file_node.id, node.module, EdgeType.IMPORTS)
                        
        except (SyntaxError, UnicodeDecodeError) as e:
            logger.debug(f"Could not parse {file_path}: {e}")
    
    def _link_documentation(self, doc_node: GraphNode) -> None:
        """Link documentation to related code files."""
        # Simple heuristic: match by name
        doc_stem = Path(doc_node.path).stem.lower()
        
        for node_id, node in self.kg.nodes.items():
            if node.node_type in (NodeType.FILE, NodeType.FUNCTION, NodeType.CLASS):
                if doc_stem in node.name.lower() or node.name.lower() in doc_stem:
                    self.kg.add_edge(node.id, doc_node.id, EdgeType.DOCUMENTED_BY)


# =============================================================================
# PACKAGE INIT
# =============================================================================

__all__ = [
    "KnowledgeGraph",
    "KnowledgeGraphBuilder",
    "GraphNode",
    "GraphEdge",
    "NodeType",
    "EdgeType",
]


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🧠 S.P.I.D.E.R. Knowledge Graph - Demo")
    print("=" * 70)
    
    kg = KnowledgeGraph()
    
    # Build example graph
    kg.add_node("driver.c", NodeType.FILE, path="/kernel/drivers/driver.c")
    kg.add_node("driver.rst", NodeType.DOCUMENTATION, path="/docs/driver.rst")
    kg.add_node("init_driver", NodeType.FUNCTION, signature="void init_driver(struct device *dev)")
    kg.add_node("test_driver.py", NodeType.TEST, path="/tests/test_driver.py")
    kg.add_node("cleanup_driver", NodeType.FUNCTION)
    
    kg.add_edge("driver.c", "init_driver", EdgeType.DEFINES)
    kg.add_edge("driver.c", "cleanup_driver", EdgeType.DEFINES)
    kg.add_edge("init_driver", "driver.rst", EdgeType.DOCUMENTED_BY)
    kg.add_edge("test_driver.py", "init_driver", EdgeType.TESTS)
    kg.add_edge("init_driver", "cleanup_driver", EdgeType.CALLS)
    
    # Queries
    print("\n📋 Search for 'driver':")
    for node, score in kg.search("driver"):
        print(f"  {node.name} ({node.node_type.name}) - score: {score:.1f}")
    
    print("\n📚 Documentation for 'init_driver':")
    for doc in kg.get_documentation_for("init_driver"):
        print(f"  {doc.name}")
    
    print("\n🧪 Tests for 'init_driver':")
    for test in kg.get_tests_for("init_driver"):
        print(f"  {test.name}")
    
    print("\n📞 Callees of 'init_driver':")
    for callee in kg.get_callees("init_driver"):
        print(f"  {callee.name}")
    
    print("\n🔗 Related to 'init_driver' (2 hops):")
    for node, dist in kg.get_related("init_driver", max_depth=2):
        print(f"  {node.name} (distance: {dist})")
    
    print(f"\n📊 Stats: {kg.get_stats()}")
    print("=" * 70)
