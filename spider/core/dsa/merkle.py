"""
Semantic Merkle Tree for Distributed Code Verification.

This module provides a content-addressable verification system for Python codebases
that is immune to cosmetic changes like comments, docstrings, and whitespace.
Only logical changes to the code will alter the Merkle root hash.
"""

import ast
import hashlib
import os
from dataclasses import dataclass, field
from typing import List, Optional


class SemanticHasher:
    """
    Produces deterministic SHA-256 hashes of Python source code based on its
    semantic content, ignoring docstrings, comments, and type hints.
    """

    def __init__(self, strip_type_hints: bool = True):
        """
        Initialize the SemanticHasher.

        Args:
            strip_type_hints: If True, type annotations are removed before hashing.
        """
        self.strip_type_hints = strip_type_hints

    def hash(self, source_code: str) -> str:
        """
        Parse Python source code and return a SHA-256 hash of its normalized AST.

        Args:
            source_code: A string containing valid Python source code.

        Returns:
            A hexadecimal SHA-256 hash string.

        Raises:
            SyntaxError: If the source code is not valid Python.
        """
        tree = ast.parse(source_code)
        normalized_tree = self._normalize(tree)
        ast_dump = ast.dump(normalized_tree, annotate_fields=True, include_attributes=False)
        return hashlib.sha256(ast_dump.encode('utf-8')).hexdigest()

    def _normalize(self, node: ast.AST) -> ast.AST:
        """
        Recursively normalize an AST node by stripping docstrings and type hints.

        Args:
            node: An AST node to normalize.

        Returns:
            A normalized copy of the AST node.
        """
        transformer = _SemanticNormalizer(strip_type_hints=self.strip_type_hints)
        return transformer.visit(node)


class _SemanticNormalizer(ast.NodeTransformer):
    """
    AST NodeTransformer that removes docstrings, type hints, and normalizes
    the tree for semantic comparison.
    """

    def __init__(self, strip_type_hints: bool = True):
        super().__init__()
        self.strip_type_hints = strip_type_hints

    def visit_Module(self, node: ast.Module) -> ast.Module:
        """Remove module-level docstrings and process children."""
        node.body = self._strip_docstring(node.body)
        node.type_ignores = []  # Remove type: ignore comments
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        """Remove class docstrings and optionally strip decorators' type info."""
        node.body = self._strip_docstring(node.body)
        if self.strip_type_hints:
            node.decorator_list = [self.visit(d) for d in node.decorator_list]
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Remove function docstrings and type annotations."""
        node.body = self._strip_docstring(node.body)
        if self.strip_type_hints:
            node.returns = None
            node.args = self._strip_arg_annotations(node.args)
        node.decorator_list = [self.visit(d) for d in node.decorator_list]
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
        """Remove async function docstrings and type annotations."""
        node.body = self._strip_docstring(node.body)
        if self.strip_type_hints:
            node.returns = None
            node.args = self._strip_arg_annotations(node.args)
        node.decorator_list = [self.visit(d) for d in node.decorator_list]
        self.generic_visit(node)
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Optional[ast.AST]:
        """
        Handle annotated assignments. Convert `x: int = 5` to `x = 5`.
        Remove standalone annotations like `x: int` entirely.
        """
        if self.strip_type_hints:
            if node.value is not None:
                # Convert annotated assignment to regular assignment
                return ast.Assign(
                    targets=[node.target],
                    value=self.visit(node.value),
                    lineno=node.lineno,
                    col_offset=node.col_offset
                )
            else:
                # Standalone annotation, remove entirely
                return None
        self.generic_visit(node)
        return node

    def _strip_docstring(self, body: List[ast.stmt]) -> List[ast.stmt]:
        """
        Remove leading docstring from a body of statements.

        Args:
            body: A list of AST statement nodes.

        Returns:
            The body with the leading docstring removed, if present.
        """
        if not body:
            return body

        first_stmt = body[0]
        if isinstance(first_stmt, ast.Expr) and isinstance(first_stmt.value, (ast.Constant, ast.Str)):
            # Check if it's a string constant (docstring)
            value = first_stmt.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                return body[1:]
            elif isinstance(value, ast.Str):  # Python < 3.8 compatibility
                return body[1:]
        return body

    def _strip_arg_annotations(self, args: ast.arguments) -> ast.arguments:
        """
        Remove all type annotations from function arguments.

        Args:
            args: An ast.arguments node.

        Returns:
            The arguments node with all annotations set to None.
        """
        for arg in args.args:
            arg.annotation = None
        for arg in args.posonlyargs:
            arg.annotation = None
        for arg in args.kwonlyargs:
            arg.annotation = None
        if args.vararg:
            args.vararg.annotation = None
        if args.kwarg:
            args.kwarg.annotation = None
        return args


@dataclass
class MerkleNode:
    """
    A node in the Semantic Merkle Tree.

    Attributes:
        hash: The SHA-256 hash of this node's content.
        children: List of child MerkleNodes.
        type: The type of code element ('module', 'file', 'class', 'function', 'directory').
        name: The name of the code element.
    """
    hash: str
    type: str
    name: str
    children: List['MerkleNode'] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"MerkleNode(type={self.type!r}, name={self.name!r}, hash={self.hash[:12]}..., children={len(self.children)})"

    def to_dict(self) -> dict:
        """Convert the node and its children to a dictionary representation."""
        return {
            'hash': self.hash,
            'type': self.type,
            'name': self.name,
            'children': [child.to_dict() for child in self.children]
        }


class CodebaseMerkleTree:
    """
    Constructs a Merkle Tree from a Python codebase directory.

    The tree structure is:
        - Root Node: The project root directory.
        - Intermediate Nodes: Subdirectories and Python files.
        - Leaf Nodes: Individual functions and classes within files.
    """

    def __init__(self, root_path: str, strip_type_hints: bool = True):
        """
        Initialize the CodebaseMerkleTree.

        Args:
            root_path: Absolute or relative path to the root directory.
            strip_type_hints: If True, type annotations are ignored in hashing.
        """
        self.root_path = os.path.abspath(root_path)
        self.hasher = SemanticHasher(strip_type_hints=strip_type_hints)
        self._root_node: Optional[MerkleNode] = None

    def build(self) -> MerkleNode:
        """
        Build the Merkle Tree from the root directory.

        Returns:
            The root MerkleNode of the constructed tree.

        Raises:
            FileNotFoundError: If the root path does not exist.
            NotADirectoryError: If the root path is not a directory.
        """
        if not os.path.exists(self.root_path):
            raise FileNotFoundError(f"Path does not exist: {self.root_path}")
        if not os.path.isdir(self.root_path):
            raise NotADirectoryError(f"Path is not a directory: {self.root_path}")

        self._root_node = self._build_directory_node(self.root_path)
        return self._root_node

    @property
    def root_hash(self) -> Optional[str]:
        """Return the root hash of the codebase, or None if not built."""
        return self._root_node.hash if self._root_node else None

    @property
    def root_node(self) -> Optional[MerkleNode]:
        """Return the root node of the tree, or None if not built."""
        return self._root_node

    def _build_directory_node(self, dir_path: str) -> MerkleNode:
        """
        Recursively build a MerkleNode for a directory.

        Args:
            dir_path: Path to the directory.

        Returns:
            A MerkleNode representing the directory.
        """
        children: List[MerkleNode] = []
        dir_name = os.path.basename(dir_path) or dir_path

        try:
            entries = sorted(os.listdir(dir_path))
        except PermissionError:
            # Return empty node for inaccessible directories
            empty_hash = hashlib.sha256(b'').hexdigest()
            return MerkleNode(hash=empty_hash, type='directory', name=dir_name, children=[])

        for entry in entries:
            entry_path = os.path.join(dir_path, entry)

            # Skip hidden files/directories and __pycache__
            if entry.startswith('.') or entry == '__pycache__':
                continue

            if os.path.isdir(entry_path):
                child_node = self._build_directory_node(entry_path)
                children.append(child_node)
            elif entry.endswith('.py'):
                child_node = self._build_file_node(entry_path)
                if child_node:
                    children.append(child_node)

        # Compute directory hash from sorted child hashes
        combined = ''.join(sorted(child.hash for child in children))
        dir_hash = hashlib.sha256(combined.encode('utf-8')).hexdigest()

        return MerkleNode(hash=dir_hash, type='directory', name=dir_name, children=children)

    def _build_file_node(self, file_path: str) -> Optional[MerkleNode]:
        """
        Build a MerkleNode for a Python file.

        Args:
            file_path: Path to the Python file.

        Returns:
            A MerkleNode representing the file, or None if parsing fails.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
        except (IOError, UnicodeDecodeError):
            return None

        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            # Return a hash of the raw content for unparseable files
            raw_hash = hashlib.sha256(source_code.encode('utf-8')).hexdigest()
            return MerkleNode(
                hash=raw_hash,
                type='file',
                name=os.path.basename(file_path),
                children=[]
            )

        children: List[MerkleNode] = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef):
                child_node = self._build_function_node(node, source_code)
                children.append(child_node)
            elif isinstance(node, ast.AsyncFunctionDef):
                child_node = self._build_async_function_node(node, source_code)
                children.append(child_node)
            elif isinstance(node, ast.ClassDef):
                child_node = self._build_class_node(node, source_code)
                children.append(child_node)

        # If no functions or classes, hash the entire module
        if not children:
            file_hash = self.hasher.hash(source_code)
        else:
            # Combine child hashes in sorted order for determinism
            combined = ''.join(sorted(child.hash for child in children))
            file_hash = hashlib.sha256(combined.encode('utf-8')).hexdigest()

        return MerkleNode(
            hash=file_hash,
            type='file',
            name=os.path.basename(file_path),
            children=children
        )

    def _build_function_node(self, node: ast.FunctionDef, source_code: str) -> MerkleNode:
        """
        Build a MerkleNode for a function definition.

        Args:
            node: The AST FunctionDef node.
            source_code: The full source code of the file.

        Returns:
            A MerkleNode representing the function.
        """
        func_source = ast.get_source_segment(source_code, node)
        if func_source:
            func_hash = self.hasher.hash(func_source)
        else:
            # Fallback: hash the AST dump directly
            func_hash = hashlib.sha256(ast.dump(node).encode('utf-8')).hexdigest()

        return MerkleNode(hash=func_hash, type='function', name=node.name, children=[])

    def _build_async_function_node(self, node: ast.AsyncFunctionDef, source_code: str) -> MerkleNode:
        """
        Build a MerkleNode for an async function definition.

        Args:
            node: The AST AsyncFunctionDef node.
            source_code: The full source code of the file.

        Returns:
            A MerkleNode representing the async function.
        """
        func_source = ast.get_source_segment(source_code, node)
        if func_source:
            func_hash = self.hasher.hash(func_source)
        else:
            func_hash = hashlib.sha256(ast.dump(node).encode('utf-8')).hexdigest()

        return MerkleNode(hash=func_hash, type='function', name=node.name, children=[])

    def _build_class_node(self, node: ast.ClassDef, source_code: str) -> MerkleNode:
        """
        Build a MerkleNode for a class definition.

        Args:
            node: The AST ClassDef node.
            source_code: The full source code of the file.

        Returns:
            A MerkleNode representing the class with its methods as children.
        """
        children: List[MerkleNode] = []

        # Extract methods as child nodes
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_source = ast.get_source_segment(source_code, item)
                if method_source:
                    method_hash = self.hasher.hash(method_source)
                else:
                    method_hash = hashlib.sha256(ast.dump(item).encode('utf-8')).hexdigest()
                children.append(MerkleNode(hash=method_hash, type='method', name=item.name, children=[]))
            elif isinstance(item, ast.AsyncFunctionDef):
                method_source = ast.get_source_segment(source_code, item)
                if method_source:
                    method_hash = self.hasher.hash(method_source)
                else:
                    method_hash = hashlib.sha256(ast.dump(item).encode('utf-8')).hexdigest()
                children.append(MerkleNode(hash=method_hash, type='method', name=item.name, children=[]))

        # Compute class hash from the entire class source
        class_source = ast.get_source_segment(source_code, node)
        if class_source:
            class_hash = self.hasher.hash(class_source)
        else:
            class_hash = hashlib.sha256(ast.dump(node).encode('utf-8')).hexdigest()

        return MerkleNode(hash=class_hash, type='class', name=node.name, children=children)

    def diff(self, other: 'CodebaseMerkleTree') -> List[dict]:
        """
        Compare this tree with another and return the differences.

        Args:
            other: Another CodebaseMerkleTree to compare against.

        Returns:
            A list of dictionaries describing the differences.
        """
        if not self._root_node or not other._root_node:
            raise ValueError("Both trees must be built before comparing.")

        differences = []
        self._diff_nodes(self._root_node, other._root_node, differences, path="")
        return differences

    def _diff_nodes(
        self,
        node_a: Optional[MerkleNode],
        node_b: Optional[MerkleNode],
        differences: List[dict],
        path: str
    ) -> None:
        """Recursively compare two nodes and collect differences."""
        current_path = f"{path}/{node_a.name}" if node_a else f"{path}/{node_b.name}" if node_b else path

        if node_a is None and node_b is not None:
            differences.append({'type': 'added', 'path': current_path, 'node': node_b.name})
            return
        if node_b is None and node_a is not None:
            differences.append({'type': 'removed', 'path': current_path, 'node': node_a.name})
            return
        if node_a is None and node_b is None:
            return

        if node_a.hash != node_b.hash:
            differences.append({
                'type': 'modified',
                'path': current_path,
                'node': node_a.name,
                'old_hash': node_a.hash,
                'new_hash': node_b.hash
            })

            # Build lookup maps for children
            children_a = {c.name: c for c in node_a.children}
            children_b = {c.name: c for c in node_b.children}

            all_names = set(children_a.keys()) | set(children_b.keys())
            for name in sorted(all_names):
                self._diff_nodes(
                    children_a.get(name),
                    children_b.get(name),
                    differences,
                    current_path
                )

    def print_tree(self, node: Optional[MerkleNode] = None, indent: int = 0) -> None:
        """
        Print a visual representation of the Merkle Tree.

        Args:
            node: The node to start printing from. Defaults to root.
            indent: Current indentation level.
        """
        if node is None:
            node = self._root_node
        if node is None:
            print("Tree not built. Call build() first.")
            return

        prefix = "  " * indent
        hash_preview = node.hash[:12]
        print(f"{prefix}[{node.type}] {node.name} ({hash_preview}...)")

        for child in node.children:
            self.print_tree(child, indent + 1)


# Convenience function for quick hashing
def hash_python_code(source_code: str, strip_type_hints: bool = True) -> str:
    """
    Convenience function to hash Python source code.

    Args:
        source_code: Python source code as a string.
        strip_type_hints: If True, type annotations are ignored.

    Returns:
        A SHA-256 hash of the semantic content.
    """
    hasher = SemanticHasher(strip_type_hints=strip_type_hints)
    return hasher.hash(source_code)


def build_codebase_tree(root_path: str, strip_type_hints: bool = True) -> CodebaseMerkleTree:
    """
    Convenience function to build a Merkle Tree from a codebase.

    Args:
        root_path: Path to the root directory.
        strip_type_hints: If True, type annotations are ignored.

    Returns:
        A built CodebaseMerkleTree instance.
    """
    tree = CodebaseMerkleTree(root_path, strip_type_hints=strip_type_hints)
    tree.build()
    return tree


if __name__ == '__main__':
    # Demo usage
    import sys

    if len(sys.argv) > 1:
        target_path = sys.argv[1]
    else:
        target_path = os.getcwd()

    print(f"Building Semantic Merkle Tree for: {target_path}\n")

    tree = CodebaseMerkleTree(target_path)
    tree.build()

    print("Tree Structure:")
    print("-" * 60)
    tree.print_tree()
    print("-" * 60)
    print(f"\nRoot Hash: {tree.root_hash}")

    # Demo: Show that cosmetic changes don't affect the hash
    print("\n--- Semantic Hash Demo ---")
    code_v1 = '''
def hello(name: str) -> str:
    """Say hello to someone."""
    # This is a greeting function
    return f"Hello, {name}!"
'''

    code_v2 = '''
def hello(name):
    return f"Hello, {name}!"
'''

    hasher = SemanticHasher()
    hash_v1 = hasher.hash(code_v1)
    hash_v2 = hasher.hash(code_v2)

    print(f"Code v1 (with docstring, comments, type hints): {hash_v1[:16]}...")
    print(f"Code v2 (stripped down):                        {hash_v2[:16]}...")
    print(f"Hashes match: {hash_v1 == hash_v2}")
