"""
S.P.I.D.E.R. Task Projector - Dynamic Task Embedding Generation
================================================================

From HYPERFORMER++ Paper (Hypernetworks-1):
"A Task Embedding (I_τ) is a compressed vector representation of a 
specific task that captures its essential characteristics."

The Projector analyzes:
1. Code AST structure (function signatures, class hierarchies)
2. Bloom Filter statistics (what patterns are present)
3. Semantic keywords (mutex, css, sql, async, etc.)
4. Complexity metrics (cyclomatic, nesting depth)

Output: A Task Fingerprint that enables:
- Zero-shot configuration prediction
- Similarity matching to historical solutions
- Domain-specific optimization paths

This is the FIRST layer in the Chameleon Engine pipeline.
"""

import hashlib
import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple
import ast

logger = logging.getLogger(__name__)


# =============================================================================
# TASK DOMAINS
# =============================================================================

class TaskDomain(Enum):
    """
    High-level task domain classification.
    
    Each domain has different optimization characteristics:
    - FRONTEND: High creativity, low precision (CSS/HTML)
    - BACKEND: Balanced (APIs, business logic)
    - DATABASE: High precision, schema-aware (SQL, ORM)
    - SYSTEMS: Maximum precision, safety-critical (threading, memory)
    - ML_DATA: Numerical precision, vectorized ops
    - TESTING: Coverage-oriented
    - UNKNOWN: Default fallback
    """
    FRONTEND = auto()      # HTML, CSS, JavaScript, UI
    BACKEND = auto()       # APIs, routing, business logic
    DATABASE = auto()      # SQL, ORM, migrations
    SYSTEMS = auto()       # Threading, memory, kernel
    ML_DATA = auto()       # NumPy, Pandas, ML models
    TESTING = auto()       # Test frameworks, assertions
    DEVOPS = auto()        # Docker, CI/CD, config
    UNKNOWN = auto()       # Fallback


# Domain keyword mappings
DOMAIN_KEYWORDS = {
    TaskDomain.FRONTEND: {
        "css", "html", "color", "padding", "margin", "display", "flex",
        "grid", "button", "div", "span", "style", "className", "onClick",
        "render", "component", "props", "useState", "useEffect", "dom",
        "querySelector", "getElementById", "animation", "transition",
    },
    TaskDomain.BACKEND: {
        "request", "response", "endpoint", "route", "api", "http", "get",
        "post", "put", "delete", "json", "serialize", "deserialize",
        "middleware", "handler", "controller", "service", "view",
    },
    TaskDomain.DATABASE: {
        "sql", "query", "select", "insert", "update", "delete", "join",
        "foreign_key", "primary_key", "index", "migration", "orm",
        "model", "CharField", "IntegerField", "ForeignKey", "objects",
        "filter", "annotate", "aggregate", "transaction", "cursor",
    },
    TaskDomain.SYSTEMS: {
        "thread", "lock", "mutex", "semaphore", "deadlock", "race",
        "condition", "barrier", "atomic", "volatile", "memory", "malloc",
        "free", "pointer", "buffer", "overflow", "kernel", "syscall",
        "signal", "interrupt", "process", "fork", "exec", "pipe",
    },
    TaskDomain.ML_DATA: {
        "numpy", "pandas", "dataframe", "series", "array", "tensor",
        "model", "train", "predict", "fit", "transform", "sklearn",
        "tensorflow", "pytorch", "epoch", "batch", "gradient", "loss",
        "optimizer", "layer", "neural", "feature", "label", "accuracy",
    },
    TaskDomain.TESTING: {
        "test", "assert", "mock", "patch", "fixture", "pytest", "unittest",
        "setup", "teardown", "expect", "should", "describe", "it",
        "coverage", "parametrize", "skip", "xfail", "marker",
    },
    TaskDomain.DEVOPS: {
        "docker", "container", "kubernetes", "k8s", "pod", "deployment",
        "service", "ingress", "helm", "terraform", "ansible", "ci", "cd",
        "pipeline", "github", "gitlab", "jenkins", "yaml", "config",
    },
}


# =============================================================================
# TASK FINGERPRINT
# =============================================================================

@dataclass
class TaskFingerprint:
    """
    A mathematical fingerprint of a coding task.
    
    This is the Task Embedding (I_τ) from HYPERFORMER++.
    It encodes the essential characteristics of a task in a format
    suitable for:
    1. Similarity computation (find related past solutions)
    2. Configuration prediction (what hyperparams will work)
    3. Domain routing (which specialized solver to use)
    """
    
    # Domain classification
    primary_domain: TaskDomain
    domain_scores: Dict[TaskDomain, float] = field(default_factory=dict)
    
    # Complexity metrics (0.0 to 1.0 normalized)
    complexity_score: float = 0.5         # Overall complexity
    cyclomatic_complexity: float = 0.0    # Branching complexity
    nesting_depth: float = 0.0            # Maximum nesting
    code_size: float = 0.0                # Lines of code
    
    # Risk assessment
    risk_level: float = 0.5               # 0=safe, 1=dangerous
    mutation_risk: float = 0.0            # How much state changes
    side_effect_risk: float = 0.0         # External effects
    
    # Pattern signatures
    keyword_vector: Dict[str, float] = field(default_factory=dict)
    ast_signature: str = ""               # Hash of AST structure
    bloom_stats: Dict[str, int] = field(default_factory=dict)
    
    # Semantic features
    has_async: bool = False
    has_threading: bool = False
    has_io: bool = False
    has_network: bool = False
    has_database: bool = False
    
    # Embedding vector (for similarity)
    embedding: List[float] = field(default_factory=list)
    
    def to_vector(self) -> List[float]:
        """Convert fingerprint to numerical vector for ML."""
        if self.embedding:
            return self.embedding
        
        # Build feature vector
        features = [
            # Domain (one-hot encoded)
            *[1.0 if d == self.primary_domain else 0.0 for d in TaskDomain],
            # Complexity
            self.complexity_score,
            self.cyclomatic_complexity,
            self.nesting_depth,
            self.code_size,
            # Risk
            self.risk_level,
            self.mutation_risk,
            self.side_effect_risk,
            # Boolean features
            float(self.has_async),
            float(self.has_threading),
            float(self.has_io),
            float(self.has_network),
            float(self.has_database),
        ]
        
        self.embedding = features
        return features
    
    def similarity(self, other: "TaskFingerprint") -> float:
        """Compute cosine similarity with another fingerprint."""
        v1 = self.to_vector()
        v2 = other.to_vector()
        
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "primary_domain": self.primary_domain.name,
            "domain_scores": {k.name: v for k, v in self.domain_scores.items()},
            "complexity_score": self.complexity_score,
            "risk_level": self.risk_level,
            "has_async": self.has_async,
            "has_threading": self.has_threading,
            "has_io": self.has_io,
            "has_network": self.has_network,
            "has_database": self.has_database,
        }


# =============================================================================
# AST ANALYZER
# =============================================================================

class ASTAnalyzer(ast.NodeVisitor):
    """
    Extracts structural features from Python AST.
    
    Analyzes:
    - Function/class counts
    - Control flow complexity
    - Import patterns
    - Call patterns
    """
    
    def __init__(self):
        self.functions: List[str] = []
        self.classes: List[str] = []
        self.imports: Set[str] = set()
        self.calls: List[str] = []
        self.max_depth = 0
        self.current_depth = 0
        self.branch_count = 0
        self.loop_count = 0
        self.try_count = 0
        
    def analyze(self, code: str) -> Dict[str, Any]:
        """Analyze code and return metrics."""
        try:
            tree = ast.parse(code)
            self.visit(tree)
            
            return {
                "functions": len(self.functions),
                "classes": len(self.classes),
                "imports": list(self.imports),
                "unique_calls": len(set(self.calls)),
                "max_nesting": self.max_depth,
                "branches": self.branch_count,
                "loops": self.loop_count,
                "try_blocks": self.try_count,
            }
        except SyntaxError:
            return {
                "functions": 0,
                "classes": 0,
                "imports": [],
                "unique_calls": 0,
                "max_nesting": 0,
                "branches": 0,
                "loops": 0,
                "try_blocks": 0,
            }
    
    def visit_FunctionDef(self, node):
        self.functions.append(node.name)
        self._track_depth(node)
    
    def visit_AsyncFunctionDef(self, node):
        self.functions.append(node.name)
        self._track_depth(node)
    
    def visit_ClassDef(self, node):
        self.classes.append(node.name)
        self._track_depth(node)
    
    def visit_Import(self, node):
        for alias in node.names:
            self.imports.add(alias.name.split('.')[0])
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        if node.module:
            self.imports.add(node.module.split('.')[0])
        self.generic_visit(node)
    
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            self.calls.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            self.calls.append(node.func.attr)
        self.generic_visit(node)
    
    def visit_If(self, node):
        self.branch_count += 1
        self._track_depth(node)
    
    def visit_For(self, node):
        self.loop_count += 1
        self._track_depth(node)
    
    def visit_While(self, node):
        self.loop_count += 1
        self._track_depth(node)
    
    def visit_Try(self, node):
        self.try_count += 1
        self._track_depth(node)
    
    def _track_depth(self, node):
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)
        self.generic_visit(node)
        self.current_depth -= 1


# =============================================================================
# TASK PROJECTOR
# =============================================================================

class TaskProjector:
    """
    The Task Projector - Core of the Chameleon Engine.
    
    From HYPERFORMER++:
    "The Projector Network takes a task description and produces
    a low-dimensional Task Embedding I_τ that captures the essence
    of the task for downstream weight generation."
    
    Our implementation uses:
    1. Keyword analysis (domain detection)
    2. AST structure (complexity metrics)
    3. Pattern matching (risk assessment)
    4. Historical similarity (experience matching)
    
    Usage:
        projector = TaskProjector()
        fingerprint = projector.project(
            problem_statement="Fix the CSS padding bug...",
            code_context="def render_button(): ...",
        )
        print(f"Domain: {fingerprint.primary_domain}")
        print(f"Risk: {fingerprint.risk_level}")
    """
    
    def __init__(self, bloom_filter=None):
        """
        Initialize the Task Projector.
        
        Args:
            bloom_filter: Optional Bloom filter for pattern detection
        """
        self.bloom_filter = bloom_filter
        self.ast_analyzer = ASTAnalyzer()
        
        # Historical embeddings for similarity matching
        self.embedding_cache: Dict[str, TaskFingerprint] = {}
        
        # Risk keywords
        self.high_risk_keywords = {
            "delete", "drop", "truncate", "rm", "remove", "kill",
            "force", "override", "unsafe", "raw", "exec", "eval",
            "sudo", "root", "admin", "password", "secret", "key",
        }
        
        self.mutation_keywords = {
            "update", "modify", "change", "set", "write", "save",
            "insert", "append", "push", "pop", "del", "clear",
        }
        
        self.stats = {
            "projections": 0,
            "cache_hits": 0,
        }
    
    def project(
        self,
        problem_statement: str,
        code_context: str = "",
        file_path: str = "",
        additional_context: Dict[str, Any] = None,
    ) -> TaskFingerprint:
        """
        Project a task into its fingerprint embedding.
        
        Args:
            problem_statement: The bug report or task description
            code_context: Related code snippets
            file_path: Path to the file being modified
            additional_context: Extra metadata
            
        Returns:
            TaskFingerprint with domain classification and metrics
        """
        self.stats["projections"] += 1
        
        # Combine all text for analysis
        full_text = f"{problem_statement} {code_context} {file_path}"
        
        # Cache check
        cache_key = hashlib.md5(full_text.encode()).hexdigest()[:16]
        if cache_key in self.embedding_cache:
            self.stats["cache_hits"] += 1
            return self.embedding_cache[cache_key]
        
        # Extract features
        keywords = self._extract_keywords(full_text)
        domain_scores = self._score_domains(keywords)
        primary_domain = max(domain_scores, key=domain_scores.get)
        
        # Analyze code structure
        ast_metrics = self._analyze_code(code_context)
        
        # Calculate complexity
        complexity = self._calculate_complexity(ast_metrics, len(full_text))
        
        # Calculate risk
        risk = self._calculate_risk(keywords, primary_domain)
        
        # Detect semantic features
        semantic = self._detect_semantic_features(keywords, full_text)
        
        # Build fingerprint
        fingerprint = TaskFingerprint(
            primary_domain=primary_domain,
            domain_scores=domain_scores,
            complexity_score=complexity["overall"],
            cyclomatic_complexity=complexity["cyclomatic"],
            nesting_depth=complexity["nesting"],
            code_size=complexity["size"],
            risk_level=risk["overall"],
            mutation_risk=risk["mutation"],
            side_effect_risk=risk["side_effect"],
            keyword_vector=dict(keywords),
            ast_signature=self._generate_ast_signature(code_context),
            has_async=semantic["async"],
            has_threading=semantic["threading"],
            has_io=semantic["io"],
            has_network=semantic["network"],
            has_database=semantic["database"],
        )
        
        # Compute and store embedding
        fingerprint.to_vector()
        
        # Cache
        self.embedding_cache[cache_key] = fingerprint
        
        return fingerprint
    
    def _extract_keywords(self, text: str) -> Counter:
        """Extract and count relevant keywords."""
        # Normalize text
        text_lower = text.lower()
        
        # Extract words
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text_lower)
        
        # Filter to relevant keywords
        all_keywords = set()
        for domain_keywords in DOMAIN_KEYWORDS.values():
            all_keywords.update(domain_keywords)
        all_keywords.update(self.high_risk_keywords)
        all_keywords.update(self.mutation_keywords)
        
        relevant = [w for w in words if w in all_keywords]
        return Counter(relevant)
    
    def _score_domains(self, keywords: Counter) -> Dict[TaskDomain, float]:
        """Score each domain based on keyword presence."""
        scores = {}
        
        for domain, domain_keywords in DOMAIN_KEYWORDS.items():
            score = sum(keywords.get(kw, 0) for kw in domain_keywords)
            # Normalize by domain keyword count
            scores[domain] = score / len(domain_keywords) if domain_keywords else 0
        
        # Ensure we have at least UNKNOWN
        if not any(scores.values()):
            scores[TaskDomain.UNKNOWN] = 1.0
        
        # Normalize to sum to 1
        total = sum(scores.values())
        if total > 0:
            for d in scores:
                scores[d] /= total
        
        return scores
    
    def _analyze_code(self, code: str) -> Dict[str, Any]:
        """Analyze code structure using AST."""
        if not code.strip():
            return {
                "functions": 0,
                "classes": 0,
                "imports": [],
                "unique_calls": 0,
                "max_nesting": 0,
                "branches": 0,
                "loops": 0,
                "try_blocks": 0,
            }
        
        analyzer = ASTAnalyzer()
        return analyzer.analyze(code)
    
    def _calculate_complexity(
        self,
        ast_metrics: Dict[str, Any],
        text_length: int,
    ) -> Dict[str, float]:
        """Calculate normalized complexity metrics."""
        # Cyclomatic complexity (normalized)
        branches = ast_metrics.get("branches", 0) + ast_metrics.get("loops", 0)
        cyclomatic = min(1.0, branches / 20)  # Cap at 20 branches
        
        # Nesting depth (normalized)
        nesting = min(1.0, ast_metrics.get("max_nesting", 0) / 10)  # Cap at 10
        
        # Code size (normalized by log scale)
        size = min(1.0, math.log10(max(1, text_length)) / 5)  # Cap at 100k chars
        
        # Overall complexity (weighted average)
        overall = (cyclomatic * 0.4 + nesting * 0.3 + size * 0.3)
        
        return {
            "overall": overall,
            "cyclomatic": cyclomatic,
            "nesting": nesting,
            "size": size,
        }
    
    def _calculate_risk(
        self,
        keywords: Counter,
        domain: TaskDomain,
    ) -> Dict[str, float]:
        """Calculate risk levels for the task."""
        # High-risk keyword count
        risk_count = sum(keywords.get(kw, 0) for kw in self.high_risk_keywords)
        risk_level = min(1.0, risk_count / 5)  # Cap at 5 risk keywords
        
        # Mutation risk
        mutation_count = sum(keywords.get(kw, 0) for kw in self.mutation_keywords)
        mutation_risk = min(1.0, mutation_count / 10)
        
        # Domain-based risk adjustment
        domain_risk_multiplier = {
            TaskDomain.SYSTEMS: 1.5,      # Threading is risky
            TaskDomain.DATABASE: 1.3,     # Data mutations are risky
            TaskDomain.FRONTEND: 0.5,     # CSS bugs are low risk
            TaskDomain.TESTING: 0.3,      # Tests are safe
        }
        multiplier = domain_risk_multiplier.get(domain, 1.0)
        
        # Side effect risk (I/O, network, etc.)
        side_effect = 0.0
        io_keywords = {"file", "open", "write", "read", "path", "directory"}
        net_keywords = {"http", "request", "socket", "connect", "url"}
        
        if any(kw in keywords for kw in io_keywords):
            side_effect += 0.3
        if any(kw in keywords for kw in net_keywords):
            side_effect += 0.3
        
        return {
            "overall": min(1.0, risk_level * multiplier),
            "mutation": mutation_risk,
            "side_effect": min(1.0, side_effect),
        }
    
    def _detect_semantic_features(
        self,
        keywords: Counter,
        text: str,
    ) -> Dict[str, bool]:
        """Detect high-level semantic features."""
        text_lower = text.lower()
        
        return {
            "async": any(kw in keywords or kw in text_lower 
                        for kw in ["async", "await", "asyncio", "coroutine"]),
            "threading": any(kw in keywords or kw in text_lower 
                            for kw in ["thread", "lock", "mutex", "semaphore", "threading"]),
            "io": any(kw in keywords or kw in text_lower 
                     for kw in ["file", "open", "read", "write", "path", "os."]),
            "network": any(kw in keywords or kw in text_lower 
                          for kw in ["http", "request", "socket", "url", "api"]),
            "database": any(kw in keywords or kw in text_lower 
                           for kw in ["sql", "query", "database", "db", "orm", "model"]),
        }
    
    def _generate_ast_signature(self, code: str) -> str:
        """Generate a hash signature of the AST structure."""
        if not code.strip():
            return ""
        
        try:
            tree = ast.parse(code)
            # Create a simplified structure hash
            structure = []
            for node in ast.walk(tree):
                structure.append(type(node).__name__)
            
            signature = hashlib.md5("".join(structure).encode()).hexdigest()[:16]
            return signature
        except SyntaxError:
            return ""
    
    def find_similar(
        self,
        fingerprint: TaskFingerprint,
        threshold: float = 0.7,
    ) -> List[Tuple[str, TaskFingerprint, float]]:
        """
        Find similar tasks from history.
        
        Args:
            fingerprint: The current task fingerprint
            threshold: Minimum similarity score
            
        Returns:
            List of (cache_key, fingerprint, similarity) tuples
        """
        similar = []
        
        for key, cached in self.embedding_cache.items():
            sim = fingerprint.similarity(cached)
            if sim >= threshold:
                similar.append((key, cached, sim))
        
        # Sort by similarity descending
        similar.sort(key=lambda x: x[2], reverse=True)
        return similar
    
    def get_stats(self) -> Dict[str, Any]:
        """Get projector statistics."""
        return {
            **self.stats,
            "cache_size": len(self.embedding_cache),
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_project(problem: str, code: str = "") -> TaskFingerprint:
    """Quick projection for simple use cases."""
    projector = TaskProjector()
    return projector.project(problem, code)


def detect_domain(text: str) -> TaskDomain:
    """Quickly detect the primary domain of a task."""
    projector = TaskProjector()
    fp = projector.project(text)
    return fp.primary_domain


# =============================================================================
# MAIN (DEMO)
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🦎 S.P.I.D.E.R. CHAMELEON - Task Projector Demo")
    print("=" * 70)
    
    projector = TaskProjector()
    
    # Test cases
    test_cases = [
        {
            "name": "CSS Bug",
            "problem": "The button padding is 2px off in Safari. The color doesn't match the design spec.",
            "code": "def render_button(style): return f'<button style=\"{style}\">'",
        },
        {
            "name": "Race Condition",
            "problem": "Multiple threads accessing shared counter causes incorrect values. Need mutex protection.",
            "code": "import threading\ndef increment():\n    global counter\n    counter += 1",
        },
        {
            "name": "SQL Injection",
            "problem": "User input is not sanitized in the query, allowing SQL injection attacks.",
            "code": "cursor.execute(f'SELECT * FROM users WHERE id = {user_id}')",
        },
        {
            "name": "API Endpoint",
            "problem": "The /api/users endpoint returns 500 when the database is empty.",
            "code": "def get_users():\n    return json.dumps(db.query('SELECT * FROM users'))",
        },
    ]
    
    for case in test_cases:
        print(f"\n{'─' * 60}")
        print(f"📋 Task: {case['name']}")
        print(f"{'─' * 60}")
        
        fp = projector.project(case["problem"], case["code"])
        
        print(f"  Domain:     {fp.primary_domain.name}")
        print(f"  Complexity: {fp.complexity_score:.2f}")
        print(f"  Risk Level: {fp.risk_level:.2f}")
        print(f"  Threading:  {fp.has_threading}")
        print(f"  Database:   {fp.has_database}")
        print(f"  Network:    {fp.has_network}")
        
        # Show top domain scores
        sorted_domains = sorted(fp.domain_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"  Top Domains: {', '.join(f'{d.name}:{s:.2f}' for d, s in sorted_domains)}")
    
    print(f"\n{'=' * 70}")
    print(f"Stats: {projector.get_stats()}")
    print(f"{'=' * 70}")
