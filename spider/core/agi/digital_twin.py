"""
S.P.I.D.E.R. Digital Twin - Neuro-Symbolic World Model
=======================================================

Born from: AGI-1 (Attention) + AGI-5 (Sparks of AGI)

The Scientific Finding:
"AGI-1 gave us Self-Attention to relate distant concepts. AGI-5 proved
that at scale, models develop an internal representation of the world
(a 'World Model')—understanding that 'Circle' in Python is geometrically
related to 'Circle' in SVG, even with different syntax."

The Missing Link:
Current Agents treat a codebase as TEXT. An AGI treats it as TOPOLOGY.

The Solution:
Build a persistent, graph-based World Model that runs alongside code:

1. Global Attention Map: Compute attention scores between files
2. Cross-Domain Generalization: Map "Business Intent" to "Implementation"
3. The Oracle: Query "If I change this API, what breaks conceptually?"

Result: The Agent stops making "Local Fixes" that cause "Global Regressions."
It reasons about the SYSTEM, not the FILE.
"""

import hashlib
import logging
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# WORLD MODEL TYPES
# =============================================================================

class EntityType(Enum):
    """Types of entities in the world model."""
    FILE = auto()
    CLASS = auto()
    FUNCTION = auto()
    VARIABLE = auto()
    CONCEPT = auto()          # Abstract business concept
    API_ENDPOINT = auto()
    DATABASE_TABLE = auto()
    UI_COMPONENT = auto()
    TEST = auto()
    CONFIGURATION = auto()


class RelationType(Enum):
    """Types of relationships between entities."""
    IMPORTS = auto()          # A imports B
    CALLS = auto()            # A calls B
    INHERITS = auto()         # A inherits from B
    IMPLEMENTS = auto()       # A implements concept B
    TESTS = auto()            # A tests B
    CONFIGURES = auto()       # A configures B
    DEPENDS_ON = auto()       # A depends on B
    RELATED_TO = auto()       # Semantic relationship
    AFFECTS = auto()          # Change in A affects B


@dataclass
class Entity:
    """An entity in the world model."""
    entity_id: str
    name: str
    entity_type: EntityType
    file_path: str = ""
    line_range: Tuple[int, int] = (0, 0)
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Computed properties
    importance_score: float = 0.0
    stability_score: float = 1.0      # How often it changes
    centrality_score: float = 0.0     # How connected it is


@dataclass
class Relationship:
    """A relationship between two entities."""
    source_id: str
    target_id: str
    relation_type: RelationType
    weight: float = 1.0
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImpactAnalysis:
    """Result of impact analysis."""
    changed_entity: Entity
    affected_entities: List[Tuple[Entity, float]]  # (entity, impact_score)
    risk_level: str                    # LOW, MEDIUM, HIGH, CRITICAL
    warnings: List[str]
    recommendations: List[str]


@dataclass
class ConceptMapping:
    """Mapping between business concept and implementation."""
    concept: str
    implementations: List[Entity]
    coverage_score: float             # How well implemented
    test_coverage: float              # How well tested


# =============================================================================
# ATTENTION CALCULATOR
# =============================================================================

class AttentionCalculator:
    """
    Calculates attention weights between entities.
    
    Uses a simplified self-attention mechanism to determine
    how strongly entities are related.
    """
    
    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.embeddings: Dict[str, List[float]] = {}
    
    def compute_embedding(self, text: str) -> List[float]:
        """
        Compute a simple embedding for text.
        
        In production, this would use a real embedding model.
        Here we use character-level features.
        """
        # Simple character-based embedding
        embedding = [0.0] * self.embedding_dim
        
        for i, char in enumerate(text):
            idx = hash(char) % self.embedding_dim
            embedding[idx] += 1.0
        
        # Normalize
        total = sum(embedding) or 1.0
        embedding = [x / total for x in embedding]
        
        return embedding
    
    def compute_attention(
        self,
        query: str,
        keys: List[str],
    ) -> List[float]:
        """
        Compute attention weights from query to keys.
        
        Returns list of attention weights (sum to 1).
        """
        if not keys:
            return []
        
        query_emb = self.compute_embedding(query)
        
        scores = []
        for key in keys:
            key_emb = self.compute_embedding(key)
            # Dot product attention
            score = sum(q * k for q, k in zip(query_emb, key_emb))
            scores.append(score)
        
        # Softmax
        max_score = max(scores) if scores else 0
        exp_scores = [2.718 ** (s - max_score) for s in scores]
        total = sum(exp_scores) or 1.0
        
        return [s / total for s in exp_scores]
    
    def get_top_related(
        self,
        query: str,
        candidates: Dict[str, Any],
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Get top-k related entities by attention."""
        keys = list(candidates.keys())
        weights = self.compute_attention(query, keys)
        
        ranked = sorted(zip(keys, weights), key=lambda x: -x[1])
        return ranked[:top_k]


# =============================================================================
# WORLD GRAPH
# =============================================================================

class WorldGraph:
    """
    Graph representation of the codebase world model.
    
    Stores entities and relationships with efficient querying.
    """
    
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.relationships: List[Relationship] = []
        
        # Indexes
        self.by_type: Dict[EntityType, Set[str]] = defaultdict(set)
        self.by_file: Dict[str, Set[str]] = defaultdict(set)
        self.outgoing: Dict[str, List[Relationship]] = defaultdict(list)
        self.incoming: Dict[str, List[Relationship]] = defaultdict(list)
    
    def add_entity(self, entity: Entity) -> None:
        """Add an entity to the graph."""
        self.entities[entity.entity_id] = entity
        self.by_type[entity.entity_type].add(entity.entity_id)
        if entity.file_path:
            self.by_file[entity.file_path].add(entity.entity_id)
    
    def add_relationship(self, rel: Relationship) -> None:
        """Add a relationship to the graph."""
        self.relationships.append(rel)
        self.outgoing[rel.source_id].append(rel)
        self.incoming[rel.target_id].append(rel)
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self.entities.get(entity_id)
    
    def get_entities_by_type(self, entity_type: EntityType) -> List[Entity]:
        return [self.entities[eid] for eid in self.by_type.get(entity_type, [])]
    
    def get_entities_in_file(self, file_path: str) -> List[Entity]:
        return [self.entities[eid] for eid in self.by_file.get(file_path, [])]
    
    def get_related(
        self,
        entity_id: str,
        direction: str = "both",
    ) -> List[Tuple[Entity, Relationship]]:
        """Get entities related to this one."""
        results = []
        
        if direction in ("out", "both"):
            for rel in self.outgoing.get(entity_id, []):
                if rel.target_id in self.entities:
                    results.append((self.entities[rel.target_id], rel))
        
        if direction in ("in", "both"):
            for rel in self.incoming.get(entity_id, []):
                if rel.source_id in self.entities:
                    results.append((self.entities[rel.source_id], rel))
        
        return results
    
    def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5,
    ) -> Optional[List[str]]:
        """Find shortest path between two entities."""
        if source_id not in self.entities or target_id not in self.entities:
            return None
        
        # BFS
        from collections import deque
        queue = deque([(source_id, [source_id])])
        visited = {source_id}
        
        while queue:
            current, path = queue.popleft()
            
            if current == target_id:
                return path
            
            if len(path) >= max_depth:
                continue
            
            for rel in self.outgoing.get(current, []):
                if rel.target_id not in visited:
                    visited.add(rel.target_id)
                    queue.append((rel.target_id, path + [rel.target_id]))
        
        return None
    
    def compute_centrality(self) -> None:
        """Compute centrality scores for all entities."""
        for entity in self.entities.values():
            in_degree = len(self.incoming.get(entity.entity_id, []))
            out_degree = len(self.outgoing.get(entity.entity_id, []))
            entity.centrality_score = (in_degree + out_degree) / max(len(self.entities), 1)


# =============================================================================
# DIGITAL TWIN
# =============================================================================

class DigitalTwin:
    """
    The Neuro-Symbolic World Model for S.P.I.D.E.R.
    
    Maintains a live representation of the codebase as a knowledge graph:
    1. Tracks entities (files, functions, classes, concepts)
    2. Tracks relationships (imports, calls, tests)
    3. Computes attention weights for impact analysis
    4. Answers "What breaks if I change X?"
    
    From AGI-1 + AGI-5:
    "An AGI treats a codebase as TOPOLOGY, not TEXT."
    
    Usage:
        twin = DigitalTwin()
        
        # Build world model
        twin.scan_codebase("/path/to/repo")
        
        # Query the oracle
        impact = twin.analyze_impact("auth.py")
        # Returns: login.html, user_service.py, test_auth.py
        
        # Map concepts to code
        mapping = twin.map_concept("User Authentication")
        # Returns: [auth.py, models/user.py, middleware/auth.py]
    """
    
    def __init__(self, llm_callback: Optional[Callable[[str], str]] = None):
        self.graph = WorldGraph()
        self.attention = AttentionCalculator()
        self.llm_callback = llm_callback
        
        self.concept_map: Dict[str, ConceptMapping] = {}
        
        self._stats = {
            "entities_tracked": 0,
            "relationships_tracked": 0,
            "impact_analyses": 0,
            "concept_mappings": 0,
        }
    
    def scan_codebase(self, root_path: str, extensions: List[str] = None) -> None:
        """
        Scan a codebase and build the world model.
        
        Args:
            root_path: Root directory of the codebase
            extensions: File extensions to include
        """
        extensions = extensions or [".py", ".js", ".ts", ".java", ".go"]
        root = Path(root_path)
        
        for file_path in root.rglob("*"):
            if file_path.is_file() and file_path.suffix in extensions:
                self._scan_file(file_path)
        
        # Compute relationships
        self._discover_relationships()
        
        # Compute centrality
        self.graph.compute_centrality()
        
        self._stats["entities_tracked"] = len(self.graph.entities)
        self._stats["relationships_tracked"] = len(self.graph.relationships)
    
    def _scan_file(self, file_path: Path) -> None:
        """Scan a single file and extract entities."""
        # Create file entity
        file_id = hashlib.md5(str(file_path).encode()).hexdigest()[:12]
        
        file_entity = Entity(
            entity_id=file_id,
            name=file_path.name,
            entity_type=EntityType.FILE,
            file_path=str(file_path),
        )
        self.graph.add_entity(file_entity)
        
        # Parse for Python files
        if file_path.suffix == ".py":
            self._parse_python_file(file_path, file_id)
    
    def _parse_python_file(self, file_path: Path, file_id: str) -> None:
        """Parse a Python file to extract entities."""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
        except:
            return
        
        import ast
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                entity = Entity(
                    entity_id=f"{file_id}_{node.name}",
                    name=node.name,
                    entity_type=EntityType.CLASS,
                    file_path=str(file_path),
                    line_range=(node.lineno, node.end_lineno or node.lineno),
                )
                self.graph.add_entity(entity)
                
                # Relationship: file contains class
                self.graph.add_relationship(Relationship(
                    source_id=file_id,
                    target_id=entity.entity_id,
                    relation_type=RelationType.DEPENDS_ON,
                ))
            
            elif isinstance(node, ast.FunctionDef):
                entity = Entity(
                    entity_id=f"{file_id}_{node.name}",
                    name=node.name,
                    entity_type=EntityType.FUNCTION,
                    file_path=str(file_path),
                    line_range=(node.lineno, node.end_lineno or node.lineno),
                )
                self.graph.add_entity(entity)
    
    def _discover_relationships(self) -> None:
        """Discover relationships between entities."""
        # Analyze imports
        for entity in self.graph.get_entities_by_type(EntityType.FILE):
            file_path = Path(entity.file_path)
            if not file_path.exists():
                continue
            
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
            except:
                continue
            
            # Find imports
            import_pattern = r'^(?:from\s+(\S+)\s+)?import\s+(.+)$'
            for match in re.finditer(import_pattern, content, re.MULTILINE):
                module = match.group(1) or match.group(2).split(',')[0].strip()
                
                # Find target entity
                for other in self.graph.get_entities_by_type(EntityType.FILE):
                    if module.replace('.', '/') in other.file_path:
                        self.graph.add_relationship(Relationship(
                            source_id=entity.entity_id,
                            target_id=other.entity_id,
                            relation_type=RelationType.IMPORTS,
                        ))
                        break
    
    def add_entity(
        self,
        name: str,
        entity_type: EntityType,
        file_path: str = "",
        description: str = "",
    ) -> Entity:
        """Manually add an entity to the world model."""
        entity_id = hashlib.md5(f"{name}{file_path}{time.time()}".encode()).hexdigest()[:12]
        
        entity = Entity(
            entity_id=entity_id,
            name=name,
            entity_type=entity_type,
            file_path=file_path,
            description=description,
        )
        
        self.graph.add_entity(entity)
        self._stats["entities_tracked"] += 1
        
        return entity
    
    def add_relationship(
        self,
        source_name: str,
        target_name: str,
        relation_type: RelationType,
    ) -> bool:
        """Add a relationship between named entities."""
        # Find entities by name
        source = None
        target = None
        
        for entity in self.graph.entities.values():
            if entity.name == source_name:
                source = entity
            if entity.name == target_name:
                target = entity
        
        if source and target:
            self.graph.add_relationship(Relationship(
                source_id=source.entity_id,
                target_id=target.entity_id,
                relation_type=relation_type,
            ))
            self._stats["relationships_tracked"] += 1
            return True
        
        return False
    
    def analyze_impact(self, changed_file: str) -> ImpactAnalysis:
        """
        Analyze the impact of changing a file.
        
        Args:
            changed_file: Path or name of changed file
            
        Returns:
            ImpactAnalysis with affected entities
        """
        self._stats["impact_analyses"] += 1
        
        # Find the changed entity
        changed_entity = None
        for entity in self.graph.entities.values():
            if changed_file in entity.file_path or changed_file == entity.name:
                changed_entity = entity
                break
        
        if not changed_entity:
            return ImpactAnalysis(
                changed_entity=Entity("", changed_file, EntityType.FILE),
                affected_entities=[],
                risk_level="UNKNOWN",
                warnings=["Entity not found in world model"],
                recommendations=["Run scan_codebase to update the model"],
            )
        
        # Find affected entities using graph traversal
        affected = []
        visited = set()
        
        def traverse(entity_id: str, depth: int, impact: float):
            if entity_id in visited or depth > 5:
                return
            visited.add(entity_id)
            
            for rel_entity, rel in self.graph.get_related(entity_id, "in"):
                rel_impact = impact * rel.weight * 0.7  # Decay factor
                if rel_impact > 0.1:
                    affected.append((rel_entity, rel_impact))
                    traverse(rel_entity.entity_id, depth + 1, rel_impact)
        
        traverse(changed_entity.entity_id, 0, 1.0)
        
        # Sort by impact
        affected.sort(key=lambda x: -x[1])
        
        # Determine risk level
        if not affected:
            risk_level = "LOW"
        elif len(affected) > 10 or any(a.entity_type == EntityType.API_ENDPOINT for a, _ in affected):
            risk_level = "HIGH"
        elif changed_entity.centrality_score > 0.5:
            risk_level = "CRITICAL"
        else:
            risk_level = "MEDIUM"
        
        # Generate warnings and recommendations
        warnings = []
        recommendations = []
        
        test_affected = [a for a, _ in affected if a.entity_type == EntityType.TEST]
        if test_affected:
            warnings.append(f"{len(test_affected)} test files may be affected")
            recommendations.append("Run affected tests before committing")
        
        if changed_entity.centrality_score > 0.3:
            warnings.append(f"High centrality score ({changed_entity.centrality_score:.2f}) - core module")
            recommendations.append("Consider incremental changes with thorough review")
        
        return ImpactAnalysis(
            changed_entity=changed_entity,
            affected_entities=affected[:20],  # Top 20
            risk_level=risk_level,
            warnings=warnings,
            recommendations=recommendations,
        )
    
    def map_concept(self, concept: str) -> ConceptMapping:
        """
        Map a business concept to implementation entities.
        
        Args:
            concept: Business concept description
            
        Returns:
            ConceptMapping with implementing entities
        """
        self._stats["concept_mappings"] += 1
        
        # Use attention to find related entities
        entity_names = {e.name: e for e in self.graph.entities.values()}
        
        top_related = self.attention.get_top_related(
            concept.lower(),
            {name.lower(): e for name, e in entity_names.items()},
            top_k=10,
        )
        
        implementations = []
        for name_lower, score in top_related:
            # Find original entity
            for name, entity in entity_names.items():
                if name.lower() == name_lower:
                    implementations.append(entity)
                    break
        
        # Calculate coverage
        coverage = min(1.0, len(implementations) / 5)  # Assume 5 is good coverage
        
        # Check test coverage
        test_entities = [e for e in implementations if e.entity_type == EntityType.TEST]
        test_coverage = len(test_entities) / max(len(implementations), 1)
        
        mapping = ConceptMapping(
            concept=concept,
            implementations=implementations,
            coverage_score=coverage,
            test_coverage=test_coverage,
        )
        
        self.concept_map[concept] = mapping
        
        return mapping
    
    def query_oracle(self, question: str) -> str:
        """
        Ask the oracle a question about the codebase.
        
        Args:
            question: Natural language question
            
        Returns:
            Answer based on world model
        """
        question_lower = question.lower()
        
        # Pattern matching for common queries
        if "what breaks" in question_lower or "impact" in question_lower:
            # Extract entity name
            words = question.split()
            for word in words:
                if word in [e.name for e in self.graph.entities.values()]:
                    impact = self.analyze_impact(word)
                    affected_names = [e.name for e, _ in impact.affected_entities[:5]]
                    return f"Changing {word} affects: {', '.join(affected_names)}. Risk: {impact.risk_level}"
        
        if "where is" in question_lower or "find" in question_lower:
            # Search entities
            for entity in self.graph.entities.values():
                if entity.name.lower() in question_lower:
                    return f"'{entity.name}' is a {entity.entity_type.name} in {entity.file_path}"
        
        if "how many" in question_lower:
            # Count queries
            if "file" in question_lower:
                count = len(self.graph.by_type.get(EntityType.FILE, []))
                return f"There are {count} files in the model"
            if "class" in question_lower:
                count = len(self.graph.by_type.get(EntityType.CLASS, []))
                return f"There are {count} classes in the model"
        
        return "I cannot answer that question with the current world model."
    
    def get_stats(self) -> Dict[str, int]:
        return {
            **self._stats,
            "entity_types": len(self.graph.by_type),
            "files_tracked": len(self.graph.by_file),
        }
    
    def print_status(self) -> None:
        """Print world model status."""
        print("\n" + "=" * 60)
        print("[*] DIGITAL TWIN STATUS")
        print("=" * 60)
        
        print(f"\n[%] World Model:")
        print(f"   Entities: {len(self.graph.entities)}")
        print(f"   Relationships: {len(self.graph.relationships)}")
        
        print(f"\n[T] Entity Types:")
        for etype in EntityType:
            count = len(self.graph.by_type.get(etype, []))
            if count > 0:
                print(f"   {etype.name}: {count}")
        
        print(f"\n[C] Concept Mappings: {len(self.concept_map)}")
        
        print("=" * 60)


# =============================================================================
# PACKAGE INIT
# =============================================================================

__all__ = [
    "DigitalTwin",
    "WorldGraph",
    "AttentionCalculator",
    "Entity",
    "Relationship",
    "EntityType",
    "RelationType",
    "ImpactAnalysis",
    "ConceptMapping",
]


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("[*] S.P.I.D.E.R. Digital Twin - Demo")
    print("=" * 70)
    
    twin = DigitalTwin()
    
    # Add some entities manually for demo
    print("\n[1] Building world model...")
    
    auth = twin.add_entity("auth.py", EntityType.FILE, "src/auth.py", "Authentication module")
    user = twin.add_entity("user.py", EntityType.FILE, "src/user.py", "User model")
    login = twin.add_entity("login.html", EntityType.FILE, "templates/login.html", "Login page")
    test_auth = twin.add_entity("test_auth.py", EntityType.TEST, "tests/test_auth.py", "Auth tests")
    
    # Add relationships
    twin.add_relationship("login.html", "auth.py", RelationType.DEPENDS_ON)
    twin.add_relationship("auth.py", "user.py", RelationType.IMPORTS)
    twin.add_relationship("test_auth.py", "auth.py", RelationType.TESTS)
    
    print(f"   Added {len(twin.graph.entities)} entities")
    print(f"   Added {len(twin.graph.relationships)} relationships")
    
    # Analyze impact
    print("\n[2] Analyzing impact of changing auth.py...")
    impact = twin.analyze_impact("auth.py")
    
    print(f"   Risk Level: {impact.risk_level}")
    print(f"   Affected entities: {len(impact.affected_entities)}")
    for entity, score in impact.affected_entities:
        print(f"      - {entity.name} ({score:.2f})")
    for warning in impact.warnings:
        print(f"   [!] {warning}")
    
    # Map concept
    print("\n[3] Mapping concept 'User Authentication'...")
    mapping = twin.map_concept("User Authentication")
    
    print(f"   Implementations found: {len(mapping.implementations)}")
    for impl in mapping.implementations[:3]:
        print(f"      - {impl.name} ({impl.entity_type.name})")
    print(f"   Coverage: {mapping.coverage_score:.0%}")
    
    # Query oracle
    print("\n[4] Querying the oracle...")
    answer = twin.query_oracle("How many files are there?")
    print(f"   Q: How many files are there?")
    print(f"   A: {answer}")
    
    twin.print_status()
